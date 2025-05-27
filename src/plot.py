import torch
import argparse
import numpy as np
from tqdm import tqdm

from html_sanitizer import Sanitizer
from scipy.stats import hmean

import pandas as pd
import matplotlib.pyplot as plt

from transformers import pipeline as lm_pipeline
from transformers import AutoTokenizer
from torch.distributions import Categorical

from utils import *


def get_generation_success(sentences, logit_lens_indices, tokenizer, k=20):
    generation_success = []
    prefixes = ["Findings show that", "I once heard that", "Then the man said:", "I believe that",
                "The news mentioned", "She saw a", "It is observed that", "Studies indicate that",
                "According to reports,", "Research suggests that", "It has been noted that", "I remember when",
                "It all started when", "The legend goes that", "If I recall correctly,", "People often say that",
                "Once upon a time,", "It’s no surprise that", "Have you ever noticed that", "I couldn't believe when",
                "The first thing I heard was", "Let me tell you a story about", "Someone once told me that",
                "It might sound strange, but", "They always warned me that", "Nobody expected that", "Funny thing is,",
                "I never thought I'd say this, but", "What surprised me most was", "The other day, I overheard that",
                "Back in the day,", "You won’t believe what happened when", "A friend of mine once said,",
                "I just found out that", "It's been a long time since", "In my experience,",
                "The craziest part was when", "If you think about it,", "I was shocked to learn that",
                "For some reason,", "I can’t help but wonder if", "It makes sense that",
                "At first, I didn't believe that", "That reminds me of the time when", "It all comes down to",
                "One time, I saw that", "I was just thinking about how", "Imagine a world where",
                "They never expected that", "I always knew that"]

    for sentence in sentences:
        sentence_ll_tokens = []
        sentence_prefix = ""
        for prefix in prefixes:
            if sentence.startswith(prefix):
                sentence_prefix = prefix
        generated_text = sentence[len(sentence_prefix):]
        tokenized_sentence = tokenizer(generated_text)["input_ids"][1:]  # both gemma and llama prepend a <bos>
        top_logit_lens_indices = logit_lens_indices[:k]
        for t in tokenized_sentence:
            if t in top_logit_lens_indices:
                sentence_ll_tokens.append(t)

        decoded_sentence_ll_tokens = [tokenizer.decode(t) for t in sentence_ll_tokens]
        generation_success.append(decoded_sentence_ll_tokens)

    generation_success_score = sum([len(t) for t in generation_success]) / len(sentences)
    return generation_success_score, generation_success


def get_axbench_generation_success(sentences, logit_lens_indices, tokenizer, k=20):
    generation_success = []
    for sentence in sentences:
        sentence_ll_tokens = []
        tokenized_sentence = tokenizer(sentence)["input_ids"][1:]  # both gemma and llama prepend a <bos>
        top_logit_lens_indices = logit_lens_indices[:k]
        for t in tokenized_sentence:
            if t in top_logit_lens_indices:
                sentence_ll_tokens.append(t)

        decoded_sentence_ll_tokens = [tokenizer.decode(t) for t in sentence_ll_tokens]
        generation_success.append(decoded_sentence_ll_tokens)

    generation_success_score = sum([len(t) for t in generation_success]) / len(sentences)
    return generation_success_score, generation_success


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default="llama3")
    parser.add_argument('--logit_lens_top_k', type=int, default=20)
    parser.add_argument('--features_file', type=str)
    parser.add_argument('--concept_name', type=str)

    parser.add_argument('--generation_cache_path', type=str)
    parser.add_argument('--perplexity_cache_path', type=str)
    parser.add_argument('--output_scores_cache_path', type=str)
    parser.add_argument('--input_scores_cache_path', type=str)
    parser.add_argument('--steering_factors_cache_path', type=str)

    parser.add_argument('--llm_concept_score_cache_path', type=str)
    parser.add_argument('--llm_fluency_score_cache_path', type=str)
    parser.add_argument('--llm_instruct_score_cache_path', type=str)

    parser.add_argument('--plot_llm_scores', type=bool, default=False)
    parser.add_argument('--late_layers', type=bool, default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    model_type = args.model_type
    device = "cuda"
    print(args)
    #
    features_by_layer = get_features_by_layers(args.features_file)
    concept_name = args.concept_name

    if model_type == "gemma2_2b":
        model_name = "google/gemma-2-2b"
        late_layer = 16
    elif model_type == "gemma2_9b":
        model_name = "google/gemma-2-9b"
        late_layer = 24
    elif model_type == "gemma2_9b_it_131":
        model_name = "google/gemma-2-9b-it"
    else:
        raise ValueError(f"Model type not supported {model_type}")

    saes = dict()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipeline = lm_pipeline("text-generation", model=model_name, device=device)
    pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id

    model = pipeline.model
    final_layer_norm = model.model.norm
    lm_head = model.lm_head

    tested_amp_factors = [0.0, 0.2, 0.4, 0.8, 1.2, 1.6, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 20.0]
    generation_caches = dict()
    perplexity_caches = dict()

    for amp_factor in tested_amp_factors:
        str_amp_factor = str(amp_factor).replace(".", "_")
        with open(f"{args.generation_cache_path.format(amp_factor=str_amp_factor)}.json", "r") as f:
            generation_caches[str_amp_factor] = json.load(f)

        with open(f"{args.perplexity_cache_path.format(amp_factor=str_amp_factor)}.json", "r") as f:
            perplexity_caches[str_amp_factor] = json.load(f)

    with open(args.output_scores_cache_path, "r") as f:
        output_scores_cache = json.load(f)

    with open(args.input_scores_cache_path, "r") as f:
        input_scores_cache = json.load(f)

    with open(args.steering_factors_cache_path, "r") as f:
        best_steering_factors_cache = json.load(f)

    with open(args.llm_concept_score_cache_path, "r") as f:
        llm_concept_score_cache = json.load(f)

    with open(args.llm_instruct_score_cache_path, "r") as f:
        llm_instruct_score_cache = json.load(f)

    with open(args.llm_fluency_score_cache_path, "r") as f:
        llm_fluency_score_cache = json.load(f)

    feature_names = []
    output_scores = []
    input_scores = []
    best_steering_factors = []
    steering_texts = []
    perplexities = []
    logit_lens_tokens = []
    generation_success_1 = []
    generation_success_5 = []
    generation_success_20 = []
    generation_success_20_tokens = []
    llm_concept_scores = []
    llm_fluency_scores = []
    llm_instruct_scores = []

    sanitizer = Sanitizer()

    for layer in sorted(list(features_by_layer.keys())):
        print(f"processing layer {layer}: features {features_by_layer[layer]}", flush=True)
        if args.late_layers and layer < late_layer:
            continue

        model = model.cpu()
        logit_lens_topk, logit_lens_confidence, logit_lens_raw_logits = cache_logit_lens(
            layer, saes, args.model_type, final_layer_norm, lm_head, args.logit_lens_top_k
        )
        torch.cuda.empty_cache()
        sae = get_sae(model_type, layer, saes).to(device)

        for feature in tqdm(features_by_layer[layer]):
            layer_feature_key = f"{layer}_{feature}"
            feature_weights_logit_lens = logit_lens_confidence[feature, :]
            feature_logit_lens_tokens_indices = logit_lens_topk.indices[feature, :].tolist()
            feature_logit_lens = [tokenizer.decode(t) for t in feature_logit_lens_tokens_indices]

            output_score = output_scores_cache.get(layer_feature_key, -1)
            input_score = input_scores_cache.get(layer_feature_key, -1)

            feature_names.append(layer_feature_key)
            logit_lens_tokens.append(feature_logit_lens)
            output_scores.append(output_score)
            input_scores.append(input_score)

            if layer_feature_key in llm_concept_score_cache:
                llm_concept_score = llm_concept_score_cache[layer_feature_key]["mean_score"]
            else:
                llm_concept_score = -1
            llm_concept_scores.append(llm_concept_score)

            if layer_feature_key in llm_fluency_score_cache:
                llm_fluency_score = llm_fluency_score_cache[layer_feature_key]["mean_score"]
            else:
                llm_fluency_score = -1
            llm_fluency_scores.append(llm_fluency_score)

            if layer_feature_key in llm_instruct_score_cache:
                llm_instruct_score = llm_instruct_score_cache[layer_feature_key]["mean_score"]
            else:
                llm_instruct_score = -1
            llm_instruct_scores.append(llm_instruct_score)

            if layer_feature_key in best_steering_factors_cache:
                optimal_amp_factor = best_steering_factors_cache[layer_feature_key]
                best_steering_factors.append(optimal_amp_factor)
                str_optimal_amp_factor = str(optimal_amp_factor).replace(".", "_")
                generations_cache = generation_caches[str_optimal_amp_factor]
                output_texts = generations_cache[layer_feature_key]

                feature_generation_success_1, _ = get_generation_success(
                    output_texts,
                    feature_logit_lens_tokens_indices,
                    tokenizer,
                    k=1
                )
                feature_generation_success_5, _ = get_generation_success(
                    output_texts,
                    feature_logit_lens_tokens_indices,
                    tokenizer,
                    k=5
                )
                feature_generation_success_20, feature_generation_success_20_list = get_generation_success(
                    output_texts,
                    feature_logit_lens_tokens_indices,
                    tokenizer,
                    k=20
                )

                perplexity_cache = perplexity_caches[str_optimal_amp_factor]
                feature_perplexity = perplexity_cache[layer_feature_key]
                feature_mean_perplexity = feature_perplexity["mean_perplexity"]
            else:
                output_texts = [""] * 100
                feature_generation_success_1 = -1
                feature_generation_success_5 = -1
                feature_generation_success_20 = -1
                feature_generation_success_20_list = [None] * 100
                feature_mean_perplexity = -1
                best_steering_factors.append(-1)

            steering_texts.append(output_texts)
            generation_success_1.append(feature_generation_success_1)
            generation_success_5.append(feature_generation_success_5)
            generation_success_20.append(feature_generation_success_20)
            generation_success_20_tokens.append(feature_generation_success_20_list)
            perplexities.append(feature_mean_perplexity)

        data = dict(
            feature=feature_names,
            layer=[int(f.split("_")[0]) for f in feature_names],
            output_score=output_scores,
            input_score=input_scores,
            logit_lens=logit_lens_tokens,
            perplexity=perplexities,
            all_texts=steering_texts,
            generation_success_1=generation_success_1,
            generation_success_5=generation_success_5,
            generation_success_20=generation_success_20,
            steering_factor=best_steering_factors,
            llm_concept_score=llm_concept_scores,
            llm_fluency_score=llm_fluency_scores,
            llm_instruct_score=llm_instruct_scores,
        )

        for i in range(10):
            attr_key = f"steering_text_{i}"
            attr_values = [sanitizer.sanitize(t[i]) for t in steering_texts]
            data[attr_key] = attr_values

            attr_key = f"gen_success_20_token_{i}"
            attr_values = [t[i] for t in generation_success_20_tokens]
            data[attr_key] = attr_values

        df = pd.DataFrame(data)
        df["entropy"] = df["entropy"].astype(float)
        df = df[df["generation_success_20"] >= 0]
        print("plotting: ", len(df), flush=True)
        df.to_csv(f"./{concept_name}_data.csv", index=False, header=True)

        # roles figure
        plt.clf()
        groupby_layers = df.groupby("layer")
        xs = df["layer"].unique()
        print(xs)
        print(groupby_layers["input_score"].quantile(0.5).values)
        plt.fill_between(xs,
                         groupby_layers["input_score"].quantile(0.75).values,
                         groupby_layers["input_score"].quantile(0.25).values,
                         interpolate=True,
                         alpha=0.2, color="blue")
        plt.plot(xs, groupby_layers["input_score"].quantile(0.5).values, marker='.', color="blue", lw=1,
                 label="Median input score")

        plt.fill_between(xs,
                         groupby_layers["output_score"].quantile(0.75).values,
                         groupby_layers["output_score"].quantile(0.25).values,
                         interpolate=True,
                         alpha=0.2, color="magenta")
        plt.plot(xs, groupby_layers["output_score"].quantile(0.5).values, marker='.', color="magenta", lw=1,
                 label="Median output score")

        plt.ylim(0, 1)
        plt.xticks(xs, fontsize=5)
        plt.title("Feature Roles Across Layers")
        plt.legend(loc="upper right")
        plt.savefig(f"./{model_type}_{concept_name}_roles.png", dpi=1000)

        # output threshold filtering figures
        output_score_ths = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9]
        mean_gen_succ_20 = []
        mean_llm_concept_score = []
        mean_llm_fluency_score = []
        mean_llm_instruct_score = []
        mean_llm_hscore = []
        n_samples = []
        for th in output_score_ths:
            to_plot = df[df["output_score"] >= th]
            mean_gen_succ_20.append(np.mean(to_plot["generation_success_20"]))
            n_samples.append(len(df) - len(to_plot))
            print(th, len(to_plot), flush=True)
            if args.plot_llm_scores:
                mean_llm_concept_score.append(np.mean(to_plot["llm_concept_score"]))
                mean_llm_fluency_score.append(np.mean(to_plot["llm_fluency_score"]))
                mean_llm_instruct_score.append(np.mean(to_plot["llm_instruct_score"]))
                hscores = [hmean([cs, fs, ins]) for cs, fs, ins in
                           zip(
                               to_plot["llm_concept_score"],
                               to_plot["llm_fluency_score"],
                               to_plot["llm_instruct_score"]
                           )]
                mean_llm_hscore.append(np.mean(hscores))
                print(th, np.mean(hscores))

        # input vs. output score figures
        plt.clf()
        plt.plot(output_score_ths, mean_gen_succ_20, marker='o', color="magenta")
        plt.xlabel("Output Score Threshold", fontsize=14)
        plt.ylabel("Mean Generation Success @ 20", fontsize=14)
        plt.yticks(np.arange(0.5, 1.6, 0.2), fontsize=14)
        plt.xscale("log")
        plt.xlim(5e-11, 1.1)
        plt.xticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1], fontsize=14)
        plt.xticks(fontsize=14)
        plt.savefig(f"./{model_type}_{concept_name}_output_score_th.png", dpi=1000)

        plt.clf()
        fig, ax = plt.subplots()
        sizes = [3 for _ in df["generation_success_20"]]
        im = ax.scatter(x=df["input_score"], y=df["output_score"], s=sizes, color="blue")
        plt.ylabel("Output Score", fontsize=14)
        plt.xlabel("Input Score", fontsize=14)
        plt.xticks(np.arange(0, 1, 0.2), fontsize=14)
        plt.yticks(np.arange(0, 1, 0.2), fontsize=14)
        plt.savefig(f"./{model_type}_{concept_name}_io_scores.png", dpi=1000)


if __name__ == "__main__":
    main()
