import os
import gc

import argparse
from tqdm import tqdm

import torch
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default="llama3")
    parser.add_argument('--logit_lens_top_k', type=int, default=20)
    parser.add_argument('--features_file', type=str)
    parser.add_argument('--feature_data_path', type=str)
    parser.add_argument('--cache_path', type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    model_type = args.model_type
    logit_lens_k = args.logit_lens_top_k
    device = "cuda:0"
    print(args)

    if model_type == "gemma2_2b":
        model_name = "google/gemma-2-2b"
    elif model_type == "gemma2_9b":
        model_name = "google/gemma-2-9b"
    elif model_type == "gemma2_9b_it_131":
        model_name = "google/gemma-2-9b-it"
    else:
        raise ValueError(f"Model type not supported {model_type}")

    features_by_layers = get_features_by_layers(args.features_file)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    final_layer_norm = model.model.norm
    lm_head = model.lm_head
    print(final_layer_norm.weight.shape, lm_head.weight.shape)

    saes = dict()

    if os.path.exists(args.cache_path):
        with open(args.cache_path, "r") as f:
            input_scores = json.load(f)
    else:
        input_scores = dict()

    for layer in features_by_layers:
        features = features_by_layers[layer]
        sae = get_sae(model_type, layer, saes)
        model = model.cpu()
        logit_lens_topk, logit_lens_confidence, logit_lens_raw_logits = cache_logit_lens(layer, saes, model_type,
                                                                                         final_layer_norm, lm_head,
                                                                                         logit_lens_k)

        model = model.to(device)
        for feature in tqdm(features):
            layer_feature_key = f"{layer}_{feature}"
            feature = int(feature)
            feature_data_path = f"{args.feature_data_path}/{layer}_{feature}.json"

            if os.path.exists(feature_data_path):
                with open(feature_data_path, "r") as f:
                    feature_data = json.load(f)
            else:
                print(f"file not found: {feature_data_path}")
                continue

            logit_lens_tokens_indices = logit_lens_topk.indices[feature, :].tolist()
            import time
            t0 = time.time()
            input_score = 0
            activated_sentences = feature_data["activations"][:100]
            if len(activated_sentences) < 10:
                continue
            for activation_data in activated_sentences:
                max_act_index = np.argmax(activation_data["values"])
                max_act_token = activation_data["tokens"][max_act_index].replace("_", " ")
                max_act_token_index = tokenizer(max_act_token)["input_ids"][1]

                tokens_in_logit_lens = 1 if max_act_token_index in logit_lens_tokens_indices else 0
                input_score += tokens_in_logit_lens

            input_score /= len(activated_sentences)
            input_scores[layer_feature_key] = input_score
            t1 = time.time()
            print(f"time: {t0 - t1:.3f}s")
            torch.cuda.empty_cache()
            gc.collect()

        with open(args.cache_path, "w") as f:
            json.dump(input_scores, f)


if __name__ == "__main__":
    main()
