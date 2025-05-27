import gc
import os
import torch
import argparse
from tqdm import tqdm

from transformers import pipeline as lm_pipeline
from transformers import AutoTokenizer

from utils import *
from sae_utils import init_hook


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default="gemma2_2b")
    parser.add_argument('--logit_lens_top_k', type=int, default=20)
    parser.add_argument('--features_file', type=str)
    parser.add_argument('--cache_path', type=str)
    parser.add_argument('--amp_factor', type=float, default=1.2)
    return parser.parse_args()


def main():
    args = parse_args()
    model_type = args.model_type
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
    pipeline = lm_pipeline("text-generation", model=model_name, device=device)
    pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id
    model = pipeline.model
    final_layer_norm = model.model.norm
    lm_head = model.lm_head
    print(final_layer_norm.weight.shape, lm_head.weight.shape)

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

    saes = dict()
    if os.path.exists(args.cache_path):
        with open(args.cache_path, "r") as f:
            generation_cache = json.load(f)
    else:
        generation_cache = dict()

    for layer in features_by_layers:
        features = features_by_layers[layer]
        model = model.to(device)
        for feature in tqdm(features):
            layer_feature_key = f"{layer}_{feature}"
            sae = get_sae(model_type, layer, saes).to(device)
            handle = init_hook(pipeline, sae, layer, feature, device, args)

            output_texts = []
            for prefix in prefixes:
                with torch.no_grad():
                    outputs = pipeline(
                        prefix,
                        do_sample=True,
                        temperature=0.7,
                        max_new_tokens=20,
                        stop_strings=".",
                        tokenizer=tokenizer,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    output_text = outputs[0]["generated_text"]
                    output_texts.append(output_text)

            handle.remove()
            for k, v in saes.items():
                saes[k] = v.cpu()
            torch.cuda.empty_cache()
            gc.collect()

            generation_cache[layer_feature_key] = output_texts
            with open(args.cache_path, "w") as f:
                json.dump(generation_cache, f)


if __name__ == "__main__":
    main()
