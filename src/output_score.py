import gc
import argparse
import os.path

from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import *
from sae_utils import AmlifySAEHook

def get_output_score(
        layer, feature, logit_lens_indices,
        sentence, sae, tokenizer,
        model, device, amp_factor=10
):
    model = model.to(device)
    sae = sae.to(device)
    sae_hook = AmlifySAEHook(layer, sae, [feature], amp_factor, device)
    model_block_to_hook = model.model.layers[layer]
    handle = model_block_to_hook.register_forward_hook(sae_hook, always_call=True)

    inputs = tokenizer(sentence, return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.to(model.device)

    outputs = model(**inputs)

    for k, v in inputs.items():
        inputs[k] = v.cpu()
    handle.remove()
    for hook in model_block_to_hook._forward_hooks.values():
        hook.remove()

    logits_after = outputs.logits[:, -1]
    intervention_logits = logits_after.squeeze()
    intervention_probs = torch.softmax(intervention_logits, dim=0).detach().cpu()

    vocab_size = intervention_probs.shape[0]
    tokens_argsort = torch.argsort(intervention_probs, dim=0, descending=True)
    ll_tokens_ranks = [(tokens_argsort == ll_token).nonzero(as_tuple=True)[0].item() for ll_token in
                       logit_lens_indices]
    top_token_score = torch.max(intervention_probs[logit_lens_indices]).item()
    rank_output_score = 1 - (min(ll_tokens_ranks) / vocab_size)

    model = model.cpu()
    sae = sae.cpu()
    torch.cuda.empty_cache()
    gc.collect()

    return rank_output_score * top_token_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default="llama3")
    parser.add_argument('--logit_lens_top_k', type=int, default=20)
    parser.add_argument('--features_file', type=str)
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
    neutral_sentence = "From my experience,"

    if os.path.exists(args.cache_path):
        with open(args.cache_path, "r") as f:
            output_scores = json.load(f)
    else:
        output_scores = dict()

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
            logit_lens_tokens_indices = logit_lens_topk.indices[feature, :].tolist()
            output_score = get_output_score(
                layer, feature, logit_lens_tokens_indices, neutral_sentence,
                sae, tokenizer,
                model, device
            )
            torch.cuda.empty_cache()
            gc.collect()
            output_scores[layer_feature_key] = output_score

        with open(args.cache_path, "w") as f:
            json.dump(output_scores, f)


if __name__ == "__main__":
    main()
