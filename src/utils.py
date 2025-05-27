import json
import torch

from sae_lens import SAE


def get_gemma2_sae(model_size, instruct, layer, saes, width="16k"):
    if layer in saes:
        sae = saes[layer]
    else:
        release = f"gemma-scope-{model_size}-{'it' if instruct else 'pt'}-res-canonical"
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release=release,
            sae_id=f"layer_{layer}/width_{width}/canonical",
        )
        print(sae, cfg_dict, sparsity)
        saes[layer] = sae
    return sae


def get_sae(model_type, layer, saes):
    if model_type == "gemma2_2b":
        sae = get_gemma2_sae("2b", False, layer, saes)
    elif model_type == "gemma2_9b":
        sae = get_gemma2_sae("9b", False, layer, saes)
    elif model_type == "gemma2_it":
        sae = get_gemma2_sae("2b", True, layer, saes)
    elif model_type == "gemma2_9b_it":
        sae = get_gemma2_sae("9b", True, layer, saes)
    elif model_type == "gemma2_9b_it_131":
        sae = get_gemma2_sae("9b", True, layer, saes, width="131k")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return sae


def get_features_by_layers(features_file):
    with open(features_file, "r") as f:
        features_by_layer = json.load(f)
        features_by_layer = {int(key): [int(v) for v in values] for key, values in features_by_layer.items()}
        return features_by_layer


def cache_logit_lens(layer, saes, model_type, final_layer_norm, lm_head, k):
    sae = get_sae(model_type, layer, saes)
    final_layer_norm = final_layer_norm.cpu()
    lm_head = lm_head.cpu()

    decoder_weights = sae.W_dec.cpu()
    print(decoder_weights.shape)
    decoder_weights = final_layer_norm(decoder_weights)
    logits = lm_head(decoder_weights)
    confidence = torch.softmax(logits, dim=1).detach().cpu()

    topk = torch.topk(confidence, dim=1, k=k)
    final_layer_norm = final_layer_norm.cpu()
    lm_head = lm_head.cpu()
    return topk, confidence, logits
