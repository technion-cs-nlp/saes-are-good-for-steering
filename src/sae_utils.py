import torch
from torch import nn
from contextlib import contextmanager
from accelerate.hooks import ModelHook, add_hook_to_module
from sae_lens import SAE

@contextmanager
def _disable_hooks(sae: SAE):
    """
    Temporarily disable hooks for the SAE. Swaps out all the hooks with a fake modules that does nothing.
    """
    try:
        for hook_name in sae.hook_dict:
            _blank_hook = nn.Identity()
            setattr(sae, hook_name, _blank_hook)
        yield
    finally:
        for hook_name, hook in sae.hook_dict.items():
            setattr(sae, hook_name, hook)


class AmlifySAEHook(ModelHook):
    def __init__(self, layer, sae, features, amp_factor, device) -> None:
        super().__init__()
        self.amp_factor = amp_factor
        self.sae = sae
        self.device = device
        self.layer = layer
        self.features = features

    def __call__(self, module, args, output):
        output_tensor = output[0]
        _, n_tokens, _ = output_tensor.shape

        # encode with SAE
        feature_acts = self.sae.encode(output_tensor).to(self.device)

        with torch.no_grad():
            with _disable_hooks(self.sae):
                feature_acts_clean = self.sae.encode(output_tensor)
                x_reconstruct_clean = self.sae.decode(feature_acts_clean)
            sae_error = self.sae.hook_sae_error(output_tensor.to(torch.float64) - x_reconstruct_clean.to(torch.float64))

        max_act_value = torch.max(feature_acts[:, -1, :]).item()
        for feature in self.features:
            feature_acts[:, -1, feature] += max_act_value * self.amp_factor

        sae_out = self.sae.decode(feature_acts)
        sae_out = sae_out + sae_error
        sae_out = sae_out.to(torch.float32)
        return tuple([sae_out] + list(output[1:]))


def init_hook(pipeline, sae, layer, feature, device, args):
    sae_hook = AmlifySAEHook(layer, sae, [feature], args.amp_factor, device)
    model_block_to_hook = pipeline.model.model.layers[layer]
    handle = model_block_to_hook.register_forward_hook(sae_hook, always_call=True)
    return handle
