import torch

class EpsilonTracker:
    def __init__(self, model_fp32, model_quant):
        self.model_fp32 = model_fp32
        self.model_quant = model_quant
        self.errors = {}
        self.hooks = []

    def _get_hook(self, name):
        def hook(module, input, output):
            self.errors[name] = output.detach()
        return hook

    def register(self):
        for name, module in self.model_fp32.named_modules():
            if hasattr(module, "weight"):
                self.hooks.append(module.register_forward_hook(self._get_hook(f"{name}_fp32")))
        for name, module in self.model_quant.named_modules():
            if hasattr(module, "weight"):
                self.hooks.append(module.register_forward_hook(self._get_hook(f"{name}_quant")))

    def compute_layer_error(self, input_ids):
        self.errors = {}
        with torch.no_grad():
            self.model_fp32(input_ids)
            self.model_quant(input_ids)
        
        results = {}
        for key in self.errors:
            if "_fp32" in key:
                base_name = key.replace("_fp32", "")
                quant_key = f"{base_name}_quant"
                if quant_key in self.errors:
                    diff = self.errors[key] - self.errors[quant_key]
                    results[base_name] = torch.norm(diff, p=2).item()
        return results

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
