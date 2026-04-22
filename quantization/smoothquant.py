import torch

def migrate_outliers(module, act_scales, alpha=0.5):
    if not hasattr(module, "weight"):
        return
    
    device = module.weight.device
    act_scales = act_scales.to(device)
    
    weight_scales = module.weight.abs().max(dim=0)[0]
    
    scales = (act_scales.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-5)
    
    module.weight.data.mul_(scales.view(1, -1))
    
    return scales
