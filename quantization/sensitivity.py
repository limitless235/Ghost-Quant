import torch

def get_spectral_norm(weight, num_iters=3):
    if weight.dim() < 2:
        return weight.abs().max()
    
    weight_mat = weight.view(weight.size(0), -1)
    rows, cols = weight_mat.size()
    
    u = torch.randn(rows, 1, device=weight.device)
    v = torch.randn(cols, 1, device=weight.device)
    
    for _ in range(num_iters):
        v = torch.matmul(weight_mat.t(), u)
        v = v / torch.norm(v)
        u = torch.matmul(weight_mat, v)
        u = u / torch.norm(u)
        
    sigma = torch.matmul(torch.matmul(u.t(), weight_mat), v)
    return sigma.item()

def compute_sensitivity(model, activation_variances):
    sensitivities = {}
    for name, module in model.named_modules():
        if hasattr(module, "weight") and name in activation_variances:
            w = module.weight.data
            sigma = get_spectral_norm(w)
            n = w.numel()
            var_a = activation_variances[name].mean().item()
            sensitivities[name] = sigma * n * var_a
            
    return sensitivities
