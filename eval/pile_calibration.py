import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer

def calibrate_activations(model, num_batches=100):
    device = next(model.parameters()).device
    dataset = load_dataset("EleutherAI/pile", split="train", streaming=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    variances = {}
    hooks = []

    def get_variance_hook(name):
        def hook(module, input, output):
            act = output.detach().float()
            var = torch.var(act, dim=(0, 1))
            if name not in variances:
                variances[name] = var
            else:
                variances[name] = 0.9 * variances[name] + 0.1 * var
        return hook

    for name, module in model.named_modules():
        if "mlp" in name or "attention" in name:
            hooks.append(module.register_forward_hook(get_variance_hook(name)))

    model.eval()
    count = 0
    with torch.no_grad():
        for example in dataset:
            inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, max_length=512).to(device)
            if inputs.input_ids.size(1) < 10:
                continue
            model(inputs.input_ids)
            count += 1
            if count >= num_batches:
                break

    for h in hooks:
        h.remove()
    
    return variances
