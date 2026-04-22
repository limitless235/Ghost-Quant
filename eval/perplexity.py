import torch
import math
from datasets import load_dataset
from transformers import GPT2Tokenizer

def calculate_perplexity(model, config):
    device = torch.device(config.device)
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i, example in enumerate(dataset):
            if i >= 100: break
            tokens = tokenizer(example['text'], return_tensors="pt", truncation=True, max_length=config.max_seq_len).to(device)
            input_ids = tokens.input_ids
            targets = input_ids.clone()
            
            outputs = model(input_ids)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = targets[:, 1:].contiguous()
            
            loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            total_loss += loss.item() * shift_labels.numel()
            total_tokens += shift_labels.numel()
            
    return math.exp(total_loss / total_tokens)
