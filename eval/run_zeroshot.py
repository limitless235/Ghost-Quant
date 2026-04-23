import torch
import json
import lm_eval
from transformers import GPT2Tokenizer
from model.config import ModelConfig
from model.shared_transformer import GhostTransformer
from eval.eval_harness_wrapper import GhostQuantLM

def run_benchmarks():
    config = ModelConfig()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    model = GhostTransformer(config)
    checkpoint = torch.load("checkpoint_phase3_quantized.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    slices = [96, 384, 768]
    tasks = ["hellaswag", "piqa", "winogrande"]
    final_results = {}

    for m in slices:
        print(f"Evaluating Matryoshka Slice M={m}...")
        lm_obj = GhostQuantLM(model, tokenizer, M=m, device=device)
        
        results = lm_eval.simple_evaluate(
            model=lm_obj,
            tasks=tasks,
            num_fewshot=0,
            batch_size=1
        )
        
        task_accs = []
        for task in tasks:
            task_res = results['results'][task]
            # Handle different versions of lm-eval keys
            acc = task_res.get('acc,none', task_res.get('acc_norm,none', task_res.get('acc', 0.0)))
            task_accs.append(acc)
            
        avg_acc = sum(task_accs) / len(tasks)
        final_results[f"M_{m}"] = {
            "tasks": results['results'],
            "average_accuracy": avg_acc
        }

    with open("benchmark_results.json", "w") as f:
        json.dump(final_results, f, indent=4)
    
    print("Zero-shot evaluation complete. Results saved to benchmark_results.json")

if __name__ == "__main__":
    run_benchmarks()
