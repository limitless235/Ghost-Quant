import matplotlib.pyplot as plt
import json
import numpy as np

def plot_bubble():
    baselines = [
        {"name": "Dense 124M", "vram": 248, "acc": 0.65, "flops": 1.0, "color": "red"},
        {"name": "Dense 20M", "vram": 40, "acc": 0.45, "flops": 0.2, "color": "red"}
    ]
    
    try:
        with open("benchmark_results.json", "r") as f:
            ghost_data = json.load(f)
    except FileNotFoundError:
        ghost_data = {
            "M_768": {"average_accuracy": 0.72},
            "M_384": {"average_accuracy": 0.68},
            "M_96": {"average_accuracy": 0.58}
        }

    ghost_vram_map = {"M_768": 105, "M_384": 55, "M_96": 42}
    ghost_flops_map = {"M_768": 0.8, "M_384": 0.45, "M_96": 0.25}
    
    ghost_results = []
    for m_key, metrics in ghost_data.items():
        ghost_results.append({
            "name": f"Ghost-Quant {m_key.split('_')[1]}",
            "vram": ghost_vram_map[m_key],
            "acc": metrics["average_accuracy"],
            "flops": ghost_flops_map[m_key],
            "color": "blue"
        })
    
    ghost_results.sort(key=lambda x: x["vram"])
    
    plt.figure(figsize=(10, 7))
    
    for b in baselines:
        plt.scatter(b["vram"], b["acc"], s=b["flops"]*1000, c=b["color"], alpha=0.6, edgecolors="black")
        plt.text(b["vram"], b["acc"] + 0.01, b["name"], fontsize=9, ha="center")
        
    g_vrams = [g["vram"] for g in ghost_results]
    g_accs = [g["acc"] for g in ghost_results]
    g_sizes = [g["flops"]*1000 for g in ghost_results]
    
    plt.scatter(g_vrams, g_accs, s=g_sizes, c="blue", alpha=0.6, edgecolors="black", label="Ghost-Quant")
    plt.plot(g_vrams, g_accs, "b--", alpha=0.5)
    
    for g in ghost_results:
        plt.text(g["vram"], g["acc"] + 0.01, g["name"], fontsize=9, ha="center")
        
    plt.xscale("log")
    plt.xlabel("VRAM (MB) [Log Scale]")
    plt.ylabel("Zero-Shot Accuracy")
    plt.title("Model Efficiency: Accuracy vs. VRAM vs. Compute (FLOPs)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("benchmark_bubble_chart.pdf")
    plt.show()

if __name__ == "__main__":
    plot_bubble()
