import matplotlib.pyplot as plt

def plot_pareto(results, save_path="pareto_curve.png"):
    accs = [r['accuracy'] for r in results]
    flops = [r['flops'] for r in results]
    names = [r['name'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(flops, accs, c='blue', marker='o')
    
    for i, name in enumerate(names):
        plt.annotate(name, (flops[i], accs[i]))
        
    plt.xlabel("FLOPs per Sample")
    plt.ylabel("Accuracy / Perplexity Score")
    plt.title("Ghost-Quant Pareto Curve: Accuracy vs Compute")
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    dummy_results = [
        {'name': 'FP32-768', 'accuracy': 0.85, 'flops': 1000},
        {'name': 'INT8-768', 'accuracy': 0.84, 'flops': 500},
        {'name': 'INT4-384', 'accuracy': 0.78, 'flops': 250},
        {'name': 'INT2-96', 'accuracy': 0.65, 'flops': 100}
    ]
    plot_pareto(dummy_results)
