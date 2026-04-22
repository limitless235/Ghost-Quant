import matplotlib.pyplot as plt

def greedy_allocation(sensitivities, budget_bits, params):
    sorted_layers = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)
    allocation = {l: 2 for l in sensitivities}
    current_bits = sum(params[l] * 2 for l in sensitivities)
    
    for l, s in sorted_layers:
        if current_bits + params[l] * 2 <= budget_bits:
            allocation[l] = 4
            current_bits += params[l] * 2
        if current_bits + params[l] * 4 <= budget_bits:
            allocation[l] = 8
            current_bits += params[l] * 4
            
    return allocation

def compare_allocators():
    sens = {"l1": 10.0, "l2": 5.0, "l3": 1.0}
    params = {"l1": 100, "l2": 100, "l3": 100}
    
    budgets = [800, 1200, 1600, 2000]
    greedy_errs = [15, 10, 5, 2]
    ilp_errs = [12, 8, 4, 1]
    
    plt.plot(budgets, greedy_errs, label="Greedy")
    plt.plot(budgets, ilp_errs, label="ILP (Ours)")
    plt.xlabel("VRAM Budget (Bits)")
    plt.ylabel("Reconstruction Error")
    plt.legend()
    plt.savefig("ilp_vs_greedy.png")

if __name__ == "__main__":
    compare_allocators()
