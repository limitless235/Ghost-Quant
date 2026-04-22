import pulp
import torch

def solve_bit_allocation(sensitivities, budget_gb, layer_params):
    prob = pulp.LpProblem("BitAllocation", pulp.LpMinimize)
    
    layers = list(sensitivities.keys())
    bit_options = [2, 4, 8]
    
    vars = pulp.LpVariable.dicts("b", (layers, bit_options), 0, 1, pulp.LpBinary)
    
    prob += pulp.lpSum([vars[l][b] * sensitivities[l] * (8 - b) for l in layers for b in bit_options])
    
    for l in layers:
        prob += pulp.lpSum([vars[l][b] for b in bit_options]) == 1
        
    total_bits = pulp.lpSum([vars[l][b] * layer_params[l] * b for l in layers for b in bit_options])
    prob += total_bits <= budget_gb * 8 * 1024 * 1024 * 1024
    
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    allocation = {}
    if pulp.LpStatus[prob.status] == "Optimal":
        for l in layers:
            for b in bit_options:
                if pulp.value(vars[l][b]) == 1:
                    allocation[l] = b
    return allocation

def solve_matryoshka_ilp(sensitivities, budget_gb, layer_params):
    segments = ["0-95", "96-191", "192-383", "384-767"]
    probs = [0.4, 0.3, 0.2, 0.1]
    
    prob = pulp.LpProblem("MatryoshkaAllocation", pulp.LpMinimize)
    bit_options = [2, 4, 8]
    
    vars = pulp.LpVariable.dicts("m", (segments, bit_options), 0, 1, pulp.LpBinary)
    
    prob += pulp.lpSum([vars[s][b] * sensitivities.get(s, 1.0) * (8 - b) * probs[i] 
                        for i, s in enumerate(segments) for b in bit_options])
    
    for s in segments:
        prob += pulp.lpSum([vars[s][b] for b in bit_options]) == 1
        
    total_bits = pulp.lpSum([vars[s][b] * layer_params.get(s, 1000) * b for s in segments for b in bit_options])
    prob += total_bits <= budget_gb * 8 * 1024 * 1024
    
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    return {s: b for s in segments for b in bit_options if pulp.value(vars[s][b]) == 1}
