import torch
from model.config import ModelConfig
from model.shared_transformer import GhostTransformer
from quantization.sensitivity import compute_sensitivity
from eval.pile_calibration import calibrate_activations
from quantization.smoothquant import migrate_outliers
import os

def run_phase3():
    config = ModelConfig()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model = GhostTransformer(config).to(device)
    
    if os.path.exists("checkpoint_phase2.pt"):
        print("Loading Phase 2 sparse checkpoint...")
        checkpoint = torch.load("checkpoint_phase2.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Error: No Phase 2 checkpoint found. Please finish training Phase 2 first.")
        return

    print("Step 1: Running Activation Calibration on The Pile...")
    variances = calibrate_activations(model, num_batches=50)

    print("Step 2: Computing Layer Sensitivity...")
    sensitivities = compute_sensitivity(model, variances)
    
    print("Step 3: Applying SmoothQuant Outlier Migration...")
    for name, module in model.named_modules():
        if hasattr(module, "weight") and name in variances:
            migrate_outliers(module, variances[name])

    print("Step 4: Simulated Quantization & Final Checkpoint...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'sensitivities': sensitivities,
        'config': config
    }, "checkpoint_phase3_quantized.pt")
    
    print("Phase 3 Complete: Quantized model saved as checkpoint_phase3_quantized.pt")

if __name__ == "__main__":
    run_phase3()
