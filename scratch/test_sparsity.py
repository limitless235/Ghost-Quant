import torch
from model.sparse_gate import apply_2_4_mask_ste

def test_sparsity():
    # Create a block of 4: [1.0, 0.1, 2.0, 0.5]
    # Smallest 2 are 0.1 and 0.5
    # Expected sparse: [1.0, 0.0, 2.0, 0.0]
    w = torch.tensor([1.0, 0.1, 2.0, 0.5], requires_grad=True)
    
    sparse_w = apply_2_4_mask_ste(w)
    print(f"Original: {w.detach().numpy()}")
    print(f"Sparse:   {sparse_w.detach().numpy()}")
    
    # Check gradients (STE)
    loss = sparse_w.sum()
    loss.backward()
    print(f"Grad:     {w.grad.numpy()}")
    
    # Verify 2 are zeroed
    assert (sparse_w == 0).sum() == 2
    # Verify grad is 1 everywhere (STE)
    assert (w.grad == 1).all()
    print("Test Passed!")

if __name__ == "__main__":
    test_sparsity()
