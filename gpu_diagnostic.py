import torch
import traceback
from torch.nn.functional import scaled_dot_product_attention

print("=" * 60)
print("🔍 GPU DIAGNOSTIC TEST")
print("=" * 60)

# Step 1: Basic CUDA check
print("\n[1] Basic CUDA test:")
try:
    x = torch.randn(1, device='cuda')
    print("✅ torch.randn on CUDA succeeded")
except Exception as e:
    print("❌ Basic CUDA failed:")
    traceback.print_exc()

# Step 2: Memory-efficient attention (used by xFormers)
print("\n[2] xFormers-like Attention test (scaled_dot_product_attention):")
try:
    q = torch.randn(2, 8, 64, device='cuda')  # [batch, seq_len, dim]
    k = torch.randn(2, 8, 64, device='cuda')
    v = torch.randn(2, 8, 64, device='cuda')
    out = scaled_dot_product_attention(q, k, v, is_causal=False)
    torch.cuda.synchronize()
    print("✅ scaled_dot_product_attention succeeded")
except Exception as e:
    print("❌ scaled_dot_product_attention FAILED:")
    traceback.print_exc()

# Step 3: Show CUDA environment
print("\n[3] CUDA Environment:")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
if torch.cuda.is_available():
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"Device name: {props.name}")
    print(f"Memory total: {props.total_memory // (1024**2)} MiB")

print("=" * 60)
print("Done.")
