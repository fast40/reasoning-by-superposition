import torch
import torch.profiler
import time

batch_size = 128
seq_len = 50
hidden_dim = 768
n_latent_passes = 4

device = "cuda"

t0 = None
def ts():
    global t0
    torch.cuda.synchronize()
    now = time.time()
    if t0 is None:
        t0 = now
    elapsed = now - t0
    return f"[{elapsed:.3f}s]"

# Simulate inputs_embeds (as if from an embedding layer)
param = torch.randn(hidden_dim, hidden_dim, device=device, requires_grad=True)
inputs_embeds = torch.randn(batch_size, seq_len, hidden_dim, device=device) @ param

# Simulate hidden_states output from a forward pass
hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device=device, requires_grad=True)

# Each batch element has a latent token at position 20
filling_indices = [(i, 20) for i in range(batch_size)]
hidden_states_offset = 0

# Warmup
for _ in range(3):
    x = inputs_embeds.clone()
    x.sum().backward(retain_graph=True)

torch.cuda.synchronize()

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
) as prof:
    for pass_idx in range(n_latent_passes):
        print(f"{ts()} pass {pass_idx}/{n_latent_passes} - clone")
        inputs_embeds = inputs_embeds.clone()

        print(f"{ts()} pass {pass_idx}/{n_latent_passes} - vectorized assignment")
        bi = torch.tensor([f[0] for f in filling_indices], device=device)
        ti = torch.tensor([f[1] for f in filling_indices], device=device)
        inputs_embeds[bi, ti] = hidden_states[bi, ti - 1 - hidden_states_offset]

    print(f"{ts()} forward done, starting backward")
    loss = inputs_embeds.sum()
    print(f"{ts()} inputs_embeds.sum() done")
    loss.backward()
    print(f"{ts()} backward done")

torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
