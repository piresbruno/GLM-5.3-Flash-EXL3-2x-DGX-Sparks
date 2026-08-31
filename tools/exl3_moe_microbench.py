import json, glob, os, time
import torch
from safetensors import safe_open

SNAP = "/root/.cache/huggingface/hub/models--brandonmusic--GLM-5.3-Flash-EXL3-4bpw/snapshots/a38a2eeb511374161f37eead45fd8beaa8d5374d"
LAYER = 15
N_EXPERTS = 288
TOPK = 8
HIDDEN = 4096
INTER = 2048
K_BITS = 4
REAL = 32            # real experts loaded; pointers cycled to fill 288 slots
TEMP_ROWS = 128
MOE_ACT_SILU = 0
LAYER_IDX_NAME = f"model.language_model.layers.{LAYER}.mlp.experts"

idx = json.load(open(f"{SNAP}/model.safetensors.index.json"))["weight_map"]
# layer count from the weight map
maxl = -1
for k in idx:
    if ".mlp.experts." in k:
        try:
            maxl = max(maxl, int(k.split(".layers.")[1].split(".mlp")[0]))
        except Exception:
            pass
N_LAYERS = maxl + 1
print(f"num_moe_layers(0-based count incl dense) = {N_LAYERS}")

# which tensors to pull (REAL experts)
wanted = {}
for e in range(REAL):
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for sfx in ("trellis", "suh", "svh", "mcg"):
            key = f"{LAYER_IDX_NAME}.{e}.{proj}.{sfx}"
            wanted[key] = idx[key]
by_file = {}
for k, f in wanted.items():
    by_file.setdefault(f, []).append(k)

tensors = {}
t_load = time.time()
for f, keys in by_file.items():
    with safe_open(f"{SNAP}/{f}", "pt") as sf:
        for k in keys:
            tensors[k] = sf.get_tensor(k).to("cuda", non_blocking=True)
torch.cuda.synchronize()
print(f"loaded {len(tensors)} tensors in {time.time()-t_load:.1f}s "
      f"({sum(t.numel()*t.element_size() for t in tensors.values())/1e9:.2f} GB)")

import exllamav3_ext

def ptr_table(prefix_proj, sfx):
    out = []
    for e in range(N_EXPERTS):
        src = tensors[f"{LAYER_IDX_NAME}.{e % REAL}.{prefix_proj}.{sfx}"]
        out.append(src)
    return out

def ptrs(ts):
    return torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device="cuda")

args_ptrs = []
for proj, sfx in (("gate_proj","trellis"),("gate_proj","suh"),("gate_proj","svh"),
                  ("up_proj","trellis"),("up_proj","suh"),("up_proj","svh"),
                  ("down_proj","trellis"),("down_proj","suh"),("down_proj","svh")):
    args_ptrs.append(ptrs(ptr_table(proj, sfx)))

def ptrs(ts):
    return torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device="cuda")

concurrency = int(exllamav3_ext.exl3_moe_max_concurrency(0))
temps = (
    torch.empty((concurrency, TEMP_ROWS, HIDDEN), dtype=torch.float16, device="cuda"),
    torch.empty((concurrency, TEMP_ROWS, HIDDEN), dtype=torch.float16, device="cuda"),
    torch.empty((concurrency, TEMP_ROWS, INTER), dtype=torch.float16, device="cuda"),
    torch.empty((concurrency, TEMP_ROWS, INTER), dtype=torch.float16, device="cuda"),
)
print(f"concurrency={concurrency}")

gen = torch.Generator(device="cuda").manual_seed(7)

def run_case(M, iters=12, warmup=3):
    x2d = torch.randn(M, HIDDEN, dtype=torch.float16, device="cuda", generator=gen)
    ids = torch.randint(0, N_EXPERTS, (M, TOPK), device="cuda", generator=gen)
    w = torch.softmax(torch.randn(M, TOPK, device="cuda", generator=gen), dim=-1)
    local = ids  # single GPU: local == global
    flat_token = torch.arange(M, device="cuda", dtype=torch.long).repeat_interleave(TOPK)
    flat_weight = w.reshape(-1).to(dtype=torch.float16)
    order = local.flatten().argsort()
    token_sorted = flat_token[order]
    weight_sorted = flat_weight[order]
    expert_count = torch.zeros(N_EXPERTS + 1, dtype=torch.long, device="cuda")
    expert_count.scatter_add_(0, local.flatten().long(),
                              torch.ones(local.numel(), dtype=torch.long, device="cuda"))
    out = torch.zeros(M, HIDDEN, dtype=torch.float32, device="cuda")
    xh = x2d.contiguous().half()
    fn = exllamav3_ext.exl3_moe
    args = (xh, out, expert_count, token_sorted, weight_sorted,
            temps[0], temps[1], temps[2], temps[3],
            MOE_ACT_SILU, K_BITS, K_BITS, K_BITS,
            *args_ptrs, True, False, True, False, True, False, 30.0)
    times = []
    for it in range(iters + warmup):
        # re-randomize routing per iter (uniform load like serving)
        if it >= warmup:
            ids = torch.randint(0, N_EXPERTS, (M, TOPK), device="cuda", generator=gen)
            w = torch.softmax(torch.randn(M, TOPK, device="cuda", generator=gen), dim=-1)
            local = ids
            flat_token = torch.arange(M, device="cuda", dtype=torch.long).repeat_interleave(TOPK)
            flat_weight = w.reshape(-1).to(dtype=torch.float16)
            order = local.flatten().argsort()
            token_sorted = flat_token[order]; weight_sorted = flat_weight[order]
            expert_count.zero_()
            expert_count.scatter_add_(0, local.flatten().long(),
                                      torch.ones(local.numel(), dtype=torch.long, device="cuda"))
            args = (xh, out, expert_count, token_sorted, weight_sorted,
                    temps[0], temps[1], temps[2], temps[3],
                    MOE_ACT_SILU, K_BITS, K_BITS, K_BITS,
                    *args_ptrs, True, False, True, False, True, False, 30.0)
        torch.cuda.synchronize(); t0 = time.time()
        if hasattr(fn, "__wrapped__") or True:
            try:
                fn(*args, -1)
            except Exception:
                fn(*args)
        torch.cuda.synchronize()
        if it >= warmup:
            times.append(time.time() - t0)
    med = sorted(times)[len(times)//2]
    return med

for M in (1024, 1792, 3584):
    med = run_case(M)
    n_moe = sum(1 for e in range(N_EXPERTS) for _ in (0,))  # placeholder
    print(f"MICRORESULT M={M} median={med*1000:.1f} ms "
          f"tok/s={M/med:.0f} per-layer={med*1000:.1f} ms")

print("DONE")
