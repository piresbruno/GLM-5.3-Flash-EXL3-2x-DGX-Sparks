# memfloor d1-2048 (20260829-1847)

| Node | Floor (KiB) | Floor (GiB) |
|---|---:|---:|
| head (10.100.24.2) | 551988 | 0.53 |
| worker (10.100.24.1) | 4528612 | 4.32 |

Binding node: **head** (min floor).
**WARN: floor < 5 GiB (5242880 KiB). Do not raise KV_CACHE_MEMORY / GPU_MEM_UTIL on this state** (crash review 2026-08-29: <5 GiB risk band).

Workload: python3 tests/bench_concurrency.py --levels 1,2,4 --rounds 2 --max-tokens 120 --temperature 0 --long-prompt-tokens 15000 --out /tmp/sat-d12048.json
