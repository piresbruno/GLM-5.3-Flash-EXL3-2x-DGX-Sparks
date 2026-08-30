# memfloor r1-auto (20260830-0702)

| Node | Floor (KiB) | Floor (GiB) |
|---|---:|---:|
| head (10.100.24.2) | 464692 | 0.44 |
| worker (10.100.24.1) | 5281028 | 5.04 |

Binding node: **head** (min floor).
**WARN: floor < 5 GiB (5242880 KiB). Do not raise KV_CACHE_MEMORY / GPU_MEM_UTIL on this state** (crash review 2026-08-29: <5 GiB risk band).

Workload: python3 tests/bench_concurrency.py --rounds 3 --levels 4 --max-tokens 128 --out results/ab/r1-auto-20260830-0633/conc.json
