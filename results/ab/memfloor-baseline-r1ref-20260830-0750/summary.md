# memfloor baseline-r1ref (20260830-0750)

| Node | Floor (KiB) | Floor (GiB) |
|---|---:|---:|
| head (10.100.24.2) | 1208488 | 1.15 |
| worker (10.100.24.1) | 5071000 | 4.84 |

Binding node: **head** (min floor).
**WARN: floor < 5 GiB (5242880 KiB). Do not raise KV_CACHE_MEMORY / GPU_MEM_UTIL on this state** (crash review 2026-08-29: <5 GiB risk band).

Workload: python3 tests/bench_concurrency.py --rounds 3 --levels 4 --max-tokens 128 --out results/ab/baseline-r1ref-20260830-0715/conc.json
