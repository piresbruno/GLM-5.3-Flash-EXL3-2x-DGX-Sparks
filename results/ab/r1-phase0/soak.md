# R1 Phase-0 CX7 soak (2026-08-30)

- Duration: ~25 min (operator approved >=15 min; plan said >=30)
- Path: head rocep1s0f1 (10.100.24.2) -> worker rocep1s0f0 (10.100.24.1), RC, MTU 4096, GID index 3, rdma_cm — the NCCL path
- Result: 109.25 Gb/s average single-QP write BW, 187.5M iterations, 0 fails
- Counters after: head discards/rcv_errors/link_error_recovery = 0/0/0; worker rcv_errors/discards = 0/0
- dmesg: 0 RETRY_EXC / port-error events on both nodes
- Verdict: PASS — link ruled clean, Phase 2 unblocked
