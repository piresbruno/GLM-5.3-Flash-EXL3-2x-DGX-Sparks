# CX7 inter-Spark link soak — 2026-08-30 20:17 (30 min)

Standing ops check (RESEARCH-PERF-NEXT "Ops finding": fleet died 2026-08-29
19:46 on `IBV_WC_RETRY_EXC_ERR` — marginal DAC link suspected). This soak
validates the link **as-is**; it does not replace the physical reseat
directive — the cable is still the same one that flaked under sustained load.

## Result: CLEAN

- Method: `ib_write_bw` head(`rocep1s0f1`) → worker(`rocep1s0f0`), RDMA
  writes, 65536-byte messages, 1800 s duration.
- **BW average 107.37 Gb/s** sustained over 1,843,180 iterations
  (record: 109.25 Gb/s from the phase-0 soak — within ~1.7%, normal).
- **0 errors / 0 retry events / 0 disconnections** in the client log.
- GPU clocks during soak: 2190 MHz SM (operator underclock) — unaffected.

## Verdict

Link currently healthy at full line rate. The 2026-08-29 flap was either
transient or load/thermal-specific; per the standing directive, reseat /
swap the QSFP DAC at the next maintenance window and re-run this soak
(`ib_write_bw -D 1800`) + an `ib_write_lat` pass before the next heavy
campaign. Until then, the link is certified for normal serving and gated
A/B work (this day's campaign ran with 0 transport failures).

Files: `server.log` (worker side), `client-soak.log` (head side, full 30-min
table).
