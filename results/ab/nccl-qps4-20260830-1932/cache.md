# cache bench — nccl-qps4

- burst 4x60000 x3 rounds: rounds 2+ hit_mean_min=0.986 (PASS @ >= 0.9)
- solo 110000 replay: hit=0.969 (PASS @ >= 0.93)
- git: d63846bad867b6dfb1ec333b78d7caee266e3b06
- image: ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:9bb1557a4234fce63d59599e44d10747eabd742beb337eebf9e7070be8a0fd58
