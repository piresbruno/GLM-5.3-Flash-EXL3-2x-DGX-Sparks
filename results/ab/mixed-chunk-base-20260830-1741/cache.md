# cache bench — mixed-chunk-base

- burst 4x60000 x3 rounds: rounds 2+ hit_mean_min=0.986 (PASS @ >= 0.9)
- solo 110000 replay: hit=0.969 (PASS @ >= 0.93)
- git: 7440f2f244819dc22b60c3d95bd33fc4f46e5950
- image: ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:9bb1557a4234fce63d59599e44d10747eabd742beb337eebf9e7070be8a0fd58
