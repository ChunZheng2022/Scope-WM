# Third-Party Notices

This repository contains Scope-WM source code plus small third-party components
needed to run the experiments. Original third-party copyright, license, and
attribution notices are retained where applicable.

## Bundled Or Referenced Components

- DINOv2 encoder source is not bundled. Provide a local torch.hub-compatible
  DINOv2 source tree through `DINOV2_LOCAL_PATH=/path/to/dinov2` and
  `DINOV2_SOURCE=local`.
- D4RL PointMaze environment registration and dataset URLs follow the public
  D4RL PointMaze registration metadata.
- LPIPS utilities use the public PerceptualSimilarity weights URL:
  `https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/`.
- `metrics/image_metrics.py` retains the public GraphDeco/Inria attribution
  present in the original metric code.
- `env/deformable_env/src/sim/sim_env/transformations.py` retains its public
  BSD-style copyright notice.
- xArm URDF/assets retain their bundled BSD-style license in
  `env/deformable_env/src/sim/assets/xarm/LICENSE`.
- `models/vqvae.py` retains the Apache-2.0 license notice for the Sonnet code
  lineage.
- `models/vit.py`, `models/proprio.py`, and `datasets/traj_dset.py` include
  short comments naming the public implementation family they were adapted
  from.
