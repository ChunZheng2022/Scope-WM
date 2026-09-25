# PyFleX Setup Notes For Rope And Granular

The rigid-task environment in `environment.yml` intentionally does not include
PyFleX. Rope and Granular require a local PyFleX build.

Install the extra Python packages used by the deformable environment wrapper:

```bash
pip install pybullet==3.2.7 beautifulsoup4==4.13.3
```

Reported deformable experiments used:

- 1 x NVIDIA RTX 2080 Ti
- Ubuntu 22.04.3
- Python 3.12.3 for the host system
- PyTorch 2.3.0+cu121
- CUDA runtime 12.1
- cuDNN 8.9.2
- PyFleX source build using a CUDA 9.2 toolkit-compatible build environment

Use local placeholders:

```bash
export PYFLEXROOT=/path/to/PyFleX
export PYTHONPATH=$PYFLEXROOT/bindings/build:$PYTHONPATH
export LD_LIBRARY_PATH=/path/to/cuda92/lib:$PYFLEXROOT/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH
export LD_PRELOAD=$PYFLEXROOT/bindings/build/libfinite_math_compat.so
```

If your PyFleX build does not need a finite-math compatibility shim, omit
`LD_PRELOAD`. The shim is a local compatibility workaround for older CUDA/PyFleX
symbols and is not part of the Scope-WM algorithm.
