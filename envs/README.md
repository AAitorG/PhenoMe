# Environments

**Python 3.12+** (see `pyproject.toml` `requires-python`). Run Conda and pip commands from the **repository root** (`PhenoMe/`).

## Conda

```bash
conda env create -f envs/environment-cpu.yml && conda activate phenome-cpu
# NVIDIA CUDA (Linux / Windows):
conda env create -f envs/environment-gpu.yml && conda activate phenome-gpu
```

## pip

```bash
# CPU stack:
pip install -r envs/requirements-cpu.txt -e .
# NVIDIA CUDA (Linux / Windows) — includes GPU-acceleration libs:
pip install -r envs/requirements-gpu.txt -e .
```

From the repo root, [`requirements.txt`](../requirements.txt) aliases `envs/requirements-cpu.txt`.

Optional stack is also declared in `pyproject.toml` as `[project.optional-dependencies] full`.

On **Windows**, `pykeops` is omitted via environment markers.
