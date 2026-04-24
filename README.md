# PhenoMe

<p align="center">
  <img src="docs/src/assets/Logo.png" alt="PhenoMe logo" width="360"/>
</p>

A modular, **dataset-agnostic** and **model-agnostic** framework for phenotyping analysis using deep learning embeddings (**visual fingerprints**) and extracted image properties.

---

## ✨ Features

- **Model Agnostic**: Extract embeddings using DINOv2 or any custom PyTorch vision model.
- **Dataset Agnostic**: Works with any image dataset structure with customizable metadata extraction.
- **Analysis Toolkit**: Distance quantification, anomaly detection, property extraction, and correlation analysis.
- **Interactive Visualizations**: High-performance Plotly-based PCA, t-SNE, and UMAP plots.
- **Automated Reporting**: Generate comprehensive standalone HTML reports.

---

## 🚀 Quick Start

Try the no-code experience with the **[Interactive Quickstart Notebook](Notebooks/tutorials/01_interactive_quickstart.ipynb)** or launch instantly in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/01_interactive_quickstart.ipynb)

---

## 📦 Installation

```bash
git clone https://github.com/AAitorG/PhenoMe.git
cd PhenoMe
```

<details open>

<summary>conda</summary>

Pick **one** — **CPU** or **GPU**:

<details open>

<summary>CPU</summary>

```bash
# CPU-oriented env (default Conda setup)
conda env create -f envs/environment-cpu.yml
conda activate phenome-cpu
```

</details>

<details open>

<summary>GPU</summary>

```bash
# GPU-oriented env
conda env create -f envs/environment-gpu.yml
conda activate phenome-gpu
```

</details>

</details>

<details>

<summary>pip</summary>

Pick **one** — **CPU** or **GPU**:

<details open>

<summary>CPU</summary>

```bash
# CPU-oriented env
pip install -r requirements.txt -e .
```

</details>

<details open>

<summary>GPU</summary>

```bash
# GPU-oriented env
pip install -r envs/requirements-gpu.txt -e .
```

</details>

</details>

> **Full instructions, verification, and troubleshooting:**
> See the [Getting started](https://AAitorG.github.io/PhenoMe/getting-started/) guide for GPU setup details, verification commands, and first-analysis walkthroughs.

---

## 📖 Documentation

**Explore the [Documentation](https://AAitorG.github.io/PhenoMe/)** for detailed guides, API references, and tutorials.

### New users (install → data → notebooks)

| Step | Link |
|------|------|
| 1. Install | [Getting started](https://AAitorG.github.io/PhenoMe/getting-started/) |
| 2. Set up your data | [Data setup](https://AAitorG.github.io/PhenoMe/guides/data-setup/) · [Experiment details](https://AAitorG.github.io/PhenoMe/guides/experiment-details/) |
| 3. Tutorials | [Notebooks/tutorials/](Notebooks/tutorials/) (start with **01**, then **02**–**03**) · [Learning paths](https://AAitorG.github.io/PhenoMe/user-paths/) |

**Concepts and examples:** [Core concepts](https://AAitorG.github.io/PhenoMe/concepts/) · [Workflows](https://AAitorG.github.io/PhenoMe/workflows/) · [Examples](https://AAitorG.github.io/PhenoMe/examples/) · [FAQ](https://AAitorG.github.io/PhenoMe/faq/) · [Glossary](https://AAitorG.github.io/PhenoMe/glossary/)

---

## Author

**Aitor González-Marfil** — [@AAitorG](https://github.com/AAitorG)

## Citation

```bibtex
@software{phenome,
  author = {Aitor González-Marfil},
  title = {PhenoMe: Model-Agnostic Deep Learning-Based Phenotyping Analysis},
  year = {2026},
  url = {https://github.com/AAitorG/PhenoMe}
}
```
