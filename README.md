<div align="center">
  <img src="docs/assets/logo.png" alt="PhenoMe logo" width="360"/>
  <p>A modular, <b>dataset-agnostic</b> and <b>model-agnostic</b> framework for phenotyping analysis, combining deep learning embeddings and image properties.</p>
</div>

---

### 🗺️ Navigation

**[✨ Features](#-features)** | **[🚀 Quick Start](#-quick-start)** | **[📦 Installation](#-installation)** | **[📖 Documentation](https://AAitorG.github.io/PhenoMe/)** | **[📜 Citation](#-citation)**

---


<div align="center">
    <video src="https://github.com/user-attachments/assets/3d8173fc-9a7e-4724-bac1-686412f54a77"></video>
</div>


---

## ✨ Features

- 🧠 **Model Agnostic**: Extract embeddings using DINOv2 or any custom vision model (PyTorch, TensorFlow, Keras, etc.) via the modular `ModelWrapper` interface.
- 📂 **Dataset Agnostic**: Works with any image dataset structure with customizable metadata extraction.
- 🛠️ **Analysis Toolkit**: Distance quantification, anomaly detection, property extraction, and correlation analysis.
- 📊 **Interactive Visualizations**: High-performance Plotly-based PCA, t-SNE, and UMAP plots.
- 📑 **Automated Reporting**: Generate comprehensive standalone HTML reports.

---

## 🚀 Quick Start

Try the no-code experience with the **[Interactive Quickstart Notebook](Notebooks/tutorials/01_interactive_quickstart.ipynb)** or launch instantly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/01_interactive_quickstart.ipynb)

---

## 📦 Installation

```bash
pip install phenome
```

PyTorch is included. An NVIDIA GPU is used when CUDA is available; the same install runs on CPU.

<details>
<summary><b>Conda or from source</b></summary>

Clone the repository, then create one environment for your hardware. These files pin the scientific stack and install this checkout.

```bash
git clone https://github.com/AAitorG/PhenoMe.git
cd PhenoMe
```

**GPU**
```bash
conda env create -f envs/environment-gpu.yml
conda activate phenome-gpu
```

**CPU**
```bash
conda env create -f envs/environment-cpu.yml
conda activate phenome-cpu
```
</details>

> **💡 Full instructions, verification, and troubleshooting:**
> See the [Getting started](https://AAitorG.github.io/PhenoMe/getting-started/) guide for install details, verification commands, and first-analysis walkthroughs.

---

## 📖 Documentation

**Explore the [Full Documentation](https://AAitorG.github.io/PhenoMe/)** for detailed guides, API references, and tutorials.

| Step | Resource |
| ---- | -------- |
| 1️⃣ **Install** | [Getting started](https://AAitorG.github.io/PhenoMe/getting-started/) |
| 2️⃣ **Set up your data** | [Data setup](https://AAitorG.github.io/PhenoMe/guides/data-setup/) · [Experiment details](https://AAitorG.github.io/PhenoMe/guides/experiment-details/) |
| 3️⃣ **Tutorials** | [Notebooks/tutorials/](Notebooks/tutorials/) |
| 4️⃣ **Examples** | [Notebooks/examples/](Notebooks/examples/) |

**Learn More:** [Core concepts](https://AAitorG.github.io/PhenoMe/concepts/) · [Workflows](https://AAitorG.github.io/PhenoMe/workflows/) · [FAQ](https://AAitorG.github.io/PhenoMe/faq/) · [Glossary](https://AAitorG.github.io/PhenoMe/glossary/)

---

## 📜 Citation

If you use PhenoMe in your research, please cite:

```bibtex
@software{phenome,
  author = {Aitor González-Marfil},
  title = {PhenoMe: Interactive AI framework for image representation analysis},
  year = {2026},
  url = {https://github.com/AAitorG/PhenoMe}
}
```
