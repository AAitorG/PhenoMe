## Using STED-FM model in the PhenoMe pipeline

**STED-FM**: A Self-Supervised Foundation Model for Robust and Generalizable Representation Learning in STED Microscopy

[STED-FM on GitHub](https://github.com/FLClab/STED-FM)

### How to use:

This branch of PhenoMe has STED-FM in its dependencies. In a python virtual environment, use `uv sync` to install all required dependencies.

Otherwise, follow the instructions on the STED-FM GitHub repo to install it as a standalone package.

Follow the steps in `sted-fm_wrapper_notebook.ipynb` to apply the PhenoMe pipeline to existing STED image datasets.

If you wish to use your own data, it should follow the file structure required by PhenoMe:

```
├── dataset/
│   └── images/
│       └── <Your STED images>
│   └── masks/
│       └── <Optional binary masks>
│   └── metadata.csv
```

You may then replace `images_dir`, `masks_dir` and `metadata_dir` with your own directories in the notebook.