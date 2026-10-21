---
title: "Audio, text, and 3D"
description: "Experimental path for a volume, an audio clip, text, or any other file."
sidebar:
  order: 1
---

2D images stay on `find_files` → `process_images`. This page is the experimental path for another modality: a volume, an audio clip, a text file, or any other file. Use a new `PhenoMe()` session, and do not call `process_images` on it.

## Steps

1. Put the files in one folder.
2. `find_files(..., extensions=[...])` keeps only those files.
3. `set_data_io(load_fn, show_fn)` stores two functions:
   - `load_fn(path)` reads one file and returns the sample (an array, a waveform, a string, or anything else).
   - `show_fn(sample)` returns a picture of shape `(H, W)` or `(H, W, C)`. Plots, the interactive explorer, and the HTML report all draw that picture.
4. `process_data(model)` writes one embedding per file, shape `(N, D)`.
5. `compute_properties(additional_property_functions=...)` is optional. Each function is `fn(sample, mask)` and is called with `mask=None`. Every file must return the same keys. Presets such as `intensity` expect a 2D image, so leave them out.
6. `plot_pca()`, clustering, and distances work as they do for images. Draw one file with `plot_data_by_index`. In Jupyter, `create_interactive_explorer()` opens the same click-to-inspect view. In Colab, use `create_colab_interactive_explorer()`.

The model is a `ModelWrapper`. `_get_embeddings` receives the batch and returns shape `(batch, features)`. Image models such as DINOv2 stay on `process_images`. For other files, use a model that accepts your tensor. The `MeanPool` class below is only a stand-in so the script runs. Replace `_get_embeddings` with your model.

Files that share one array shape are stacked. A volume, clip, or string whose shape differs from the others needs `collate_fn` on `process_data`. It receives the samples in the batch and returns one tensor. After `load_results`, call `set_data_io` again before a plot, the explorer, or a report. The checkpoint does not store the loader or the picture.

## 3D volume

One `.npy` file per volume, shape `(Z, Y, X)`. `orthogonal_projections` draws the three maximum-intensity views side by side.

```python
import numpy as np
import torch
from phenome import PhenoMe
from phenome.experimental import orthogonal_projections
from phenome.utils.model_wrapper import ModelWrapper

class MeanPool(ModelWrapper):
    def __init__(self, dim=8):
        super().__init__(model=object())
        self.dim = dim

    def _get_embeddings(self, tensor: torch.Tensor) -> torch.Tensor:
        flat = tensor.reshape(tensor.shape[0], -1).float().mean(dim=1, keepdim=True)
        return flat.repeat(1, self.dim)

def load_volume(path):
    return np.load(path)

def show_volume(volume):
    return orthogonal_projections(volume, axis_order="zyx")

def volume_props(volume, mask):
    return {"mean": float(np.mean(volume))}

pm = PhenoMe(device="cpu")
pm.find_files("volumes", extensions=[".npy"])
pm.set_data_io(load_volume, show_volume)
pm.process_data(MeanPool(), batch_size=4)
pm.compute_properties(additional_property_functions={"any": volume_props})
pm.plot_pca()
pm.plot_data_by_index(0)
```

`axis_order="zyx"` means the array axes are Z, Y, X. Pass another permutation of `"zyx"` when the file is stored differently.

## Audio

Read each `.wav` with the standard library. `spectrogram_image` is one picture you can pass. Another `show_fn` is fine. Clips of different lengths need `collate_fn`. The `pm` object and `MeanPool` model are the ones from the volume script.

```python
import wave

import numpy as np
import torch
from phenome.experimental import spectrogram_image

def load_wave(path):
    with wave.open(path, "rb") as handle:
        frames = handle.readframes(handle.getnframes())
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32)
    return audio / 32768.0

def collate_waves(samples):
    width = max(len(sample) for sample in samples)
    rows = np.zeros((len(samples), width), dtype=np.float32)
    for index, sample in enumerate(samples):
        rows[index, : len(sample)] = sample
    return torch.from_numpy(rows)

def show_wave(samples):
    return spectrogram_image(samples, sample_rate=8000)

def wave_props(samples, mask):
    return {"rms": float(np.sqrt(np.mean(np.square(samples))))}

pm.find_files("clips", extensions=[".wav"])
pm.set_data_io(load_wave, show_wave)
pm.process_data(MeanPool(), collate_fn=collate_waves)
pm.compute_properties(additional_property_functions={"any": wave_props})
pm.plot_pca()
```

## Text

`load_fn` can return a string. Strings do not stack, so pass `collate_fn`. `text_card` draws the string as an image the explorer can show. Again, `pm` and `MeanPool` come from the volume script.

```python
from pathlib import Path

from phenome.experimental import text_card

def load_text(path):
    return Path(path).read_text(encoding="utf-8")

def collate_text(samples):
    rows = []
    for text in samples:
        ids = [ord(ch) % 64 for ch in text[:32]]
        ids += [0] * (32 - len(ids))
        rows.append(ids)
    return torch.tensor(rows, dtype=torch.float32)

def show_text(text):
    return text_card(text)

def text_props(text, mask):
    return {"n_chars": len(text)}

pm.find_files("notes", extensions=[".txt"])
pm.set_data_io(load_text, show_text)
pm.process_data(MeanPool(), collate_fn=collate_text)
pm.compute_properties(additional_property_functions={"any": text_props})
pm.plot_pca()
```

The three helpers live in `phenome.experimental`: `orthogonal_projections`, `spectrogram_image`, and `text_card`. Nothing there runs unless you pass it to `set_data_io`.

[Tutorial 07](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/07_other_modalities.ipynb) runs this order on a small fluorescence volume, a few audio clips, and short text files. It downloads those files into a local folder.
