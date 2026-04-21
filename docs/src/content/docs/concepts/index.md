---
title: "Core concepts"
description: High-level overview of the ideas behind PhenoMe.
sidebar:
  order: 4
---

import { Card, CardGrid } from '@astrojs/starlight/components';

This section explains the core ideas behind PhenoMe. Skim the overview first—the rest of the documentation assumes the vocabulary introduced here.

## What is phenotyping?

**Phenotyping** characterises the observable traits of biological samples. In microscopy this means analysing cell shape (morphology), where proteins are located (fluorescence patterns), or how structures change under treatment.

PhenoMe automates this with deep learning. It is **model-agnostic** (it works with almost any vision model) and **dataset-agnostic** (it does not care how your files are organised).

## Processing flow

![Pipeline Overview](../../../assets/pipeline_overview.png)

PhenoMe processes data through a bifurcated 4-step pipeline:

1.  **Discovery (Input)** — Discover images and extract experimental metadata.
2.  **Embeddings (A)** — Load images, apply transforms, and extract deep learning embeddings.
3.  **Properties (B)** — Compute interpretable classical features (shape, texture, intensity) in parallel.
4.  **Analysis & Visualization (C)** — Integrate both sources to quantify phenotypic distances and produce interactive plots.

## Explore concepts

<CardGrid stagger>
  <Card title="Embeddings & Models" icon="rocket">
    How visual fingerprints are extracted and the lifecycle of embeddings.
    [Read more](/PhenoMe/concepts/embeddings/)
  </Card>
  <Card title="Channel Modes" icon="seti:julia">
    How to handle multi-channel fluorescence vs. brightfield images.
    [Read more](/PhenoMe/concepts/channel-modes/)
  </Card>
  <Card title="Distance Metrics" icon="magnifier">
    Measuring phenotypic change between treated and control groups.
    [Read more](/PhenoMe/concepts/distances/)
  </Card>
  <Card title="Metadata & Properties" icon="document">
    Linking images to experimental context and classical image features.
    [Read more](/PhenoMe/concepts/metadata-and-properties/)
  </Card>
  <Card title="Visualization & Results" icon="setting">
    Dimensionality reduction (PCA/UMAP), results container structure, and
    sharing/portability of checkpoints.
    [Read more](/PhenoMe/concepts/visualization-and-results/)
  </Card>
</CardGrid>
