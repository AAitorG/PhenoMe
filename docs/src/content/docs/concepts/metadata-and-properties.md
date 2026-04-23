---
title: "Metadata & Properties"
description: Contextual metadata and classical image features.
---

## Metadata

**Metadata** links an image to its experimental context:

- Which drug was applied?
- At what concentration?
- Which plate well is this from?
- What timepoint was this taken at?

By organising images with metadata, PhenoMe can colour plots by drug concentration, calculate distances relative to controls, and compare across plates. See [Experiment details](/PhenoMe/guides/experiment-details/).

## Properties

Where **embeddings** are abstract vectors from a deep learning model, **properties** are classical, explainable image measurements (cell count, average intensity, nuclear area, ...).

PhenoMe computes and stores properties alongside embeddings, so you can ask whether simple properties (like nucleus size) already explain the grouping a deep-learning model sees.

See [Select properties](/PhenoMe/guides/select-properties/) for user-defined metrics and [Property interpretation](/PhenoMe/concepts/property-interpretation/) for the built-ins.
