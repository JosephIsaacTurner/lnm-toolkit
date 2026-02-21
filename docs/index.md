# lnm-toolkit

Welcome to the documentation for **lnm-toolkit**, a specialized library for lesion network mapping (LNM).

## Overview

The `lnm-toolkit` provides a streamlined workflow for:
- Loading and masking spatial NIfTI data.
- Performing case-control sensitivity analyses.
- Integrating with the **PRISM** statistical backend for permutation-based GLMs.
- Conjunction and agreement analysis between different network results.

## Key Features

- **Spatial Boss**: `LNMDataset` handles all NIfTI file loading and spatial masking, converting 3D images into perfectly aligned 2D arrays.
- **PRISM Integration**: Seamlessly passes data to the PRISM backend for robust statistical testing while maintaining spatial metadata.
- **Convenience**: Provides high-level methods for complex spatial analyses (sensitivity, conjunction) with automated NIfTI export.

## Getting Started

Follow the [Installation](installation.md) guide to get set up, then check out the [Quickstart](quickstart.md) to run your first analysis.
