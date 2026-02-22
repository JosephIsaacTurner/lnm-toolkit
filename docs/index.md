# lnm-toolkit

Welcome to the documentation for **lnm-toolkit**, a specialized and high-performance library for lesion-to-network mapping (LNM) analysis.

## Overview

The `lnm-toolkit` provides a streamlined, professional workflow for investigating the relationship between focal brain lesions and brain networks. It simplifies the complex process of loading, masking, and statistically analyzing spatial NIfTI data, allowing researchers to focus on their scientific questions.

## Key Features

- **The `lnm` CLI Tool**: A powerful command-line interface that allows you to run full analyses—from data loading to conjunction results—with a single command.
- **Robust Statistical Backend**: Seamlessly integrates with **PRISM** for permutation-based GLM analysis, providing rigorous statistical testing for network maps.
- **Advanced Spatial Analysis**:
    - **Sensitivity Analysis**: Calculate subject overlap percentages (subject agreement) at every voxel.
    - **Conjunction Analysis**: Identify regions where significant GLM effects overlap with high subject agreement.
    - **Agreement Maps**: Generate signed agreement maps between different network results.
- **Automated Data Management**: The `LNMDataset` class handles all NIfTI loading and spatial masking, ensuring all data is perfectly aligned in a common analysis space.
- **Covariate Control**: Easily control for factors like ROI volume and network centrality directly within the analysis workflow.

## Getting Started

Whether you are a CLI power user or prefer writing Python scripts, getting started is easy:

1.  **[Installation](installation.md)**: Set up your environment and dependencies.
2.  **[Quickstart](quickstart.md)**: Follow our guide to run your first case-control analysis using the provided example dataset.
3.  **[API Reference](api/index.md)**: Explore the detailed documentation for the toolkit's classes and functions.

For more information and to view the source code, visit our [GitHub repository](https://github.com/example/lnm-toolkit) and read the [README](../README.md).
