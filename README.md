# LNM-Toolkit

A comprehensive and high-performance Python library for **Lesion-to-Network Mapping (LNM)** analysis.

## Overview

The `lnm-toolkit` provides a specialized workflow for investigating the relationship between focal brain lesions and brain networks. While originally designed for lesion mapping, the underlying techniques generalize to any analysis of Regions of Interest (ROIs) and their connectivity profiles, including:
- **TMS-Network Mapping**: Analyzing connectivity of stimulation targets.
- **DBS-Network Mapping**: Analyzing connectivity of electrode locations.
- **General ROI Connectivity**: Any analysis comparing focal ROIs across subjects and their functional/structural connectivity maps.

### What it Does
- **Analyzes Pre-Generated Connectivity Maps**: This toolkit assumes you have already generated connectivity maps (e.g., using a normative connectome) for each subject and saved them as NIfTI images.
- **Subject Data Loading**: Easily load data using a CSV file containing filepaths to subject-level network and ROI data.
- **Sensitivity Analysis**: Calculate subject overlap percentages at every voxel.
- **GLM Analysis**: Perform robust permutation-based General Linear Models (via the **PRISM** backend) to compare groups while controlling for covariates like age, sex, or ROI volume.
- **Conjunction Analysis**: Identify regions where significant statistical effects overlap with high subject agreement.

## Installation

```bash
# Recommended: Create a conda environment
conda create -n analysis_env python=3.13
conda activate analysis_env

# Install the toolkit directly from the repository
# This will automatically install dependencies like prism-neuro from PyPI
pip install -e .
```

## Quick Start

The easiest way to get started is to use a CSV file with your subject data and run the `lnm` command-line tool.

### CLI Example: Case-Control Conjunction

```bash
lnm --csv participants.csv \
    --subject-col subject \
    --network-col network_filepaths \
    --roi-col roi_filepaths \
    --output-prefix results/my_analysis \
    --analysis conjunction \
    --filter-col group \
    --filter-values cases controls \
    --contrast-col group \
    --contrast-values cases controls
```

For a more detailed guide, see [QUICKSTART.md](QUICKSTART.md).

## Documentation

Full documentation is available in the `docs/` directory and can be built with `mkdocs`.

## Testing

Run tests using `pytest`:

```bash
conda run -n analysis_env pytest tests/
```
