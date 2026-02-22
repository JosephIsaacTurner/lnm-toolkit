# LNM-Toolkit

A comprehensive toolkit for Lesion-to-Network Mapping (LNM) analysis.

## Overview

The `lnm-toolkit` provides tools for analyzing the relationship between focal brain lesions and brain networks. It supports:
- **Sensitivity Analysis**: Calculating subject overlap maps.
- **GLM Analysis**: Permutation-based General Linear Models (via the `prism` backend).
- **Conjunction Analysis**: Identifying regions that are both significant in the GLM and have high subject overlap.
- **CLI Tool**: Easy-to-use command-line interface for running complex analyses.

## Installation

```bash
# Recommended: Create a conda environment
conda create -n analysis_env python=3.13
conda activate analysis_env

# Install dependencies (including prism)
# ... instructions for prism installation ...

# Install the toolkit
pip install -e .
```

## Quick Start

See [QUICKSTART.md](QUICKSTART.md) for a detailed guide on running your first analysis.

### Command Line Interface

```bash
lnm --csv example_data/aphasia_recovery_participants.csv 
    --subject-col subject 
    --network-col t 
    --roi-col roi_2mm 
    --output-prefix results/my_analysis 
    --analysis conjunction 
    --filter-col wab_type 
    --filter-values Broca NoAphasia 
    --contrast-col wab_type 
    --contrast-values Broca NoAphasia
```

## Documentation

Full documentation is available in the `docs/` directory and can be built with `mkdocs`.

## Testing

Run tests using `pytest`:

```bash
conda run -n analysis_env pytest tests/
```
