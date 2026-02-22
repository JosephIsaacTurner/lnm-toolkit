# Quickstart: Case-Control Analysis with LNM-Toolkit

This document provides a quick overview of how to perform a case-control analysis using the `lnm-toolkit`. We will use the example dataset provided in the `example_data` directory.

## 1. Installation

First, make sure you have the necessary dependencies installed. You can install them using conda:

```bash
conda activate analysis_env
# Install the toolkit in editable mode
pip install -e .
```

The `lnm-toolkit` depends on the `prism` library, which should already be available in the `analysis_env` environment.

## 2. Using the CLI (Recommended)

The easiest way to run an analysis is using the `lnm` command-line tool.

### Example: Case-Control Conjunction Analysis

To compare two groups (e.g., 'Broca' vs 'NoAphasia' in the `wab_type` column), controlling for `age_at_stroke`, you can run:

```bash
lnm --csv example_data/aphasia_recovery_participants.csv \
    --subject-col subject \
    --network-col t \
    --roi-col roi_2mm \
    --output-prefix results/aphasia_analysis \
    --analysis conjunction \
    --filter-col wab_type \
    --filter-values Broca NoAphasia \
    --contrast-col wab_type \
    --contrast-values Broca NoAphasia \
    --covariates age_at_stroke \
    --add-intercept \
    --n-permutations 1000
```

This will:
1. Load the participant CSV.
2. Filter for subjects where `wab_type` is 'Broca' or 'NoAphasia'.
3. Run a GLM comparing these two groups while controlling for `age_at_stroke`.
4. Run a sensitivity analysis on the 'Broca' group (cases).
5. Perform a conjunction between the sensitivity and GLM results.

## 3. Using the Python API

If you prefer to work in a notebook or script, you can use the `LNMDataset` class.

```python
import pandas as pd
import numpy as np
from lnm import PandasDatasetLoader

# Load the participant data
df = pd.read_csv('example_data/aphasia_recovery_participants.csv')

# Filter for Broca and NoAphasia
filtered_df = df[df['wab_type'].isin(['Broca', 'NoAphasia'])].copy()

# 1 for Broca (cases), 0 for NoAphasia (controls)
group_labels = (filtered_df['wab_type'] == 'Broca').astype(int).values

# Load data
loader = PandasDatasetLoader(
    df=filtered_df,
    subject_col='subject',
    network_col='t',
    mask_col='roi_2mm',
    output_prefix='results/api_test',
    design_matrix=group_labels.reshape(-1, 1),
    contrast_matrix=np.array([1, 0]), # Assuming 1 covariate + intercept might be added
    cases_control_labels=group_labels,
    add_intercept=True,
    n_permutations=100
)
ds = loader.load()

# Run conjunction analysis
results = ds.network_conjunction_analysis()
```

## 4. Interpreting Output Files

The results will be saved as NIfTI files with the specified prefix:
- `_vox_sensstat.nii.gz`: Overlap percentage map of the cases.
- `_vox_tstat.nii.gz`: T-statistic map from the GLM.
- `_vox_conjstat_fwep.nii.gz`: Conjunction map showing regions passing both sensitivity and FWEp thresholds.
- `_vox_conjstat_fdrp.nii.gz`: Conjunction map showing regions passing both sensitivity and FDRp thresholds.
- `_vox_agreementstat.nii.gz`: Signed agreement map.
