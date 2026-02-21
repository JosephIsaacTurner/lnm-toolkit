# Quickstart: Case-Control Analysis

This guide provides a quick overview of how to perform a case-control analysis using the `lnm-toolkit` with the included example dataset.

## 1. Data Preparation

The example data is located in `example_data/`. We will use `participants.csv` which contains subject metadata, including the `wab_type` column for grouping.

```python
import pandas as pd

# Load the participant data
participant_data = pd.read_csv('example_data/participants.csv')

# Filter for Broca and NoAphasia groups
filtered_df = participant_data[participant_data['wab_type'].isin(['Broca', 'NoAphasia'])].copy()

# Create a case-control column (1 for Broca, 0 for NoAphasia)
filtered_df['case_control'] = (filtered_df['wab_type'] == 'Broca').astype(int)
```

## 2. Loading the Dataset

We use the `PandasDatasetLoader` to handle file paths and initial masking.

```python
from lnm import PandasDatasetLoader

# Define paths
data_dir = 'example_data'

# Load data using PandasDatasetLoader
loader = PandasDatasetLoader(
    df=filtered_df,
    subject_col='subject',
    network_col='t',
    mask_col='roi_2mm',
    data_dir=data_dir
)
data = loader.load()
```

## 3. Running the Analysis

Now, we can use the `LNMDataset` class to run a GLM with covariate control.

```python
from lnm import LNMDataset
import numpy as np

# Create the dataset
ds = LNMDataset(
    networks=data.networks,
    mask_img=data.mask_img,
    roi_masks=data.roi_masks,
    output_prefix='results/my_analysis',
    control_roi_volume=True,
    control_roi_centrality=True
)
ds.load_data()

# Create design matrix and add covariates
ds.design_matrix = filtered_df[['case_control']].values
ds.add_roi_volume_covar()
ds.add_roi_centrality_covar()
ds.add_intercept()

# Define contrast matrix (testing the case_control effect)
# [case_control, volume, centrality, intercept]
ds.contrast_matrix = np.array([1, 0, 0, 0])

# Run GLM analysis
results = ds.network_glm_analysis()
```

## 4. Interpreting Results

The results are exported as NIfTI files with the specified prefix:
- `results/my_analysis_vox_tstat.nii.gz`: T-statistics for the comparison.
- `results/my_analysis_vox_stat_uncp.nii.gz`: Uncorrected p-values.

For sensitivity or conjunction analysis:

```python
# Sensitivity analysis
sens_results = ds.network_sensitivity_analysis(threshold=7, group_threshold=0.75)

# Conjunction analysis
conj_results = ds.network_conjunction_analysis(threshold=7, sens_thresh=0.75)
```
