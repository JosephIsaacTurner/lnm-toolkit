# Quickstart: Case-Control Analysis with LNM-Toolkit

This document provides a quick overview of how to perform a case-control analysis using the `lnm-toolkit`. We will use the example dataset provided in the `example_data` directory.

## 1. Installation

First, make sure you have the necessary dependencies installed. You can install them using pip:

```bash
pip install pandas nilearn scikit-learn
```

The `lnm-toolkit` also depends on the `prism` library, which is not available on PyPI. You will need to install it manually. Please make sure it is available in your Python environment.

Next, install the `lnm-toolkit` in editable mode:

```bash
pip install -e .
```

## 2. Data Preparation

The example data is located in the `example_data` directory. The `participants.csv` file contains the metadata for the subjects, including the `wab_type` column, which we will use for case-control analysis.

We will filter the data to include only subjects with `wab_type` of 'Broca' or 'NoAphasia'.

```python
import pandas as pd

# Load the participant data
participant_data = pd.read_csv('example_data/participants.csv')

# Filter for Broca and NoAphasia
filtered_df = participant_data[participant_data['wab_type'].isin(['Broca', 'NoAphasia'])].copy()

# Create a case-control column (1 for Broca, 0 for NoAphasia)
filtered_df['case_control'] = (filtered_df['wab_type'] == 'Broca').astype(int)
```

## 3. Running the Analysis

Now, we can use the `PandasDatasetLoader` and `LNMDataset` classes to run the analysis.

```python
from lnm import LNMDataset, PandasDatasetLoader
import numpy as np

# Define paths
data_dir = 'example_data'
output_dir = 'results'

# Load data using PandasDatasetLoader
loader = PandasDatasetLoader(
    df=filtered_df,
    subject_col='subject',
    nifti_col='t',
    mask_col='roi_2mm',
    data_dir=data_dir
)
data = loader.load()

# Create the dataset
ds = LNMDataset(
    networks=data.networks,
    mask_img=data.mask_img,
    roi_masks=data.roi_masks,
    output_dir=output_dir,
    control_roi_volume=True,
    control_roi_centrality=True
)
ds.load_data()

# Create design matrix
ds.design_matrix = data.design_matrix[['case_control']].values
ds.add_roi_volume_covar()
ds.add_roi_centrality_covar()

# Define contrast matrix
ds.contrast_matrix = np.array([1, 0, 0])

# Run GLM analysis
# This will fail if prism is not installed
try:
    results = ds.network_glm_analysis()
    print("GLM analysis complete. Results saved in the 'results' directory.")
except ModuleNotFoundError:
    print("GLM analysis failed. Please make sure the 'prism' library is installed.")
except Exception as e:
    print(f"An error occurred during GLM analysis: {e}")

```

## 4. Interpreting the Results

The results of the analysis will be saved in the `results` directory. The `t_stat.nii.gz` file contains the t-statistics for the case-control comparison, and the `p_value.nii.gz` file contains the corresponding p-values.

You can use a NIfTI viewer to visualize the results.
