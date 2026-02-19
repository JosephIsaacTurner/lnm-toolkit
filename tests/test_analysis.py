import pytest
import pandas as pd
import numpy as np
from lnm import LNMDataset, PandasDatasetLoader
from lnm.utils import standardize_data
from lnm.analysis import network_conjunction_analysis, network_sensitivity_analysis
import os
import nibabel as nib

@pytest.fixture
def participant_data():
    return pd.read_csv('example_data/participants.csv')

def test_case_control_analysis(participant_data, tmp_path):
    # Filter data for Broca and NoAphasia
    filtered_df = participant_data[participant_data['wab_type'].isin(['Broca', 'NoAphasia'])].copy()
    
    # Create a case-control column
    filtered_df['case_control'] = (filtered_df['wab_type'] == 'Broca').astype(int)
    
    # Define paths
    output_dir = str(tmp_path)
    
    # Make sure output_dir ends with a slash
    if not output_dir.endswith('/'):
        output_dir += '/'
    
    # Load data and create dataset using PandasDatasetLoader
    loader = PandasDatasetLoader(
        df=filtered_df,
        subject_col='subject',
        network_col='t',
        mask_col='roi_2mm',
        output_prefix=output_dir,
        control_roi_volume=True,
        control_roi_centrality=True,
        add_intercept=True,
        design_matrix=filtered_df[['case_control']].values,
        contrast_matrix=np.array([1, 0, 0, 0]),
        n_permutations=100
    )
    ds = loader.load()

    # Run GLM analysis
    results = ds.network_glm_analysis()

    # List the output files that should have been created
    containing_files = os.listdir(output_dir)
    print("Output files created:", containing_files)
    
    # Check that output files were created
    assert os.path.exists(os.path.join(output_dir, '_vox_tstat.nii.gz'))
    assert os.path.exists(os.path.join(output_dir, '_vox_stat_uncp.nii.gz'))
    
    # More advanced checks could be added here, e.g., on the result values
    assert results is not None
