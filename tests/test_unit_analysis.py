import pytest
import numpy as np
from sklearn.utils import Bunch
from lnm import analysis

def test_network_sensitivity_analysis():
    # Create synthetic data: 10 subjects, 100 voxels
    n_subjects = 10
    n_voxels = 100
    network_data = np.zeros((n_subjects, n_voxels))
    
    # Subject 0-4 have value 10 in voxel 0-9
    network_data[0:5, 0:10] = 10
    # Subject 5-9 have value -10 in voxel 10-19
    network_data[5:10, 10:20] = -10
    
    # Run analysis with threshold 7
    results = analysis.network_sensitivity_analysis(network_data, threshold=7)
    
    # Voxel 0-9 should have 50% positive overlap (since 5/10 subjects pass threshold)
    # Voxel 10-19 should have 50% negative overlap (-50)
    # Others should be 0
    assert np.all(results.sensstat[0:10] == 50)
    assert np.all(results.sensstat[10:20] == -50)
    assert np.all(results.sensstat[20:] == 0)

def test_network_conjunction_analysis():
    # Mock sensitivity results
    n_voxels = 100
    sensstat = np.zeros(n_voxels)
    sensstat[0:10] = 80  # 80% overlap
    sensstat[10:20] = -80 # -80% overlap
    sensitivity_results = Bunch(sensstat=sensstat)
    
    # Mock GLM results
    tstat = np.zeros(n_voxels)
    tstat[0:10] = 5  # Positive T-stat
    tstat[10:20] = -5 # Negative T-stat
    
    # 1-p values: 0.99 means p=0.01
    fwep = np.zeros(n_voxels)
    fwep[0:20] = 0.99 
    
    fdrp = np.zeros(n_voxels)
    fdrp[0:20] = 0.99
    
    glm_results = Bunch(
        tstat=tstat,
        tstat_fwep=fwep,
        tstat_fdrp=fdrp
    )
    
    # Run conjunction analysis
    results = analysis.network_conjunction_analysis(
        sensitivity_results, 
        glm_results, 
        sensitivity_group_threshold=0.75, 
        alpha=0.05
    )
    
    # Check conjunction maps
    # Voxel 0-9 should pass (80 > 75 and p < 0.05) -> 1
    # Voxel 10-19 should pass (-80 < -75 and p < 0.05) -> -1
    assert np.all(results.conjstat_fwep[0:10] == 1)
    assert np.all(results.conjstat_fwep[10:20] == -1)
    assert np.all(results.conjstat_fwep[20:] == 0)
    
    assert np.all(results.conjstat_fdrp[0:10] == 1)
    assert np.all(results.conjstat_fdrp[10:20] == -1)
    assert np.all(results.conjstat_fdrp[20:] == 0)

    # Agreement stat should be non-zero in these regions
    assert np.all(results.agreementstat[0:10] > 0)
    assert np.all(results.agreementstat[10:20] < 0)
    assert np.all(results.agreementstat[20:] == 0)
