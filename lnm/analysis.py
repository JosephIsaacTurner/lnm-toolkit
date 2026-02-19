
# lnm-toolkit.analysis.py
import numpy as np
from sklearn.utils import Bunch
from .io import save_results_to_nifti
from .utils import threshold_and_binarize_overlap_sensitivity, agreement_map


def network_sensitivity_analysis(network_data, threshold=7, group_threshold=0.75, output_dir=None, mask_img=None) -> Bunch:
    """
    Core math function for sensitivity. 
    Expects purely 2D numpy arrays. Binarizes based on threshold and calculates overlap.
    Returns a Bunch with 1D arrays for raw and thresholded sensitivity.
    Optionally calls save_results_to_nifti if output_dir is provided.
    """
    sensitivity_map = threshold_and_binarize_overlap_sensitivity(network_data, threshold=threshold, percent=True)
    
    thresholded_sensitivity_map = np.where(np.abs(sensitivity_map) >= (group_threshold * 100), sensitivity_map, 0)

    results = Bunch(
        sensitivity_map=sensitivity_map,
        thresholded_sensitivity_map=thresholded_sensitivity_map
    )

    if output_dir and mask_img:
        save_results_to_nifti(output_dir, mask_img, results)

    return results


def network_conjunction_analysis(sensitivity_results, glm_results, output_dir=None, mask_img=None) -> Bunch:
    """
    Runs sensitivity on cases and conjoins it with GLM results.
    Multiplies maps to find agreement and conjunction.
    Expects 2D numpy arrays. Returns a Bunch with all 1D array results.
    Optionally calls save_results_to_nifti.
    """

    # Sensitivity in cases
    sensitivity_map = sensitivity_results.thresholded_sensitivity_map

    # GLM results
    glm_map = glm_results.tstat # Assuming the key for the t-map is 't_map'
    
    # Conjunction using agreement map
    conjunction_map = agreement_map(sensitivity_map, glm_map)

    results = Bunch(
        case_sensitivity_map=sensitivity_map,
        glm_map=glm_map,
        thresholded_glm_map=glm_map,
        conjunction_map=conjunction_map,
    )

    if output_dir and mask_img:
        save_results_to_nifti(output_dir, mask_img, results)
        
    return results


