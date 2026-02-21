
# lnm-toolkit.analysis.py
import numpy as np
from sklearn.utils import Bunch
from prism.preprocessing import ResultSaver
from .utils import threshold_and_binarize_overlap_sensitivity, agreement_map, conjunction_region_map


def network_sensitivity_analysis(network_data, threshold=7, group_threshold=0.75, output_prefix=None, mask_img=None) -> Bunch:
    """
    Core math function for sensitivity. 
    Expects purely 2D numpy arrays. Binarizes based on threshold and calculates overlap.
    Returns a Bunch with 1D arrays for raw and thresholded sensitivity.
    Optionally calls ResultSaver if output_prefix is provided.
    """
    sensitivity_map = threshold_and_binarize_overlap_sensitivity(network_data, threshold=threshold, percent=True)
    
    results = Bunch(
        sensstat=sensitivity_map
    )

    if output_prefix and mask_img:
        saver = ResultSaver(output_prefix=output_prefix, mask_img=mask_img)
        saver.save_results(results)

    return results


def network_conjunction_analysis(sensitivity_results, glm_results, sensitivity_group_threshold=0.75, alpha=0.05, output_prefix=None, mask_img=None) -> Bunch:
    """
    Runs sensitivity on cases and conjoins it with GLM results.
    Multiplies maps to find agreement and conjunction.
    Expects 2D numpy arrays. Returns a Bunch with all 1D array results.
    Optionally calls ResultSaver.
    """

    # Sensitivity in cases
    sensitivity_map = sensitivity_results.sensstat

    # GLM results
    # Use tstat as primary key from PRISM
    glm_map = getattr(glm_results, 'tstat', getattr(glm_results, 'vox_tstat', None))
    uncp_map = getattr(glm_results, 'tstat_uncp', getattr(glm_results, 'vox_tstat_uncp', None))
    fdrp_map = getattr(glm_results, 'tstat_fdrp', getattr(glm_results, 'vox_tstat_fdrp', None))
    fwep_map = getattr(glm_results, 'tstat_fwep', getattr(glm_results, 'vox_tstat_fwep', None))
    
    if glm_map is None:
        raise KeyError("glm_results must contain 'tstat' or 'vox_tstat' key.")
    
    # Conjunction using agreement map
    agreement_stat = agreement_map(sensitivity_map, glm_map, normalize=True)

    # Conjunction map: 
    # def conjunction_region_map(sens_map, glm_stat_map, glm_fwep_map, glm_fdrp_map, sensitivity_group_threshold=0.75, alpha=0.05):

    conjunction_fwep_regions, conjunction_fdrp_regions = conjunction_region_map(
        sens_map=sensitivity_map,
        glm_stat_map=glm_map,
        glm_fwep_map=fwep_map,
        glm_fdrp_map=fdrp_map,
        sensitivity_group_threshold=sensitivity_group_threshold,
        alpha=alpha
    )

    results = Bunch(
        sensstat=sensitivity_map,
        tstat=glm_map,
        agreementstat=agreement_stat,
        conjstat_fwep=conjunction_fwep_regions,
        conjstat_fdrp=conjunction_fdrp_regions
    )

    if output_prefix and mask_img:
        saver = ResultSaver(output_prefix=output_prefix, mask_img=mask_img)
        saver.save_results(results)
        
    return results


