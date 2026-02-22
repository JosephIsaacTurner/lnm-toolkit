
# lnm-toolkit.analysis.py
import numpy as np
from sklearn.utils import Bunch
from prism.preprocessing import ResultSaver
import os
from tqdm import tqdm
import nibabel as nib
from nilearn.maskers import NiftiMasker
from .utils import threshold_and_binarize_overlap_sensitivity, agreement_map, conjunction_region_map


def network_sensitivity_analysis(network_data, threshold=7, output_prefix=None, mask_img=None) -> Bunch:
    """Core math function for sensitivity (overlap) analysis.

    Expects pre-masked 2D numpy arrays. Binarizes each subject's data based on 
    the provided threshold and calculates the percentage of subject overlap at each voxel.

    Args:
        network_data (np.ndarray): 2D array of flattened network maps (n_subjects, n_voxels).
        threshold (float): Z-score threshold for individual subject binarization.
        output_prefix (str, optional): Prefix for saving NIfTI results via ResultSaver.
        mask_img (nib.Nifti1Image or str, optional): Mask for inverse-transforming results.

    Returns:
        Bunch: A dictionary-like object containing:
            - **sensstat**: 1D array of the group-level sensitivity (overlap percentage) map.
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
    """Performs conjunction and agreement analysis between sensitivity and GLM results.

    Conjoins group-level sensitivity maps with GLM statistics to find regions 
    where significant effects overlap with high subject agreement.

    Args:
        sensitivity_results (Bunch): Results from `network_sensitivity_analysis`.
        glm_results (Bunch): Results from `network_glm_analysis` (PRISM output).
        sensitivity_group_threshold (float): Overlap percentage threshold (default 0.75).
        alpha (float): Significance threshold for GLM maps (default 0.05).
        output_prefix (str, optional): Prefix for saving NIfTI results.
        mask_img (nib.Nifti1Image or str, optional): Mask for inverse-transforming results.

    Returns:
        Bunch: A dictionary-like object containing:
            - **sensstat**: The original sensitivity map.
            - **tstat**: The original GLM T-statistic map.
            - **agreementstat**: Normalized agreement map (signed product).
            - **conjstat_fwep**: Binary map of regions passing both sensitivity and FWEp thresholds.
            - **conjstat_fdrp**: Binary map of regions passing both sensitivity and FDRp thresholds.

    Raises:
        KeyError: If `glm_results` does not contain a 'tstat' or 'vox_tstat' key.
    """
    
    # Sensitivity in cases
    sensitivity_map = sensitivity_results.sensstat

    # GLM results
    # Use tstat as primary key from PRISM
    glm_map = getattr(glm_results, 'tstat', getattr(glm_results, 'vox_tstat', None))
    fdrp_map = getattr(glm_results, 'tstat_fdrp', getattr(glm_results, 'vox_tstat_fdrp', None))
    fwep_map = getattr(glm_results, 'tstat_fwep', getattr(glm_results, 'vox_tstat_fwep', None))

    if any(isinstance(map_, nib.Nifti1Image) for map_ in [glm_map, fdrp_map, fwep_map]):
        masker = NiftiMasker(mask_img=mask_img).fit() if mask_img is not None else None
        if masker is None:
            raise ValueError("mask_img must be provided if GLM results are NIfTI images.")
        if glm_map is not None:
            glm_map = np.squeeze(masker.transform(glm_map))
        if fdrp_map is not None:
            fdrp_map = np.squeeze(masker.transform(fdrp_map))
        if fwep_map is not None:
            fwep_map = np.squeeze(masker.transform(fwep_map))
    
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


def network_sensitivity_permutation_analysis(full_network_data, permuted_indices, cases_control_labels, threshold=7, group_threshold=0.75, output_prefix=None, mask_img=None):
    """Performs permutation analysis for network sensitivity maps.

    Iterates through permuted indices, recalculates sensitivity maps for each 
    permutation, and optionally saves them to disk.

    Args:
        full_network_data (np.ndarray): 2D array of flattened network maps for all subjects.
        permuted_indices (np.ndarray): 2D array of indices for each permutation (n_permutations, n_subjects).
        cases_control_labels (np.ndarray): Boolean or 0/1 array indicating which subjects are cases.
        threshold (float): Z-score threshold for individual subject binarization.
        group_threshold (float): Percentage threshold (0-1) for group-level overlap.
        output_prefix (str, optional): Prefix for saving results.
        mask_img (nib.Nifti1Image or str, optional): Mask for inverse-transforming results.

    Returns:
        Bunch: A dictionary-like object containing:
            - **permuted_sensitivity_maps**: 2D array of permuted sensitivity maps (n_permutations, n_voxels).
    """
    n_permutations = permuted_indices.shape[0]
    result_saver = ResultSaver(output_prefix=output_prefix, mask_img=mask_img, save_permutations=True, permutation_output_dir=os.path.join(os.path.dirname(output_prefix), "sensitivity_permutations")) if output_prefix and mask_img else None
    permuted_sensitivity_maps = np.zeros((n_permutations, full_network_data.shape[1]))
    for i in tqdm(range(n_permutations), desc="Running sensitivity permutations"):
        permuted_idx = permuted_indices[i]
        permuted_data = full_network_data[permuted_idx, :]
        if cases_control_labels is not None:
            permuted_data = permuted_data[cases_control_labels.astype(bool), :]
            permuted_sensitivity_map = threshold_and_binarize_overlap_sensitivity(permuted_data, threshold=threshold, percent=True)
            permuted_sensitivity_maps[i, :] = permuted_sensitivity_map
            if result_saver is not None:
                # def save_permutation(self, permuted_stats, perm_idx, contrast_idx, *args, **kwargs):
                result_saver.save_permutation(
                    permuted_stats=permuted_sensitivity_map,
                    perm_idx=i,
                    contrast_idx=0,  # Assuming a single contrast for sensitivity analysis
                )
    return Bunch(permuted_sensitivity_maps=permuted_sensitivity_maps)