# lnm-toolkit/utils.py

import numpy as np
from nilearn.maskers import NiftiMasker
from scipy.stats import zscore
import nibabel as nib
import pandas as pd


def standardize_data(covariate_data):
    """Cleans up GLM inputs by standardizing continuous columns.

    Uses Z-scoring for continuous data but ignores binary (0/1) columns 
    to preserve the interpretation of dummy variables.

    Args:
        covariate_data (pd.DataFrame): DataFrame containing covariates to standardize.

    Returns:
        pd.DataFrame: A copy of the input DataFrame with standardized columns.
    """
    covariate_data_standardized = covariate_data.copy()
    for col in covariate_data_standardized.columns:
        # Check if the column is binary (contains only 0s and 1s)
        if not (set(covariate_data_standardized[col].unique()) <= {0, 1}):
            covariate_data_standardized[col] = zscore(covariate_data_standardized[col])
    return covariate_data_standardized

def threshold_and_binarize_overlap_sensitivity(data, threshold, percent=False, two_tailed=True):
    """Calculates subject overlap percentage after thresholding.

    Takes a 2D array of subjects and voxel values, binarizes them based on a 
    threshold (two-sided by default), and returns the mean overlap across subjects.

    Args:
        data (np.ndarray): 2D array of network data (n_subjects, n_voxels).
        threshold (float): Absolute value threshold for binarization.
        percent (bool, default=False): If True, multiply results by 100.
        two_tailed (bool, default=True): If True, handles both positive and negative overlap.

    Returns:
        np.ndarray: 1D array representing the overlap/sensitivity map.
    """
    data_positive = np.where(data > threshold, 1, 0)
    data_negative = np.where(data < - threshold, -1, 0)
    data_positive_mean = np.mean(data_positive, axis=0)
    data_negative_mean = np.mean(data_negative, axis=0)
    if two_tailed:
        data_overlap = np.where(np.abs(data_positive_mean) > np.abs(data_negative_mean), data_positive_mean, data_negative_mean)
    else:
        data_overlap = data_positive_mean
    if percent:
        data_overlap = data_overlap * 100
    return np.squeeze(data_overlap)

def recentered_zscore(data):
    """Z-scores data while preserving the original zero-crossover point.

    Standard Z-scoring can shift the zero point if the mean is non-zero. This 
    function recenters the Z-scored data so that the original zero remains at zero.

    Args:
        data (np.ndarray): Array of values to Z-score.

    Returns:
        np.ndarray: Recentered Z-scored array.
    """
    # Compute the z-score for the data
    z = zscore(data)
    
    # Find the index of the element closest to zero in the original data
    zero_index = np.argmin(np.abs(data))
    
    # Recenter the z-scored data so that the element at zero_index becomes 0
    z = z - z[zero_index]
    
    return z

def agreement_map(map_one, map_two, mask_img=None, normalize=False):
    """Computes a signed agreement map between two spatial maps.

    Multiplies two maps together, keeping regions of positive agreement (both > 0) 
    positive and negative agreement (both < 0) negative. Disagreements are zeroed.

    Args:
        map_one (str, nib.Nifti1Image, or np.ndarray): First input map.
        map_two (str, nib.Nifti1Image, or np.ndarray): Second input map.
        mask_img (str or nib.Nifti1Image, optional): Mask to use if inputs are NIfTIs.
        normalize (bool, default=False): Whether to use `recentered_zscore` before multiplication.

    Returns:
        np.ndarray or nib.Nifti1Image: The resulting agreement map.
    """
    
    # Load data
    masker = NiftiMasker(mask_img=mask_img).fit() if mask_img is not None else None
    data_one = np.squeeze(masker.transform(map_one)) if (isinstance(map_one, str) or isinstance(map_one, nib.Nifti1Image)) else map_one
    data_two = np.squeeze(masker.transform(map_two)) if (isinstance(map_two, str) or isinstance(map_two, nib.Nifti1Image)) else map_two

    # Define maks where both maps are positive or negative
    positive_mask = np.logical_and(data_one > 0, data_two > 0)
    negative_mask = np.logical_and(data_one < 0, data_two < 0)

    # Multiply the two maps
    if normalize:
        multiplied = recentered_zscore(data_one) * recentered_zscore(data_two)
    else:
        multiplied = data_one * data_two
    
    # Set regions of positive agreement to be positive, regions of negative agreement to be negative
    multiplied[positive_mask] = 1 * multiplied[positive_mask]
    multiplied[negative_mask] = -1 * multiplied[negative_mask]

    # Zero out regions of disagreement
    multiplied[np.logical_and(~positive_mask, ~negative_mask)] = 0

    return masker.inverse_transform(np.nan_to_num(multiplied)) if mask_img is not None else np.nan_to_num(multiplied)

def conjunction_region_map(sens_map, glm_stat_map, glm_fwep_map, glm_fdrp_map, sensitivity_group_threshold=0.75, alpha=0.05):
    """Identifies regions passing both sensitivity and significance thresholds.

    Args:
        sens_map (np.ndarray): Percentage overlap map from sensitivity analysis.
        glm_stat_map (np.ndarray): T-statistic or similar map from GLM.
        glm_fwep_map (np.ndarray): FWE-corrected p-value map (stored as 1-p).
        glm_fdrp_map (np.ndarray): FDR-corrected p-value map (stored as 1-p).
        sensitivity_group_threshold (float, default=0.75): Overlap percentage threshold.
        alpha (float, default=0.05): Significance threshold for p-value maps.

    Returns:
        tuple: (conjunction_fwep_regions, conjunction_fdrp_regions) binary maps.
    """
    positive_agreement = np.where(np.logical_and(sens_map > (sensitivity_group_threshold * 100), glm_stat_map > 0), 1, 0)
    negative_agreement = np.where(np.logical_and(sens_map < (-sensitivity_group_threshold * 100), glm_stat_map < 0), -1, 0)
    significant_positive_agreement_fwep = np.where(np.logical_and(positive_agreement == 1, (1-glm_fwep_map) < alpha), 1, 0)
    significant_negative_agreement_fwep = np.where(np.logical_and(negative_agreement == -1, (1-glm_fwep_map) < alpha), -1, 0)
    significant_positive_agreement_fdrp = np.where(np.logical_and(positive_agreement == 1, (1-glm_fdrp_map) < alpha), 1, 0)
    significant_negative_agreement_fdrp = np.where(np.logical_and(negative_agreement == -1, (1-glm_fdrp_map) < alpha), -1, 0)
    conjunction_fwep_regions = significant_positive_agreement_fwep + significant_negative_agreement_fwep
    conjunction_fdrp_regions = significant_positive_agreement_fdrp + significant_negative_agreement_fdrp
    return conjunction_fwep_regions, conjunction_fdrp_regions
    positive_agreement = np.where(np.logical_and(sens_map > (sensitivity_group_threshold * 100), glm_stat_map > 0), 1, 0)
    negative_agreement = np.where(np.logical_and(sens_map < (-sensitivity_group_threshold * 100), glm_stat_map < 0), -1, 0)
    significant_positive_agreement_fwep = np.where(np.logical_and(positive_agreement == 1, (1-glm_fwep_map) < alpha), 1, 0)
    significant_negative_agreement_fwep = np.where(np.logical_and(negative_agreement == -1, (1-glm_fwep_map) < alpha), -1, 0)
    significant_positive_agreement_fdrp = np.where(np.logical_and(positive_agreement == 1, (1-glm_fdrp_map) < alpha), 1, 0)
    significant_negative_agreement_fdrp = np.where(np.logical_and(negative_agreement == -1, (1-glm_fdrp_map) < alpha), -1, 0)
    conjunction_fwep_regions = significant_positive_agreement_fwep + significant_negative_agreement_fwep
    conjunction_fdrp_regions = significant_positive_agreement_fdrp + significant_negative_agreement_fdrp
    return conjunction_fwep_regions, conjunction_fdrp_regions