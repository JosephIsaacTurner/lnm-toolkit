# lnm-toolkit/utils.py

import numpy as np
from nilearn.maskers import NiftiMasker
from scipy.stats import zscore
import nibabel as nib
import pandas as pd


def standardize_data(covariate_data):
    """
    Cleans up GLM inputs. Standardizes continuous columns using StandardScaler 
    but ignores binary (0/1) columns to preserve dummy variables.
    """
    covariate_data_standardized = covariate_data.copy()
    for col in covariate_data_standardized.columns:
        # Check if the column is binary (contains only 0s and 1s)
        if not (set(covariate_data_standardized[col].unique()) <= {0, 1}):
            covariate_data_standardized[col] = zscore(covariate_data_standardized[col])
    return covariate_data_standardized

def threshold_and_binarize_overlap_sensitivity(data, threshold, percent=False, two_tailed=True):
    """
    The goal of this function is to take in a 2d array of data (n_subject, n_image_points)
    and return a 1d array of the percentage of subjects that have a value greater than the threshold.
    We do this in a two-sided fashion. If a subject has a value greater than the threshold, we assign it a 1. 
    If a subject has a value less than the negative threshold, we assign it a -1.
    We then take the mean of the positive and negative values across subjects.
    Finally, we take the maximum of the absolute value of the positive and negative means to determine which value to keep at each point.
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
    """Function for Z-scoring data without changing the crossover point from positive to negative values."""
    # Compute the z-score for the data
    z = zscore(data)
    
    # Find the index of the element closest to zero in the original data
    zero_index = np.argmin(np.abs(data))
    
    # Recenter the z-scored data so that the element at zero_index becomes 0
    z = z - z[zero_index]
    
    return z

def agreement_map(map_one, map_two, mask_img=None, normalize=False):
    """
    Function to compute an agreement map between two maps. The agreement map is computed by multiplying the two maps
    together, and setting regions of positive agreement to be positive, regions of negative agreement to be negative,
    and regions of disagreement to be zero.
    
    Parameters
    ----------
    map_one : str
        Path to the first map.
    map_two : str    
        Path to the second map.
    mask_img : str
        Path to the mask image or a Nifti1Image obj.
    normalize : bool, default=False
        Whether to normalize the maps before computing the agreement map.
    
    Returns
    -------
    agreement_img : nibabel.Nifti1Image
        The agreement map between the two maps.
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
    """
    Function to compute a conjunction map between sensitivity and GLM results. The conjunction map is computed by identifying regions where there is agreement between the sensitivity map and the GLM stat map, and where the GLM results are significant at both FWE and FDR corrected levels.
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