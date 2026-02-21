# lnm-toolkit/dataset.py

import numpy as np
import os
import nibabel as nib
from sklearn.utils import Bunch
from sklearn.preprocessing import StandardScaler
from prism.datasets import Dataset as PrismDataset
from nilearn.maskers import NiftiMasker
from prism.preprocessing import ResultSaver
from . import analysis


class LNMDataset:
    """
    Main data manager for lesion network mapping. 
    Converts 3D NIfTI inputs into flat 2D arrays using a common mask.
    This guarantees that all downstream tools (like PRISM or standalone analyses)
    get perfectly aligned numpy arrays and never have to guess about spatial mapping.
    """
    def __init__(self, networks, mask_img, roi_masks=None, design_matrix=None, 
                 contrast_matrix=None, cases_control_labels=None, statistic='t', 
                 output_prefix=None, control_roi_volume=False, control_roi_centrality=False, 
                 add_intercept=False, glm_config=None, n_permutations=1000):
        self.networks = networks
        self.mask_img = mask_img
        self.roi_masks = roi_masks
        self.design_matrix = design_matrix
        self.contrast_matrix = contrast_matrix
        self.cases_control_labels = cases_control_labels
        self.statistic = statistic
        self.output_prefix = output_prefix
        self.control_roi_volume = control_roi_volume
        self.control_roi_centrality = control_roi_centrality
        self._add_intercept = add_intercept
        self.glm_config = glm_config
        self.n_permutations = n_permutations
        
        # Populated by load_data()
        self.network_data = None
        self.roi_data = None
        self.n_subjects = 0
        self.n_voxels = 0
        self.roi_volume = None
        self.roi_centrality = None

    def load_data(self):
        """
        Fuses inputs into the analysis space.
        Uses NiftiMasker fitted on self.mask_img to transform networks and roi_masks 
        into (n_subjects, n_voxels) 2D arrays. Sets the subject and voxel counts.
        """
        masker = NiftiMasker(mask_img=self.mask_img).fit()
        
        if isinstance(self.networks, np.ndarray) and self.networks.ndim == 2:
            self.network_data = self.networks
        else:
            self.network_data = masker.transform(self.networks)
            
        if self.roi_masks is not None:
            if isinstance(self.roi_masks, np.ndarray) and self.roi_masks.ndim == 2:
                self.roi_data = self.roi_masks
            else:
                self.roi_data = masker.transform(self.roi_masks)
                
        self.n_subjects, self.n_voxels = self.network_data.shape
        self._check_subject_counts()
        
        # Auto-populate volume and centrality metrics
        if self.roi_data is not None:
            self.roi_volume = self.calculate_roi_volume(self.roi_data)
            
        self.roi_centrality = self.calculate_roi_centrality(self.network_data, self.roi_data)

    def _check_subject_counts(self):
        """
        Fails fast if the number of rows in network_data, roi_data, 
        and design_matrix don't match.
        """
        # Network data is our ground truth for n_subjects
        if self.roi_data is not None and self.roi_data.shape[0] != self.n_subjects:
            raise ValueError(f"Subject mismatch: Networks ({self.n_subjects}) vs ROIs ({self.roi_data.shape[0]})")
            
        if self.design_matrix is not None and self.design_matrix.shape[0] != self.n_subjects:
            raise ValueError(f"Subject mismatch: Networks ({self.n_subjects}) vs Design Matrix ({self.design_matrix.shape[0]})")
            
        if self.cases_control_labels is not None and len(self.cases_control_labels) != self.n_subjects:
            raise ValueError(f"Subject mismatch: Networks ({self.n_subjects}) vs Labels ({len(self.cases_control_labels)})")

    @staticmethod
    def calculate_roi_volume(roi_data):
        """
        Takes a 2D ROI array and returns a 1D array of non-zero voxel counts per subject.
        """
        # Simply count non-zero voxels per subject
        return np.count_nonzero(roi_data, axis=1)

    @staticmethod
    def calculate_roi_centrality(network_data, roi_data=None):
        """
        Returns a 1D array of average network values per subject. 
        Excludes voxels falling within the roi_data mask to prevent circularity.
        """
        if roi_data is not None:
            # Create a boolean mask where ROI is 0 (non-lesion voxels)
            valid_voxels = roi_data == 0
            # Sum the valid network values and divide by the number of valid voxels per subject
            return np.sum(network_data * valid_voxels, axis=1) / np.sum(valid_voxels, axis=1)
        
        # If no ROIs provided, just take the mean of the whole network map
        return np.mean(network_data, axis=1)
    
    def add_roi_volume_covar(self):
        """
        Calculates volume and appends it to self.design_matrix. 
        Checks for collinearity with existing columns first.
        """
        if self.roi_volume is None:
            return

        new_col = self.roi_volume.reshape(-1, 1)
        new_col = StandardScaler().fit_transform(new_col)

        if self.design_matrix is None:
            self.design_matrix = new_col
            return
        
        dm = self.design_matrix
        if dm.ndim == 1:
            dm = dm.reshape(-1, 1)

        # Check for collinearity
        combined = np.hstack([dm, new_col])
        corr_matrix = np.corrcoef(combined, rowvar=False)
        
        # Correlation of the new column with all previous columns
        corr_with_new = corr_matrix[:-1, -1]

        if np.all(np.abs(corr_with_new) < 0.9):
            self.design_matrix = combined

    def add_roi_centrality_covar(self):
        """
        Calculates centrality and appends it to self.design_matrix.
        Checks for collinearity with existing columns first.
        """
        if self.roi_centrality is None:
            return
            
        new_col = self.roi_centrality.reshape(-1, 1)
        new_col = StandardScaler().fit_transform(new_col)

        if self.design_matrix is None:
            self.design_matrix = new_col
            return

        dm = self.design_matrix
        if dm.ndim == 1:
            dm = dm.reshape(-1, 1)

        # Check for collinearity
        combined = np.hstack([dm, new_col])
        corr_matrix = np.corrcoef(combined, rowvar=False)

        # Correlation of the new column with all previous columns
        corr_with_new = corr_matrix[:-1, -1]

        if np.all(np.abs(corr_with_new) < 0.9):
            self.design_matrix = combined

    def add_intercept(self):   
        """
        Adds an intercept column to the design matrix if it doesn't already exist.
        """
        if self.design_matrix is None:
            self.design_matrix = np.ones((self.n_subjects, 1))
            return

        # Check if an intercept column already exists
        if np.any(np.all(self.design_matrix == 1, axis=0)):
            return

        self.design_matrix = np.hstack([self.design_matrix, np.ones((self.n_subjects, 1))])

    def prepare_glm_config(self, **kwargs) -> PrismDataset:
        """
        The bridge to PRISM.
        Instantiates a PrismDataset using our pre-computed 2D arrays (network_data, design_matrix). 
        Crucially, we pass self.mask_img just so PRISM can inverse-transform the results later, 
        but passing the 2D data forces PRISM to skip its own internal NiftiMasker logic.
        """
        if self.control_roi_volume:
            self.add_roi_volume_covar()
        if self.control_roi_centrality:
            self.add_roi_centrality_covar()
        if self._add_intercept:
            self.add_intercept()
            
        return PrismDataset(
            data=self.network_data,
            design=self.design_matrix,
            contrast=self.contrast_matrix,
            output_prefix=self.output_prefix,
            n_permutations=self.n_permutations,
            **kwargs
        )

    def network_sensitivity_analysis(self, threshold=7, group_threshold=0.75) -> Bunch:
        """
        Convenience method. Slices self.network_data for cases (if labels exist) 
        and passes it to the standalone network_sensitivity_analysis function.
        """
        if self.cases_control_labels is not None:
            network_data = self.network_data[self.cases_control_labels.astype(bool)]
            print("shape of network data for sensitivity:", network_data.shape)
        else:
            network_data = self.network_data

        return analysis.network_sensitivity_analysis(
            network_data=network_data,
            threshold=threshold,
            group_threshold=group_threshold,
            output_prefix=self.output_prefix,
            mask_img=self.mask_img
        )

    def network_glm_analysis(self, **kwargs) -> Bunch:
        """
        Convenience method. Calls self.prepare_glm_config() to get a PrismDataset, 
        runs permutation_analysis(), and returns the PRISM results.
        """
        # 1. Instantiate the PRISM dataset
        prism_ds = self.prepare_glm_config(**kwargs)
        
        # 2. Run the permutation analysis via the analysis module's wrapper
        results = prism_ds.permutation_analysis()
        
        # 3. Export to NIfTI using our toolkit's I/O, not PRISM's
        if self.output_prefix and self.mask_img:
            saver = ResultSaver(output_prefix=self.output_prefix, mask_img=self.mask_img)
            saver.save_results(results)
            
        return results

    def network_conjunction_analysis(self, threshold=7, sens_thresh=0.75, **kwargs) -> Bunch:
        """
        Convenience method. Passes the full dataset and labels to the 
        standalone network_conjunction_analysis function.
        """

        sensitivity_results = self.network_sensitivity_analysis(threshold=threshold, group_threshold=sens_thresh)
        glm_results = self.network_glm_analysis(**kwargs)

        return analysis.network_conjunction_analysis(
            sensitivity_results=sensitivity_results,
            glm_results=glm_results,
            output_prefix=self.output_prefix,
            mask_img=self.mask_img
        )

