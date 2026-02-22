# lnm-toolkit/dataset.py

import numpy as np
from sklearn.utils import Bunch
from sklearn.preprocessing import StandardScaler
from prism.datasets import Dataset as PrismDataset
from nilearn.maskers import NiftiMasker
from prism.preprocessing import ResultSaver
from . import analysis


class LNMDataset:
    """Main data manager for lesion network mapping.

    Converts 3D NIfTI inputs into flat 2D arrays using a common mask.
    This guarantees that all downstream tools (like PRISM or standalone analyses)
    get perfectly aligned numpy arrays and never have to guess about spatial mapping.

    Attributes:
        networks (list or np.ndarray): List of NIfTI file paths or a 2D array of network data.
        mask_img (nib.Nifti1Image or str): Master mask image to define the analysis space.
        roi_masks (list or np.ndarray, optional): List of NIfTI file paths or a 2D array of ROI masks.
        design_matrix (np.ndarray, optional): GLM design matrix (n_subjects, n_covariates).
        contrast_matrix (np.ndarray, optional): GLM contrast matrix.
        cases_control_labels (np.ndarray, optional): Boolean or 0/1 array indicating cases for sensitivity analysis.
        statistic (str): The statistic type for PRISM (default 't').
        output_prefix (str, optional): Prefix for output file paths.
        control_roi_volume (bool): Whether to automatically add ROI volume as a covariate.
        control_roi_centrality (bool): Whether to automatically add network centrality as a covariate.
        add_intercept (bool): Whether to automatically add an intercept column to the design matrix.
        glm_config (dict, optional): Extra configuration for the PRISM GLM.
        n_permutations (int): Number of permutations for statistical testing.
        network_data (np.ndarray): Flattened network data (n_subjects, n_voxels).
        roi_data (np.ndarray): Flattened ROI data (n_subjects, n_voxels).
        n_subjects (int): Number of subjects in the dataset.
        n_voxels (int): Number of voxels within the master mask.
        roi_volume (np.ndarray): Calculated volume per subject.
        roi_centrality (np.ndarray): Calculated network centrality per subject.
    """
    def __init__(self, networks, mask_img, roi_masks=None, design_matrix=None, 
                 contrast_matrix=None, cases_control_labels=None, statistic='t', 
                 output_prefix=None, control_roi_volume=False, control_roi_centrality=False, 
                 add_intercept=False, n_permutations=1000, **glm_config):
        """Initializes the LNMDataset with necessary parameters."""
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
        self.n_permutations = n_permutations
        self.sensitivity_threshold = 7.0
        self.group_sensitivity_threshold = 0.75
        self.glm_config = glm_config
        
        # Populated by load_data()
        self.network_data = None
        self.roi_data = None
        self.n_subjects = 0
        self.n_voxels = 0
        self.roi_volume = None
        self.roi_centrality = None

    def load_data(self):
        """Fuses inputs into the analysis space.

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
        """Validates that all input data arrays have matching subject counts.

        Raises:
            ValueError: If subject counts between networks, ROIs, labels, or design matrix mismatch.
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
        """Calculates ROI volume per subject.

        Args:
            roi_data (np.ndarray): 2D array of flattened ROI masks.

        Returns:
            np.ndarray: 1D array of non-zero voxel counts per subject.
        """
        # Simply count non-zero voxels per subject
        return np.count_nonzero(roi_data, axis=1)

    @staticmethod
    def calculate_roi_centrality(network_data, roi_data=None):
        """Calculates average network connectivity per subject.

        Excludes voxels falling within the roi_data mask to prevent circularity.

        Args:
            network_data (np.ndarray): 2D array of flattened network maps.
            roi_data (np.ndarray, optional): 2D array of flattened ROI masks.

        Returns:
            np.ndarray: 1D array of average network values per subject.
        """
        if roi_data is not None:
            # Create a boolean mask where ROI is 0 (non-lesion voxels)
            valid_voxels = roi_data == 0
            # Sum the valid network values and divide by the number of valid voxels per subject
            return np.sum(network_data * valid_voxels, axis=1) / np.sum(valid_voxels, axis=1)
        
        # If no ROIs provided, just take the mean of the whole network map
        return np.mean(network_data, axis=1)
    
    def add_roi_volume_covar(self):
        """Appends standardized ROI volume to the design matrix.

        Checks for collinearity (r < 0.9) with existing columns before appending.
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
        """Appends standardized network centrality to the design matrix.

        Checks for collinearity (r < 0.9) with existing columns before appending.
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
        """Adds an intercept column (all ones) to the design matrix if it doesn't exist."""
        if self.design_matrix is None:
            self.design_matrix = np.ones((self.n_subjects, 1))
            return

        # Check if an intercept column already exists
        if np.any(np.all(self.design_matrix == 1, axis=0)):
            return

        self.design_matrix = np.hstack([self.design_matrix, np.ones((self.n_subjects, 1))])

    def prepare_glm_config(self) -> PrismDataset:
        """Instantiates a PRISM Dataset object for GLM analysis.

        Handles automatic covariate addition and passes pre-computed 2D arrays 
        directly to PRISM to bypass internal masking.

        Returns:
            PrismDataset: An initialized PRISM dataset ready for permutation testing.
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
            mask_img=self.mask_img,
            **self.glm_config
        )

    def network_sensitivity_analysis(self) -> Bunch:
        """Performs a sensitivity (overlap) analysis on cases.

        Returns:
            Bunch: Statistical results containing 'raw_sensstat' and 'sensstat' arrays.
        """
        if self.cases_control_labels is not None:
            network_data = self.network_data[self.cases_control_labels.astype(bool)]
        else:
            network_data = self.network_data

        return analysis.network_sensitivity_analysis(
            network_data=network_data,
            threshold=self.sensitivity_threshold,
            output_prefix=self.output_prefix,
            mask_img=self.mask_img
        )

    def network_glm_analysis(self) -> Bunch:
        """Runs a permutation-based GLM using the PRISM backend.

        Returns:
            Bunch: Results containing t-stats, p-values, etc. as 1D/2D arrays.
        """
        # 1. Instantiate the PRISM dataset
        prism_ds = self.prepare_glm_config()
        
        # 2. Run the permutation analysis via the analysis module's wrapper
        results = prism_ds.permutation_analysis()
        
        # 3. Export to NIfTI using our toolkit's I/O, not PRISM's
        if self.output_prefix and self.mask_img:
            saver = ResultSaver(output_prefix=self.output_prefix, mask_img=self.mask_img)
            saver.save_results(results)
            
        return results

    def network_conjunction_analysis(self) -> Bunch:
        """Conjoins group-level sensitivity with GLM results.

        Returns:
            Bunch: Results containing sensitivity, GLM, conjunction, and agreement maps.
        """

        sensitivity_results = self.network_sensitivity_analysis()
        glm_results = self.network_glm_analysis()

        return analysis.network_conjunction_analysis(
            sensitivity_results=sensitivity_results,
            glm_results=glm_results,
            output_prefix=self.output_prefix,
            mask_img=self.mask_img
        )

    def network_sensitivity_permutation_analysis(self) -> Bunch:
        """
        Performs permutation analysis for network sensitivity maps.

        Returns:
            Bunch: Contains permuted sensitivity maps for all permutations.
        """
        prism_ds = self.prepare_glm_config()
        permutation_indices = prism_ds.generate_permutation_indices()
        permutation_indices = list(permutation_indices.values())[0]
        return analysis.network_sensitivity_permutation_analysis(
            full_network_data=self.network_data,
            permuted_indices=permutation_indices,
            cases_control_labels=self.cases_control_labels,
            threshold=self.sensitivity_threshold,
            group_threshold=self.group_sensitivity_threshold,
            output_prefix=self.output_prefix,
            mask_img=self.mask_img
        )
