# lnm/loaders.py

import pandas as pd
import os
from nilearn.image import mean_img, math_img
from sklearn.utils import Bunch

from .dataset import LNMDataset

class PandasDatasetLoader:
    """Helper class to initialize an `LNMDataset` from a pandas DataFrame.

    Automates the extraction of network paths and ROI paths from a DataFrame 
    and optionally handles master mask generation.

    Attributes:
        df (pd.DataFrame): The source DataFrame containing subject metadata and paths.
        subject_col (str): Column name for subject IDs.
        network_col (str): Column name for NIfTI network file paths.
        mask_col (str): Column name for NIfTI ROI mask file paths.
        mask_img (nib.Nifti1Image or str, optional): Pre-defined master mask.
        kwargs (dict): Additional parameters passed to `LNMDataset`.
    """
    def __init__(self, df, subject_col, network_col, mask_col, mask_img=None, **kwargs):
        """Initializes the loader with dataset-specific mapping."""
        self.df = df
        self.subject_col = subject_col
        self.network_col = network_col
        self.mask_col = mask_col
        self.mask_img = mask_img
        self.kwargs = kwargs

    def load(self):
        """Executes the data loading and returns an initialized `LNMDataset`.

        If no `mask_img` was provided, a master mask is automatically generated 
        by thresholding the mean of all input networks.

        Returns:
            LNMDataset: An initialized dataset object with flattened spatial data.
        """
        networks = self.df[self.network_col].tolist()
        roi_masks = self.df[self.mask_col].tolist()
        if self.mask_img is not None:
            mask_img = self.mask_img
        else:
            mask_img = math_img("np.where(img != 0, 1, 0)", img=mean_img(networks))

        ds = LNMDataset(
            networks=networks,
            mask_img=mask_img,
            roi_masks=roi_masks,
            **self.kwargs
        )
        ds.load_data()
        if 'design_matrix' not in self.kwargs:
            ds.design_matrix = self.df
        return ds
