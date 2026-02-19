# lnm/loaders.py

import pandas as pd
import os
from nilearn.image import mean_img, math_img
from sklearn.utils import Bunch

from .dataset import LNMDataset

class PandasDatasetLoader:
    """
    Loads data from a pandas DataFrame and returns a fully initialized LNMDataset object.
    """
    def __init__(self, df, subject_col, network_col, mask_col, mask_img=None, **kwargs):
        self.df = df
        self.subject_col = subject_col
        self.network_col = network_col
        self.mask_col = mask_col
        self.mask_img = mask_img
        self.kwargs = kwargs

    def load(self):
        """
        Loads the data and returns a LNMDataset object.
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
