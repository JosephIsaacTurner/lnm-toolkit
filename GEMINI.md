GEMINI.md : lnm-toolkit development guide
This document provides instructions for AI agents developing the lnm-toolkit library. Read carefully to understand the data flow, especially the integration with the prism statistical backend.

1. Core Architecture and Data Flow
LNMDataset is the spatial boss. It handles all NIfTI file loading and spatial masking.

PandasDatasetLoader is a convenience class that loads data from a pandas DataFrame and returns a fully initialized LNMDataset object.

Flatten everything. 3D NIfTI images are converted into 2D numpy arrays (n_subjects, n_voxels) using nilearn.maskers.NiftiMasker fitted to a master mask_img.

Math tools are separate. Analysis functions (network_sensitivity_analysis, network_conjunction_analysis, etc.) exist as standalone functions in analysis.py. They expect 2D or 1D numpy arrays as inputs, never NIfTI files.

Dataset wrappers. LNMDataset contains thin wrapper methods that pass its internal 2D arrays to the standalone functions for convenience.

2. PRISM Integration Instructions (CRITICAL)
The prism library has its own Dataset class that handles permutation testing. It can accept either NIfTI files or raw arrays. We are explicitly bypassing PRISM's internal spatial masking.

When writing the prepare_glm_config and network_glm_analysis integration, follow these exact rules:

Pass 2D Arrays Only: When instantiating prism.datasets.Dataset, pass data=self.network_data (the 2D numpy array), not the original NIfTI files.

Understand PRISM's Behavior: Because we pass a 2D numpy array, PRISM will evaluate is_nifti_like(data) as False.

PRISM will drop the mask: PRISM will ignore the mask_img argument internally and issue a warning ("Mask ignored for non-NIfTI data"). This is expected and desired.

No spatial stats in PRISM: Because PRISM treats our input as tabular data, it will route to permutation_analysis(), not permutation_analysis_nifti().

PRISM's ResultSaver: PRISM will return a sklearn.utils.Bunch object containing the statistical results as 1D/2D arrays. You must use `prism.preprocessing.ResultSaver` to inverse-transform these arrays back into 3D NIfTI space.

PRISM Implementation Blueprint
Python
# Inside LNMDataset

def prepare_glm_config(self, **kwargs) -> PrismDataset:
    # Pass our pre-masked 2D arrays directly to PRISM.
    # We pass design and contrast matrices as well.
    return PrismDataset(
        data=self.network_data,
        design=self.design_matrix,
        contrast=self.contrast_matrix,
        output_prefix=self.output_prefix,
        **kwargs
    )

def network_glm_analysis(self, **kwargs) -> Bunch:
    # 1. Instantiate the PRISM dataset
    prism_ds = self.prepare_glm_config(**kwargs)
    
    # 2. Run the permutation analysis (returns a Bunch of numpy arrays)
    results = prism_ds.permutation_analysis()
    
    # 3. Export to NIfTI using prism's ResultSaver
    if self.output_prefix and self.mask_img:
        from prism.preprocessing import ResultSaver
        saver = ResultSaver(output_prefix=self.output_prefix, mask_img=self.mask_img)
        saver.save_results(results)
        
    return results
3. Code Style and Preferences
Keep it minimal. Write ad-hoc, functional code meant for personal research use.

No over-engineering. Do not add heavy production-style boilerplate.

No excessive error handling. Avoid bloated try/except blocks. Fail fast on bad inputs.

Vectorize. Use numpy operations instead of python for loops wherever possible.

Formatting. Do not use em dashes in docstrings or comments. Use colons or commas instead.

4. Feature Development Backlog
- [x] utils.py: Finish standardize_data (scale continuous columns, ignore binary dummy variables).
- [x] dataset.py: Finish add_roi_volume_covar and add_roi_centrality_covar. Check for highly correlated columns in self.design_matrix before appending to avoid rank deficiency in the GLM.
- [x] analysis.py: Finish the network_conjunction_analysis and network_sensitivity_analysis functions, ensuring they properly route to save_results_to_nifti at the end.