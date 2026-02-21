# API Reference

The `lnm-toolkit` API is organized around several core modules:

- **[LNMDataset](dataset.md)**: The primary data manager and analysis entry point.
- **[Analysis](analysis.md)**: Standalone mathematical functions for sensitivity and conjunction mapping.
- **[Loaders](loaders.md)**: Convenience classes for loading and preprocessing NIfTI data from DataFrames.
- **[Utils](utils.md)**: Helper functions for spatial calculations, standardization, and thresholding.

## Key Classes

- `LNMDataset`: Handles spatial masking, data alignment, and method wrappers for analysis.
- `PandasDatasetLoader`: Automates the creation of a dataset from tabular data and NIfTI file paths.

## Primary Workflows

1. **Load Data**: Use `PandasDatasetLoader` to convert file paths into a structured `LNMDataset`.
2. **Process Spatial Data**: Call `ds.load_data()` to mask and flatten 3D NIfTI images into 2D arrays.
3. **Configure GLM**: Define design matrices, contrasts, and add anatomical covariates (volume/centrality).
4. **Analyze**: Run `network_glm_analysis`, `network_sensitivity_analysis`, or `network_conjunction_analysis`.
5. **Export**: Results are automatically exported to NIfTI space using the PRISM-integrated `ResultSaver`.
