# Command Line Interface (CLI)

The `lnm-toolkit` provides a robust and easy-to-use command-line interface, `lnm`, which automates many of the steps involved in lesion network mapping.

## Overview

The `lnm` tool is designed to take a CSV file containing subject IDs and NIfTI filepaths, filter the data, and perform statistical analysis in one step.

```bash
lnm --csv participants.csv --subject-col subject --network-col t --roi-col roi_2mm --output-prefix results/my_analysis ...
```

## Basic Usage

### Case-Control Conjunction Analysis

The most common workflow involves comparing cases and controls in a GLM and then conjoining that with the subject agreement map (sensitivity) of the cases.

```bash
lnm --csv participants.csv 
    --subject-col subject 
    --network-col t 
    --roi-col roi_2mm 
    --output-prefix results/aphasia_analysis 
    --analysis conjunction 
    --filter-col wab_type 
    --filter-values Broca NoAphasia 
    --contrast-col wab_type 
    --contrast-values Broca NoAphasia 
    --covariates age_at_stroke 
    --add-intercept 
    --n-permutations 1000
```

## Available Options

- `--csv`: Path to the CSV file containing subject data (metadata and filepaths).
- `--subject-col`: Column name for subject IDs.
- `--network-col`: Column name for subject-level network file paths (NIfTIs).
- `--roi-col`: Column name for subject-level ROI mask file paths (NIfTIs).
- `--output-prefix`: Prefix for all saved NIfTI output files.
- `--analysis`: Type of analysis to run. Options: `glm`, `sensitivity`, `conjunction`, `sensitivity-permutation`. Default: `conjunction`.
- `--filter-col`: (Optional) Column to use for filtering rows in the CSV.
- `--filter-values`: (Optional) One or more values from the filter column to keep.
- `--covariates`: (Optional) One or more columns to include as covariates in the GLM.
- `--contrast-col`: (Optional) Column to use for the primary group comparison (e.g., cases vs. controls).
- `--contrast-values`: (Optional) Two values from the contrast column to compare (e.g., `Broca NoAphasia`).
- `--control-roi-volume`: (Flag) Include subject-level ROI volume as a covariate in the GLM.
- `--control-roi-centrality`: (Flag) Include subject-level network centrality as a covariate in the GLM.
- `--add-intercept`: (Flag) Include an intercept (constant term) in the design matrix.
- `--n-permutations`: Number of permutations for statistical testing. Default: `1000`.
- `--sensitivity-threshold`: Individual subject Z-score threshold for binarization. Default: `7.0`.
- `--group-sensitivity-threshold`: Group-level overlap percentage threshold (0-1). Default: `0.75`.

## Interpreting Output

The tool will save multiple NIfTI images starting with your specified `--output-prefix`:
- `_vox_sensstat.nii.gz`: Overlap percentage map of the cases.
- `_vox_tstat.nii.gz`: T-statistic map from the GLM.
- `_vox_conjstat_fwep.nii.gz`: Conjunction map showing regions passing both sensitivity and FWEp thresholds.
- `_vox_conjstat_fdrp.nii.gz`: Conjunction map showing regions passing both sensitivity and FDRp thresholds.
- `_vox_agreementstat.nii.gz`: Signed agreement map.
