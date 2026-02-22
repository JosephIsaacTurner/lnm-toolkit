import argparse
import pandas as pd
import numpy as np
import os
import sys
from .loaders import PandasDatasetLoader
from .dataset import LNMDataset

def main():
    parser = argparse.ArgumentParser(description="LNM Toolkit CLI")
    
    # Data loading arguments
    parser.add_argument("--csv", required=True, help="Path to the CSV file containing subject data.")
    parser.add_argument("--subject-col", required=True, help="Column name for subject IDs.")
    parser.add_argument("--network-col", required=True, help="Column name for network file paths.")
    parser.add_argument("--roi-col", required=True, help="Column name for ROI mask file paths.")
    parser.add_argument("--mask-img", help="Path to a master mask image (optional).")
    parser.add_argument("--output-prefix", required=True, help="Prefix for output file paths.")

    # Analysis selection
    parser.add_argument("--analysis", choices=["glm", "sensitivity", "conjunction", "sensitivity-permutation"], default="conjunction", help="Type of analysis to run.")

    # Filtering
    parser.add_argument("--filter-col", help="Column to filter by.")
    parser.add_argument("--filter-values", nargs="+", help="Values to keep in the filter column.")

    # GLM parameters
    parser.add_argument("--covariates", nargs="*", help="List of columns to use as covariates in the GLM.")
    parser.add_argument("--contrast-col", help="Column to use as the primary contrast (for GLM or sensitivity cases).")
    parser.add_argument("--contrast-values", nargs=2, help="Two values from contrast-col to compare (e.g., cases control). Only for GLM.")
    parser.add_argument("--control-roi-volume", action="store_true", help="Control for ROI volume in GLM.")
    parser.add_argument("--control-roi-centrality", action="store_true", help="Control for network centrality in GLM.")
    parser.add_argument("--add-intercept", action="store_true", help="Add an intercept to the design matrix.")
    parser.add_argument("--n-permutations", type=int, default=1000, help="Number of permutations for GLM/sensitivity permutations.")

    # Sensitivity parameters
    parser.add_argument("--sensitivity-threshold", type=float, default=7.0, help="Threshold for individual subject binarization.")
    parser.add_argument("--group-sensitivity-threshold", type=float, default=0.75, help="Percentage threshold for group-level overlap (0-1).")

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.csv)

    # Filter data if specified
    if args.filter_col and args.filter_values:
        df = df[df[args.filter_col].isin(args.filter_values)]
        print(f"Filtered dataset to {len(df)} subjects.")

    # Prepare cases_control_labels for sensitivity analysis
    cases_control_labels = None
    if args.contrast_col:
        if args.analysis in ["sensitivity", "conjunction", "sensitivity-permutation"]:
            # For sensitivity, we usually just want one group (cases)
            # If two contrast values are provided, the first one is considered 'cases'
            if args.contrast_values:
                cases_control_labels = (df[args.contrast_col] == args.contrast_values[0]).astype(int).values
            else:
                print("Warning: contrast-col specified but no contrast-values. Assuming all subjects are cases for sensitivity.")
                cases_control_labels = np.ones(len(df))

    # Prepare GLM components
    design_matrix = None
    contrast_matrix = None
    if args.analysis in ["glm", "conjunction"]:
        if args.covariates:
            design_matrix = df[args.covariates].values
        
        if args.contrast_col and args.contrast_values:
            # Simple t-test between two groups
            group_labels = (df[args.contrast_col] == args.contrast_values[0]).astype(int).values
            if design_matrix is not None:
                design_matrix = np.column_stack([group_labels, design_matrix])
            else:
                design_matrix = group_labels.reshape(-1, 1)
            
            # Contrast is [1, 0, 0, ...] for the group difference
            n_cols = design_matrix.shape[1]
            if args.add_intercept:
                n_cols += 1
            contrast_matrix = np.zeros(n_cols)
            contrast_matrix[0] = 1

    # Initialize Dataset
    loader = PandasDatasetLoader(
        df=df,
        subject_col=args.subject_col,
        network_col=args.network_col,
        mask_col=args.roi_col,
        mask_img=args.mask_img,
        design_matrix=design_matrix,
        contrast_matrix=contrast_matrix,
        cases_control_labels=cases_control_labels,
        output_prefix=args.output_prefix,
        control_roi_volume=args.control_roi_volume,
        control_roi_centrality=args.control_roi_centrality,
        add_intercept=args.add_intercept,
        n_permutations=args.n_permutations
    )
    
    ds = loader.load()
    ds.sensitivity_threshold = args.sensitivity_threshold
    ds.group_sensitivity_threshold = args.group_sensitivity_threshold

    # Run Analysis
    print(f"Running {args.analysis} analysis...")
    if args.analysis == "glm":
        ds.network_glm_analysis()
    elif args.analysis == "sensitivity":
        ds.network_sensitivity_analysis()
    elif args.analysis == "conjunction":
        ds.network_conjunction_analysis()
    elif args.analysis == "sensitivity-permutation":
        ds.network_sensitivity_permutation_analysis()
    
    print(f"Analysis complete. Results saved with prefix: {args.output_prefix}")

if __name__ == "__main__":
    main()
