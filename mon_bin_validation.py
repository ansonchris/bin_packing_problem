import pandas as pd
import numpy as np

def calculate_bin_metrics_to_df(cutoffs, train_dfs, test_dfs):
    """
    Binning verification core function: calculate all bin statistics based on given score cutoffs, return 4 DataFrames
    Parameters:
        cutoffs (list[float]): Sorted score bin boundaries (output from solve function, ascending order)
        train_dfs (list[pd.DataFrame]): List of 10 train datasets (each contains 'score' and 'target' columns)
        test_dfs (list[pd.DataFrame]): List of 6 validation datasets (each contains 'score' and 'target' columns)
    Returns:
        df_single_train (pd.DataFrame): Statistics of each bin in 10 individual train sets
        df_single_val (pd.DataFrame): Statistics of each bin in 6 individual validation sets
        df_merge_train (pd.DataFrame): Statistics of merged 10 train sets by bin
        df_merge_val (pd.DataFrame): Statistics of merged 6 validation sets by bin
    """
    # -------------------------- 1. Generate bin interval labels (Left-closed & Right-open rule) --------------------------
    bin_labels = []
    n_bins = len(cutoffs) - 1
    for i in range(n_bins):
        if i == n_bins - 1:
            bin_labels.append(f"[{cutoffs[i]:.2f}, {cutoffs[i+1]:.2f}]")  # Last bin includes upper bound
        else:
            bin_labels.append(f"[{cutoffs[i]:.2f}, {cutoffs[i+1]:.2f})") # Other bins: left closed, right open

    # -------------------------- 2. Iterate 10 train sets → Generate df_single_train --------------------------
    single_train_data = []
    for idx, df in enumerate(train_dfs, 1):
        dataset_id = f"Tr{idx:02d}"  # Dataset ID: Tr01 ~ Tr10
        total_cnt = len(df)          # Total customers of current single dataset
        # Bin data by given cutoffs
        df['bin'] = pd.cut(df['score'], bins=cutoffs, include_lowest=True, labels=bin_labels)
        bin_count = df['bin'].value_counts().sort_index()
        
        # Fill data for all bins (0 if no customers in the bin)
        for bin_name in bin_labels:
            bin_cnt = bin_count.get(bin_name, 0)
            bin_ratio = round(bin_cnt / total_cnt * 100, 2)  # Percentage, keep 2 decimal places
            single_train_data.append({
                "dataset_id": dataset_id,
                "bin_interval": bin_name,
                "customer_count": bin_cnt,
                "percentage(%)": bin_ratio,
                "total_dataset_customers": total_cnt
            })
    df_single_train = pd.DataFrame(single_train_data)

    # -------------------------- 3. Iterate 6 validation sets → Generate df_single_val --------------------------
    single_val_data = []
    for idx, df in enumerate(test_dfs, 1):
        dataset_id = f"Ts{idx:02d}"  # Dataset ID: Ts01 ~ Ts06
        total_cnt = len(df)          # Total customers of current single dataset
        # Bin data by given cutoffs
        df['bin'] = pd.cut(df['score'], bins=cutoffs, include_lowest=True, labels=bin_labels)
        bin_count = df['bin'].value_counts().sort_index()
        
        # Fill data for all bins (0 if no customers in the bin)
        for bin_name in bin_labels:
            bin_cnt = bin_count.get(bin_name, 0)
            bin_ratio = round(bin_cnt / total_cnt * 100, 2)  # Percentage, keep 2 decimal places
            single_val_data.append({
                "dataset_id": dataset_id,
                "bin_interval": bin_name,
                "customer_count": bin_cnt,
                "percentage(%)": bin_ratio,
                "total_dataset_customers": total_cnt
            })
    df_single_val = pd.DataFrame(single_val_data)

    # -------------------------- 4. Merge 10 train sets → Generate df_merge_train --------------------------
    train_merged = pd.concat(train_dfs, ignore_index=True)
    train_merged['bin'] = pd.cut(train_merged['score'], bins=cutoffs, include_lowest=True, labels=bin_labels)
    # Group statistics: total customers & bad sample count per bin
    merge_train_stats = train_merged.groupby('bin', observed=False).agg(
        total_customers=('score', 'count'),
        bad_sample_count=('target', 'sum')
    ).sort_index().reset_index()
    # Calculate bad sample rate (%)
    merge_train_stats['bad_sample_rate(%)'] = round(merge_train_stats['bad_sample_count'] / merge_train_stats['total_customers'] * 100, 2)
    # Add merged global statistics
    merge_train_stats['total_merged_customers'] = len(train_merged)
    merge_train_stats['total_merged_bad_samples'] = train_merged['target'].sum()
    df_merge_train = merge_train_stats.rename(columns={'bin': 'bin_interval'})

    # -------------------------- 5. Merge 6 validation sets → Generate df_merge_val --------------------------
    val_merged = pd.concat(test_dfs, ignore_index=True)
    val_merged['bin'] = pd.cut(val_merged['score'], bins=cutoffs, include_lowest=True, labels=bin_labels)
    # Group statistics: total customers & bad sample count per bin
    merge_val_stats = val_merged.groupby('bin', observed=False).agg(
        total_customers=('score', 'count'),
        bad_sample_count=('target', 'sum')
    ).sort_index().reset_index()
    # Calculate bad sample rate (%)
    merge_val_stats['bad_sample_rate(%)'] = round(merge_val_stats['bad_sample_count'] / merge_val_stats['total_customers'] * 100, 2)
    # Add merged global statistics
    merge_val_stats['total_merged_customers'] = len(val_merged)
    merge_val_stats['total_merged_bad_samples'] = val_merged['target'].sum()
    df_merge_val = merge_val_stats.rename(columns={'bin': 'bin_interval'})

    # Return 4 DataFrames in fixed order
    return df_single_train, df_single_val, df_merge_train, df_merge_val

# =====================  Calling Example (Replace with your real data directly) =====================
if __name__ == "__main__":
    # -------------------------- Replace with your real data ↓ --------------------------
    # 1. Your bin boundaries (output from solve function, must be sorted in ascending order)
    optimal_cutoffs = [300.0, 420.5, 510.2, 605.8, 720.0, 850.0]
    
    # 2. Simulate 10 train sets & 6 validation sets (REMOVE this part and use your real data)
    def gen(n):
        s = np.random.randint(300, 850, n)
        target = np.random.binomial(1, 1/(1+np.exp((s-580)/50)))
        return pd.DataFrame({'score': s, 'target': target})
    train_datasets = [gen(1500) for _ in range(10)]  # 10 train datasets
    val_datasets = [gen(800) for _ in range(6)]      # 6 validation datasets

    # -------------------------- Execute function & get 4 DataFrames ↓ --------------------------
    df_single_train, df_single_val, df_merge_train, df_merge_val = calculate_bin_metrics_to_df(
        cutoffs=optimal_cutoffs,
        train_dfs=train_datasets,
        test_dfs=val_datasets
    )
    
    with pd.ExcelWriter('binning_verification_results.xlsx', engine='openpyxl') as writer:
        df_single_train.to_excel(writer, sheet_name='1_Single_Train_Sets', index=False)
        df_single_val.to_excel(writer, sheet_name='2_Single_Validation_Sets', index=False)
        df_merge_train.to_excel(writer, sheet_name='3_Merged_Train_Sets', index=False)
        df_merge_val.to_excel(writer, sheet_name='4_Merged_Validation_Sets', index=False)


