import pandas as pd
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

class BinningOptimizer:
    def __init__(self, min_write_off_ratio=0.10, initial_rest_bins=10,
                 first_bin_max_pct=1.0, second_bin_max_pct=1.0, third_bin_max_pct=1.0):
        """
        Args:
            min_write_off_ratio: Minimum proportion of write-offs required in the first bin.
            initial_rest_bins: Number of granular bins for the remaining data before merging.
            first_bin_max_pct: Max allowed sample proportion in the first bin (0~1).
            second_bin_max_pct: Max allowed sample proportion in the second bin.
            third_bin_max_pct: Max allowed sample proportion in the third bin.
        """
        self.min_write_off_ratio = min_write_off_ratio
        self.initial_rest_bins = initial_rest_bins
        self.first_bin_max_pct = first_bin_max_pct
        self.second_bin_max_pct = second_bin_max_pct
        self.third_bin_max_pct = third_bin_max_pct
        self.first_bin_cutoff = None
        self.final_boundaries = None

    def _find_critical_cutoff(self, df):
        """Step 1: Maximize first bin width subject to C1, C2, and first-bin sample proportion constraint."""
        samples = df['sample'].unique()
        totals_wo = df.groupby('sample')['write_off'].sum()
        total_counts = df.groupby('sample').size()   # total rows per sample

        # Evaluate at 1000 quantiles
        candidates = df['score'].quantile(np.linspace(0, 1, 1000)).unique()

        best_cutoff = None
        # Iterate from highest score down to maximize width
        for cut in reversed(candidates):
            c1_passed = True
            c2_passed = True
            pct_passed = True

            for s in samples:
                mask = (df['sample'] == s) & (df['score'] <= cut)
                sub_bin = df[mask]

                # Constraint 1: Loss-adjusted return < 0
                la_return = sub_bin['return'].sum() - sub_bin['loss'].sum()
                if la_return >= 0:
                    c1_passed = False
                    break

                # Constraint 2: Write-off ratio >= min_write_off_ratio
                total_wo = totals_wo[s]
                wo_ratio = (sub_bin['write_off'].sum() / total_wo) if total_wo > 0 else 0
                if wo_ratio < self.min_write_off_ratio:
                    c2_passed = False
                    break

                # NEW: Sample proportion constraint for first bin
                sample_prop = len(sub_bin) / total_counts[s]
                if sample_prop > self.first_bin_max_pct:
                    pct_passed = False
                    break

            if c1_passed and c2_passed and pct_passed:
                best_cutoff = cut
                break

        if best_cutoff is None:
            raise ValueError("No cutoff found satisfying C1, C2, and first-bin sample proportion constraint for all samples.")
        return best_cutoff

    def _check_bin_proportions(self, df, boundaries):
        """Check that the second and third bins (if exist) satisfy sample proportion constraints for all samples."""
        samples = df['sample'].unique()
        total_counts = df.groupby('sample').size()

        for s in samples:
            sample_df = df[df['sample'] == s].copy()
            sample_df['bin'] = pd.cut(sample_df['score'], bins=boundaries, include_lowest=True)
            bin_counts = sample_df.groupby('bin', observed=False).size()
            # Convert bin interval to string index for easy access; bins are ordered.
            bin_labels = bin_counts.index.tolist()
            if len(bin_labels) >= 2:
                second_prop = bin_counts.iloc[1] / total_counts[s]
                if second_prop > self.second_bin_max_pct:
                    return False
            if len(bin_labels) >= 3:
                third_prop = bin_counts.iloc[2] / total_counts[s]
                if third_prop > self.third_bin_max_pct:
                    return False
        return True

    def _ensure_monotonicity(self, df):
        """Step 2: Iteratively merge remaining bins to satisfy C3 (monotonic default rates)
        and sample proportion constraints for 2nd and 3rd bins."""
        samples = df['sample'].unique()

        df_rest = df[df['score'] > self.first_bin_cutoff]
        if df_rest.empty:
            return [df['score'].min() - 1e-5, self.first_bin_cutoff, df['score'].max() + 1e-5]

        # Create initial granular boundaries for the rest
        rest_quantiles = np.linspace(0, 1, self.initial_rest_bins + 1)[1:]
        rest_boundaries = df_rest['score'].quantile(rest_quantiles).unique().tolist()

        boundaries = [df['score'].min() - 1e-5, self.first_bin_cutoff] + rest_boundaries
        boundaries[-1] += 1e-5

        # Check initial proportions (optional, but good practice)
        if not self._check_bin_proportions(df, boundaries):
            # Try to increase initial_rest_bins to make initial bins smaller? Or just warn.
            print("Warning: Initial bins violate 2nd/3rd bin proportion constraints. Try increasing initial_rest_bins.")

        while len(boundaries) > 3:
            monotonic = True
            merge_idx = -1
            # Find first violation of monotonicity across samples
            for s in samples:
                sample_df = df[df['sample'] == s].copy()
                sample_df['bin'] = pd.cut(sample_df['score'], bins=boundaries, include_lowest=True)
                dr = sample_df.groupby('bin', observed=False)['is_default'].mean().fillna(0).values
                for i in range(len(dr) - 1):
                    if dr[i] <= dr[i+1]:
                        monotonic = False
                        if i == 0:
                            merge_idx = 2   # merge bin2 and bin3 (cannot touch bin1)
                        else:
                            merge_idx = i + 1
                        break
                if not monotonic:
                    break

            if monotonic:
                break

            # Try to merge the violating boundary, but first check proportion constraints after merge
            merged_boundaries = boundaries.copy()
            merged_boundaries.pop(merge_idx)
            if self._check_bin_proportions(df, merged_boundaries):
                boundaries = merged_boundaries
            else:
                # If this merge violates proportion constraints, we need to try a different merge.
                # Find another violation (i.e., try next i or next sample). Since the current merge_idx
                # is the first violation, we must adjust. Simple strategy: force merge at a different index.
                # Alternatively, we could increase initial_rest_bins and restart.
                # Here we attempt to merge the next possible boundary if it exists.
                alt_merged = False
                # Look for other violations at different positions (i from 0 to len(dr)-2)
                # We already know the first violation at (i, sample). Try to merge at a different i.
                # For simplicity, we try to merge the next boundary after merge_idx (if exists).
                for offset in [1, -1]:
                    alt_idx = merge_idx + offset
                    if 2 <= alt_idx < len(boundaries) - 1:
                        alt_boundaries = boundaries.copy()
                        alt_boundaries.pop(alt_idx)
                        if self._check_bin_proportions(df, alt_boundaries):
                            boundaries = alt_boundaries
                            alt_merged = True
                            break
                if not alt_merged:
                    raise RuntimeError("Cannot achieve monotonicity while respecting 2nd/3rd bin proportion constraints. "
                                       "Try increasing initial_rest_bins or relaxing max_pct parameters.")
        return boundaries

    def fit(self, df):
        """Executes the full binning optimization logic with additional proportion constraints."""
        print("Finding critical point for First Bin...")
        self.first_bin_cutoff = self._find_critical_cutoff(df)
        print(f"First bin upper boundary locked at score: {self.first_bin_cutoff:.4f}")

        print("Optimizing remaining bins for monotonic default rates and proportion constraints...")
        self.final_boundaries = self._ensure_monotonicity(df)
        print(f"Final optimal boundaries: {self.final_boundaries}")

        # Final validation of proportion constraints
        if not self._check_bin_proportions(df, self.final_boundaries):
            print("Warning: Final boundaries violate 2nd/3rd bin proportion constraints.")
        return self.final_boundaries
    
    
# ==========================================
# Example Usage & Dummy Data Generation
# ==========================================
if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate mock data: 1 dev sample, 6 test samples
    samples = ['dev'] + [f'test_{i}' for i in range(1, 7)]
    data = []
    
    for s in samples:
        n_obs = 5000
        # Score from 0 to 1000
        score = np.random.normal(500, 150, n_obs)
        
        # Loss and Return (lower scores = higher loss, lower return)
        loss = np.random.uniform(50, 200, n_obs) - (score * 0.1)
        ret = np.random.uniform(0, 100, n_obs) + (score * 0.1)
        
        # Write off expenses
        write_off = np.random.uniform(10, 50, n_obs)
        
        # Default flag (probability of default heavily increases with score for demonstration)
        prob_default = 1 / (1 + np.exp((score - 500) / 100))
        is_default = np.random.binomial(1, prob_default)
        
        df_temp = pd.DataFrame({
            'sample': s, 'score': score, 'return': ret, 'loss': loss, 
            'write_off': write_off, 'is_default': is_default
        })
        data.append(df_temp)
        
    df_all = pd.concat(data, ignore_index=True)
    
    # Run the Optimizer
    optimizer = BinningOptimizer(
                            min_write_off_ratio=0.10,
                            initial_rest_bins=20,           # 更细的初始分箱有助于满足比例约束
                            first_bin_max_pct=0.12,         # 首个分箱样本占比 ≤ 25%
                            second_bin_max_pct=0.15,        # 第二个分箱样本占比 ≤ 20%
                            third_bin_max_pct=0.05          # 第三个分箱样本占比 ≤ 15%
                        )
    final_bins = optimizer.fit(df_all)
    
    try:
        final_bins = optimizer.fit(df_all)
        
        # Display Results for the Development Sample
        dev_df = df_all[df_all['sample'] == 'dev'].copy()
        dev_df['final_bin'] = pd.cut(dev_df['score'], bins=final_bins)
        
        results = dev_df.groupby('final_bin', observed=False).agg(
            obs_count=('score', 'count'),
            total_return=('return', 'sum'),
            total_loss=('loss', 'sum'),
            write_off_sum=('write_off', 'sum'),
            default_rate=('is_default', 'mean')
        ).reset_index()
        
        results['loss_adj_return'] = results['total_return'] - results['total_loss']
        results['write_off_pct'] = results['write_off_sum'] / results['write_off_sum'].sum()
        
        print("\n--- Validation on Dev Sample ---")
        print(results[['final_bin', 'obs_count', 'loss_adj_return', 'write_off_pct', 'default_rate']])
        
    except ValueError as e:
        print(e)