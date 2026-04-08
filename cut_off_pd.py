import pandas as pd
import numpy as np
import warnings

# Suppress pandas FutureWarnings for clean output
warnings.simplefilter(action='ignore', category=FutureWarning)

class BinningOptimizer:
    def __init__(self, min_write_off_ratio=0.10, initial_rest_bins=10):
        """
        Args:
            min_write_off_ratio: Minimum proportion of write-offs required in the first bin.
            initial_rest_bins: Number of granular bins to initialize for the remaining data 
                               before agglomerative merging for monotonicity.
        """
        self.min_write_off_ratio = min_write_off_ratio
        self.initial_rest_bins = initial_rest_bins
        self.first_bin_cutoff = None
        self.final_boundaries = None

    def _find_critical_cutoff(self, df):
        """Step 1: Maximize first bin width subject to C1 and C2 for all samples."""
        samples = df['sample'].unique()
        totals_wo = df.groupby('sample')['write_off'].sum()
        
        # Evaluate at 1000 quantiles to balance speed and precision
        candidates = df['score'].quantile(np.linspace(0, 1, 1000)).unique()
        
        best_cutoff = None
        
        # Iterate from highest possible score down to lowest to MAXIMIZE the bin width
        for cut in reversed(candidates):
            c1_passed = True
            c2_passed = True
            
            for s in samples:
                # Filter first bin for current sample
                mask = (df['sample'] == s) & (df['score'] <= cut)
                sub_bin = df[mask]
                
                # Constraint 1: Loss-adjusted return < 0
                la_return = sub_bin['return'].sum() - sub_bin['loss'].sum()
                if la_return >= 0:
                    c1_passed = False
                    break
                    
                # Constraint 2: Write-off >= min_write_off_ratio
                # Handle division by zero if total write_off is 0
                total_wo = totals_wo[s]
                wo_ratio = (sub_bin['write_off'].sum() / total_wo) if total_wo > 0 else 0
                
                if wo_ratio < self.min_write_off_ratio:
                    c2_passed = False
                    break
                    
            if c1_passed and c2_passed:
                best_cutoff = cut
                break # Since we iterate in reverse, the first match is the maximum possible width
                
        if best_cutoff is None:
            raise ValueError("No cutoff found that satisfies both C1 (<0 return) and C2 (write-off ratio) for all samples.")
            
        return best_cutoff

    def _ensure_monotonicity(self, df):
        """Step 2: Iteratively merge remaining bins to satisfy C3 across all samples."""
        samples = df['sample'].unique()
        
        # Data outside the first bin
        df_rest = df[df['score'] > self.first_bin_cutoff]
        if df_rest.empty:
            return [df['score'].min() - 1e-5, self.first_bin_cutoff, df['score'].max() + 1e-5]
            
        # Create initial granular boundaries for the rest of the data
        rest_quantiles = np.linspace(0, 1, self.initial_rest_bins + 1)[1:] 
        rest_boundaries = df_rest['score'].quantile(rest_quantiles).unique().tolist()
        
        boundaries = [df['score'].min() - 1e-5, self.first_bin_cutoff] + rest_boundaries
        boundaries[-1] += 1e-5 # Ensure max value is captured
        
        while len(boundaries) > 3: # Need at least: Min, First Cutoff, Max
            monotonic = True
            merge_idx = -1
            
            for s in samples:
                sample_df = df[df['sample'] == s].copy()
                
                # Assign data to current bins
                sample_df['bin'] = pd.cut(sample_df['score'], bins=boundaries, include_lowest=True)
                
                # Calculate default rate (Constraint 3)
                dr = sample_df.groupby('bin', observed=False)['is_default'].mean().fillna(0).values
                
                # Check if strictly non-decreasing
                for i in range(len(dr) - 1):
                    if dr[i] >= dr[i+1]:
                        monotonic = False
                        
                        if i == 0:
                            # CRITICAL: If Bin 1 > Bin 2, we CANNOT alter boundary 1 (first_bin_cutoff)
                            # because it was optimized for C1 and C2.
                            # Instead, we merge Bin 2 and Bin 3 to raise the default rate of the second bin.
                            merge_idx = 2 
                        else:
                            # Merge bin `i` and `i+1` by dropping the boundary between them
                            merge_idx = i + 1 
                        break
                
                if not monotonic:
                    break # Break sample loop to perform the merge
                    
            if monotonic:
                break # All samples passed C3
            else:
                # Execute merge by removing the violating boundary
                boundaries.pop(merge_idx)
                
        return boundaries

    def fit(self, df):
        """Executes the full binning optimization logic."""
        print("Finding critical point for First Bin...")
        self.first_bin_cutoff = self._find_critical_cutoff(df)
        print(f"First bin upper boundary locked at score: {self.first_bin_cutoff:.4f}")
        
        print("Optimizing remaining bins for monotonic default rates...")
        self.final_boundaries = self._ensure_monotonicity(df)
        print(f"Final optimal boundaries: {self.final_boundaries}")
        
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
        prob_default = 1 / (1 + np.exp(-(score - 500) / 100))
        is_default = np.random.binomial(1, prob_default)
        
        df_temp = pd.DataFrame({
            'sample': s, 'score': score, 'return': ret, 'loss': loss, 
            'write_off': write_off, 'is_default': is_default
        })
        data.append(df_temp)
        
    df_all = pd.concat(data, ignore_index=True)
    
    # Run the Optimizer
    optimizer = BinningOptimizer(min_write_off_ratio=0.10, initial_rest_bins=15)
    
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