import pandas as pd
import numpy as np
import warnings

# Suppress pandas FutureWarnings for clean output
warnings.simplefilter(action='ignore', category=FutureWarning)

class BinningOptimizer:
    def __init__(self, min_write_off_ratio=0.10, initial_rest_bins=10):
        self.min_write_off_ratio = min_write_off_ratio
        self.initial_rest_bins = initial_rest_bins
        self.first_bin_cutoff = None
        self.final_boundaries = None

    def _find_critical_cutoff(self, df):
        """Step 1: Maximize first bin width (starting from min score) subject to C1 and C2."""
        samples = df['sample'].unique()
        totals_wo = df.groupby('sample')['write_off'].sum()
        
        # Evaluate at 1000 quantiles for precision
        candidates = df['score'].quantile(np.linspace(0, 1, 1000)).unique()
        
        best_cutoff = None
        
        # Iterate from highest possible score down to lowest to MAXIMIZE the bin width
        for cut in reversed(candidates):
            c1_passed = True
            c2_passed = True
            
            for s in samples:
                mask = (df['sample'] == s) & (df['score'] <= cut)
                sub_bin = df[mask]
                
                # Constraint 1: Loss-adjusted return < 0
                la_return = sub_bin['return'].sum() - sub_bin['loss'].sum()
                if la_return >= 0:
                    c1_passed = False
                    break
                    
                # Constraint 2: Write-off >= min_write_off_ratio
                total_wo = totals_wo[s]
                wo_ratio = (sub_bin['write_off'].sum() / total_wo) if total_wo > 0 else 0
                
                if wo_ratio < self.min_write_off_ratio:
                    c2_passed = False
                    break
                    
            if c1_passed and c2_passed:
                best_cutoff = cut
                break
                
        if best_cutoff is None:
            raise ValueError("No cutoff found that satisfies C1 (<0 return) and C2 (write-off ratio) for all samples.")
            
        return best_cutoff

    def _ensure_monotonicity(self, df):
        """Step 2: Iteratively merge bins to satisfy monotonically DECREASING default rates."""
        samples = df['sample'].unique()
        
        df_rest = df[df['score'] > self.first_bin_cutoff]
        if df_rest.empty:
            return [df['score'].min() - 1e-5, self.first_bin_cutoff, df['score'].max() + 1e-5]
            
        rest_quantiles = np.linspace(0, 1, self.initial_rest_bins + 1)[1:] 
        rest_boundaries = df_rest['score'].quantile(rest_quantiles).unique().tolist()
        
        boundaries = [df['score'].min() - 1e-5, self.first_bin_cutoff] + rest_boundaries
        boundaries[-1] += 1e-5 
        
        while len(boundaries) > 3:
            monotonic = True
            merge_idx = -1
            
            for s in samples:
                sample_df = df[df['sample'] == s].copy()
                sample_df['bin'] = pd.cut(sample_df['score'], bins=boundaries, include_lowest=True)
                
                # Calculate default rate
                dr = sample_df.groupby('bin', observed=False)['is_default'].mean().fillna(0).values
                
                # CHECK: Default rate must be DECREASING as score (bin index) increases
                for i in range(len(dr) - 1):
                    if dr[i] <= dr[i+1]: # Violation: rate stayed flat or increased
                        monotonic = False
                        # If violation involves the first optimized bin, merge subsequent bins
                        merge_idx = 2 if i == 0 else i + 1
                        break
                
                if not monotonic:
                    break
                    
            if monotonic:
                break
            else:
                boundaries.pop(merge_idx)
                
        return boundaries

    def fit(self, df):
        print("Finding critical point for First Bin...")
        self.first_bin_cutoff = self._find_critical_cutoff(df)
        print(f"First bin upper boundary locked at score: {self.first_bin_cutoff:.4f}")
        
        print("Optimizing remaining bins for monotonic decreasing default rates...")
        self.final_boundaries = self._ensure_monotonicity(df)
        print(f"Final optimal boundaries: {self.final_boundaries}")
        
        return self.final_boundaries


if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate mock data: 1 dev sample, 6 test samples
    samples = ['dev'] + [f'test_{i}' for i in range(1, 7)]
    data = []
    
    for s in samples:
        n_obs = 5000
        score = np.random.normal(500, 150, n_obs)
        
        # High score = High Return, Low Loss
        loss = np.random.uniform(100, 300, n_obs) - (score * 0.2)
        ret = np.random.uniform(0, 50, n_obs) + (score * 0.1)
        write_off = np.random.uniform(10, 50, n_obs)
        
        # Default probability DECREASES as score increases
        prob_default = 1 / (1 + np.exp((score - 500) / 100))
        is_default = np.random.binomial(1, prob_default)
        
        data.append(pd.DataFrame({
            'sample': s, 'score': score, 'return': ret, 'loss': loss, 
            'write_off': write_off, 'is_default': is_default
        }))
        
    df_all = pd.concat(data, ignore_index=True)
    
    # Run the Optimizer
    optimizer = BinningOptimizer(min_write_off_ratio=0.10, initial_rest_bins=15)
    
    try:
        final_bins = optimizer.fit(df_all)
        
        # --- VALIDATION FOR ALL SAMPLES ---
        all_results = []
        
        for s in samples:
            sample_df = df_all[df_all['sample'] == s].copy()
            sample_df['final_bin'] = pd.cut(sample_df['score'], bins=final_bins)
            
            res = sample_df.groupby('final_bin', observed=False).agg(
                obs_count=('score', 'count'),
                total_return=('return', 'sum'),
                total_loss=('loss', 'sum'),
                write_off_sum=('write_off', 'sum'),
                default_rate=('is_default', 'mean')
            ).reset_index()
            
            res['sample'] = s
            res['loss_adj_return'] = res['total_return'] - res['total_loss']
            res['write_off_pct'] = res['write_off_sum'] / sample_df['write_off'].sum()
            
            # Reorder columns
            res = res[['sample', 'final_bin', 'obs_count', 'loss_adj_return', 'write_off_pct', 'default_rate']]
            all_results.append(res)
            
        # Combine all validation results into one large DataFrame
        final_validation_report = pd.concat(all_results, ignore_index=True)
        
        print("\n" + "="*80)
        print("CONSOLIDATED VALIDATION RESULTS (ALL SAMPLES)")
        print("="*80)
        # Setting display options to show all rows
        with pd.option_context('display.max_rows', None, 'display.width', 1000):
            print(final_validation_report)
            
    except ValueError as e:
        print(f"Error: {e}")