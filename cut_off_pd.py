import pandas as pd
import numpy as np

class BinningOptimizer:
    def __init__(self, min_wo_ratio=0.10, target_bins=5):
        self.min_wo_ratio = min_wo_ratio
        self.target_bins = target_bins
        self.first_bin_boundary = None
        self.all_boundaries = None

    def fit(self, df):
        samples = df['sample'].unique()
        
        # 1. Generate search candidates (percentiles 1 through 50)
        # We search the lower half of the score distribution for the "Risk Bin"
        potential_cuts = np.percentile(df['score'], np.linspace(1, 50, 100))
        
        best_cutoff = None
        diagnostic_log = []

        for cut in potential_cuts:
            sample_status = []
            all_pass = True
            
            for s in samples:
                sub = df[(df['sample'] == s) & (df['score'] <= cut)]
                if sub.empty:
                    all_pass = False
                    continue
                
                # Metrics
                la_return = sub['return'].sum() - sub['loss'].sum()
                wo_pct = sub['write_off'].sum() / df[df['sample'] == s]['write_off'].sum()
                
                # Constraints
                c1 = la_return < 0
                c2 = wo_pct >= self.min_wo_ratio
                
                if not (c1 and c2):
                    all_pass = False
                
                sample_status.append({'sample': s, 'C1_pass': c1, 'C2_pass': c2, 'LAR': la_return, 'WO%': wo_pct})
            
            if all_pass:
                best_cutoff = cut # Keep expanding to maximize width
            else:
                # If we already found a best_cutoff and now it's failing, we stop.
                if best_cutoff is not None:
                    break
                # Otherwise, keep a log of why we are failing for the first few cuts
                if len(diagnostic_log) < 5:
                    diagnostic_log.append({'cutoff': cut, 'status': sample_status})

        if best_cutoff is None:
            print("\n--- DIAGNOSTIC ALERT: Optimization Failed ---")
            print("Check the first candidate cutoff results:")
            diag_df = pd.DataFrame(diagnostic_log[0]['status'])
            print(diag_df.to_string(index=False))
            print("\nPossible Issues: ")
            print("- If LAR is > 0: Your riskiest bin is already profitable. C1 is impossible.")
            print("- If WO% is < 10%: The bin is too narrow. C2 is impossible.")
            raise ValueError("Could not find a valid first bin. See diagnostics above.")
        
        self.first_bin_boundary = best_cutoff
        
        # 2. Define Remaining Boundaries
        remaining_scores = df[df['score'] > best_cutoff]['score']
        other_cuts = np.percentile(remaining_scores, np.linspace(0, 100, self.target_bins))
        boundaries = np.sort(np.unique(np.concatenate([[df['score'].min() - 1e-5], [best_cutoff], other_cuts])))
        
        # 3. Monotonicity (Default Rate Increasing)
        self.all_boundaries = self._adjust_for_monotonicity(df, boundaries, samples)
        return self.all_boundaries

    def _adjust_for_monotonicity(self, df, bins, samples):
        current_bins = list(bins)
        while len(current_bins) > 3:
            failed_idx = -1
            for s in samples:
                temp = df[df['sample'] == s].copy()
                temp['b'] = pd.cut(temp['score'], bins=current_bins)
                dr = temp.groupby('b', observed=False)['is_default'].mean().values
                
                for i in range(len(dr) - 1):
                    if dr[i] >= dr[i+1]: # Violation: We want DR to increase
                        # Do not remove the first boundary (the optimized one)
                        failed_idx = i + 1 if (i + 1) != 1 else i + 2
                        break
                if failed_idx != -1: break
            
            if failed_idx == -1: break
            if failed_idx >= len(current_bins): failed_idx = len(current_bins) - 1
            current_bins.pop(failed_idx)
            
        return current_bins

# ==========================================
# Robust Data Generation (Ensures a "Red" Bin exists)
# ==========================================
def generate_robust_data():
    samples = ['Dev'] + [f'Test_{i}' for i in range(1, 7)]
    all_data = []
    for s in samples:
        n = 5000
        score = np.random.uniform(300, 850, n)
        
        # Higher score = Lower Default Prob
        p = 1 / (1 + np.exp((score - 550) / 60))
        is_default = np.random.binomial(1, p)
        
        # LOSS: Very high for scores < 400
        loss = np.where(score < 420, np.random.uniform(800, 1200), np.random.uniform(0, 50))
        # RETURN: Low for scores < 400
        ret = np.where(score < 420, np.random.uniform(0, 50), np.random.uniform(100, 400))
        wo = np.where(score < 450, np.random.uniform(50, 100), np.random.uniform(1, 10))
        
        all_data.append(pd.DataFrame({
            'sample': s, 'score': score, 'is_default': is_default,
            'loss': loss, 'return': ret, 'write_off': wo
        }))
    return pd.concat(all_data)

# Run the process
df = generate_robust_data()
opt = BinningOptimizer(min_wo_ratio=0.10, target_bins=6)

try:
    final_bins = opt.fit(df)
    
    # Generate report
    report_list = []
    for s in df['sample'].unique():
        sdf = df[df['sample'] == s].copy()
        sdf['bin'] = pd.cut(sdf['score'], bins=final_bins)
        stats = sdf.groupby('bin', observed=False).agg(
            DR=('is_default', 'mean'),
            LAR=('return', lambda x: x.sum() - sdf.loc[x.index, 'loss'].sum()),
            WO_Pct=('write_off', lambda x: x.sum() / sdf['write_off'].sum())
        ).reset_index()
        stats.insert(0, 'sample', s)
        report_list.append(stats)

    print("\n--- SUCCESS! Final Validated Bins ---")
    print(pd.concat(report_list).to_string(index=False))

except ValueError as e:
    print(e)