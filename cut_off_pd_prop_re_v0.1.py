import pandas as pd
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

class BinningOptimizer:
    def __init__(self, min_charge_off_ratio=0.10, initial_rest_bins=10,
                 first_bin_max_pct=1.0, second_bin_max_pct=1.0, third_bin_max_pct=1.0,
                 remaining_bin_max_pct=0.1):
        """
        Args:
            min_charge_off_ratio: Minimum charge-off ratio required for the first bin
            initial_rest_bins: Initial number of granular bins for remaining samples
            first_bin_max_pct: Max sample proportion for the first bin (0~1)
            second_bin_max_pct: Max sample proportion for the second bin (0~1)
            third_bin_max_pct: Max sample proportion for the third bin (0~1)
            remaining_bin_max_pct: Max sample proportion for 4th+ bins (0~1)
        """
        self.min_charge_off_ratio = min_charge_off_ratio
        self.initial_rest_bins = initial_rest_bins
        self.first_bin_max_pct = first_bin_max_pct
        self.second_bin_max_pct = second_bin_max_pct
        self.third_bin_max_pct = third_bin_max_pct
        self.remaining_bin_max_pct = remaining_bin_max_pct
        self.first_bin_cutoff = None
        self.final_boundaries = None

    def _find_critical_cutoff(self, df):
        """Step 1: Find the maximum valid cutoff for the first bin that satisfies all constraints"""
        samples = df['sample'].unique()
        totals_co = df.groupby('sample')['charge_off'].sum()
        total_counts = df.groupby('sample').size()

        candidates = df['score'].quantile(np.linspace(0, 1, 1000)).unique()
        best_cutoff = None

        for cut in reversed(candidates):
            c1_passed = True
            c2_passed = True
            pct_passed = True

            for s in samples:
                mask = (df['sample'] == s) & (df['score'] <= cut)
                sub_bin = df[mask]

                # Constraint 1: LAR = return - charge_off < 0
                lar = sub_bin['return'].sum() - sub_bin['charge_off'].sum()
                if lar >= 0:
                    c1_passed = False
                    break

                # Constraint 2: Charge-off ratio >= threshold
                total_co = totals_co[s]
                co_ratio = (sub_bin['charge_off'].sum() / total_co) if total_co > 0 else 0
                if co_ratio < self.min_charge_off_ratio:
                    c2_passed = False
                    break

                # Constraint: First bin sample proportion limit
                sample_prop = len(sub_bin) / total_counts[s]
                if sample_prop > self.first_bin_max_pct:
                    pct_passed = False
                    break

            if c1_passed and c2_passed and pct_passed:
                best_cutoff = cut
                break

        if best_cutoff is None:
            raise ValueError("No valid cutoff found for the first bin that meets all constraints")
        return best_cutoff

    def _check_bin_proportions(self, df, boundaries):
        """Validate sample proportion constraints for all bins:
        1. Custom limits for first 3 bins
        2. Unified limit for remaining bins
        """
        samples = df['sample'].unique()
        total_counts = df.groupby('sample').size()

        for s in samples:
            sample_df = df[df['sample'] == s].copy()
            sample_df['bin'] = pd.cut(sample_df['score'], bins=boundaries, include_lowest=True)
            bin_counts = sample_df.groupby('bin', observed=False).size()

            for idx, (_, cnt) in enumerate(bin_counts.items()):
                bin_prop = cnt / total_counts[s]
                if idx == 0 and bin_prop > self.first_bin_max_pct:
                    return False
                elif idx == 1 and bin_prop > self.second_bin_max_pct:
                    return False
                elif idx == 2 and bin_prop > self.third_bin_max_pct:
                    return False
                elif idx >= 3 and bin_prop > self.remaining_bin_max_pct:
                    return False
        return True

    def _ensure_monotonicity(self, df):
        """Step 2: Merge bins via dynamic programming to ensure monotonic default rates + proportion constraints"""
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
                dr = sample_df.groupby('bin', observed=False)['is_default'].mean().fillna(0).values
                
                for i in range(len(dr)-1):
                    if dr[i] <= dr[i+1]:
                        monotonic = False
                        merge_idx = 2 if i == 0 else i+1
                        break
                if not monotonic:
                    break

            if monotonic:
                break

            # Optimized merging logic: iterate all valid positions
            merged_success = False
            for candidate_idx in range(2, len(boundaries)-1):
                temp_bound = boundaries.copy()
                temp_bound.pop(candidate_idx)
                if self._check_bin_proportions(df, temp_bound):
                    boundaries = temp_bound
                    merged_success = True
                    break

            if not merged_success:
                raise RuntimeError("Failed to meet monotonicity and proportion constraints. Relax parameters and try again.")
        return boundaries

    def fit(self, df):
        print("Finding critical cutoff for the first bin...")
        self.first_bin_cutoff = self._find_critical_cutoff(df)
        print(f"First bin upper boundary: {self.first_bin_cutoff:.4f}")

        print("Optimizing remaining bins...")
        self.final_boundaries = self._ensure_monotonicity(df)
        print(f"Final bin boundaries: {[round(x,4) for x in self.final_boundaries]}")
        return self.final_boundaries

# ==========================================
# Demo: Data Generation & Model Execution
# ==========================================
if __name__ == "__main__":
    np.random.seed(42)
    samples = ['dev'] + [f'test_{i}' for i in range(1, 7)]
    data = []

    for s in samples:
        n_obs = 5000
        score = np.random.normal(500, 150, n_obs)
        
        # Business Rule: Lower score = Lower return + Higher charge_off (LAR < 0)
        return_val = np.random.uniform(10, 30, n_obs) - (score * 0.02)
        charge_off = np.random.uniform(40, 80, n_obs) + (600 - score)*0.1
        
        # New feature: Outstanding balance
        oustanding_balance = np.random.uniform(1000, 50000, n_obs)
        
        # Default label (lower score = higher default probability)
        prob_default = 1 / (1 + np.exp((score - 500) / 100))
        is_default = np.random.binomial(1, prob_default)

        df_temp = pd.DataFrame({
            'sample': s,
            'score': score,
            'return': return_val,
            'charge_off': charge_off,
            'oustanding_balance': oustanding_balance,
            'is_default': is_default
        })
        data.append(df_temp)

    df_all = pd.concat(data, ignore_index=True)

    # Initialize optimizer
    optimizer = BinningOptimizer(
        min_charge_off_ratio=0.10,
        initial_rest_bins=25,
        first_bin_max_pct=0.12,
        second_bin_max_pct=0.15,
        third_bin_max_pct=0.08,
        remaining_bin_max_pct=0.12
    )

    try:
        bins = optimizer.fit(df_all)
        
        # Generate binning report for development sample
        dev_df = df_all[df_all['sample'] == 'dev'].copy()
        dev_df['bin'] = pd.cut(dev_df['score'], bins=bins)
        
        report = dev_df.groupby('bin', observed=False).agg(
            sample_count=('score', 'count'),
            total_return=('return', 'sum'),
            total_charge_off=('charge_off', 'sum'),
            total_outstanding_balance=('oustanding_balance', 'sum'),
            default_rate=('is_default', 'mean')
        ).reset_index()

        # Calculate core metrics
        report['loss_adjusted_return(LAR)'] = report['total_return'] - report['total_charge_off']
        report['sample_pct(%)'] = report['sample_count'] / report['sample_count'].sum() * 100
        report['charge_off_pct(%)'] = report['total_charge_off'] / report['total_charge_off'].sum() * 100

        print("\n==================== Development Sample Binning Report ====================")
        print(report.round(4))

    except Exception as e:
        print(f"Error: {e}")