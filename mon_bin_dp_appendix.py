import pandas as pd
import numpy as np

class GuaranteedOptimizer:
    def __init__(self, train_dfs, test_dfs, a=0.05, b=0.25, C=0.50):
        self.train_dfs = train_dfs
        self.test_dfs = test_dfs
        self.all_dfs = train_dfs + test_dfs
        self.n_train, self.n_test = len(train_dfs), len(test_dfs)
        self.a, self.b, self.C = a, b, C
        
        # KEY FIX: Tolerance buffer
        # We require a gap of at least 1e-6 to consider it "decreasing"
        self.tol = 1e-6 

    def _calculate_micro_stats(self, n_bins=100):
        combined = pd.concat(self.all_dfs)[['score', 'target']]
        # Use more micro-bins (100+) to minimize boundary errors
        _, self.bin_edges = pd.qcut(combined['score'], q=n_bins, retbins=True, duplicates='drop')
        self.bin_edges = sorted(list(set(self.bin_edges)))
        self.n_micro_bins = len(self.bin_edges) - 1
        
        # Pre-aggregate micro-bin stats
        self.micro_stats = np.zeros((self.n_micro_bins, len(self.all_dfs), 2))
        for idx, df in enumerate(self.all_dfs):
            cuts = pd.cut(df['score'], bins=self.bin_edges, include_lowest=True, labels=False)
            stats = df.groupby(cuts, observed=False)['target'].agg(['count', 'sum'])
            for b_idx in stats.index:
                if 0 <= b_idx < self.n_micro_bins:
                    self.micro_stats[b_idx, idx, 0] = stats.loc[b_idx, 'count']
                    self.micro_stats[b_idx, idx, 1] = stats.loc[b_idx, 'sum']

    def _check_monotonicity(self, prev_metrics, curr_metrics):
        """
        Ensures strict decreasing order with a safety buffer.
        """
        # Condition 3: Aggregate Monotonicity
        if curr_metrics['tr_br'] >= (prev_metrics['tr_br'] - self.tol):
            return False
        if curr_metrics['ts_br'] >= (prev_metrics['ts_br'] - self.tol):
            return False
            
        # Condition 4: Average Test Monotonicity
        if curr_metrics['avg_ts_br'] >= (prev_metrics['avg_ts_br'] - self.tol):
            return False
            
        return True

    # ... [Rest of the solve logic remains same, but calls _check_monotonicity] ...
    
    
    
#%%


import numpy as np
import pandas as pd

class BinningRepairKit:
    def __init__(self, train_dfs, test_dfs, a=0.05, b=0.25, C=0.50):
        self.all_dfs = train_dfs + test_dfs
        self.n_train = len(train_dfs)
        self.a, self.b, self.C = a, b, C

    def get_metrics(self, low, high, is_last):
        """Calculates exact metrics for a specific score range."""
        props, bad_rates = [], []
        tr_cnt, tr_bad = 0, 0
        ts_cnt, ts_bad = 0, 0
        set_test_brs = []

        for i, df in enumerate(self.all_dfs):
            mask = (df['score'] >= low) & (df['score'] <= high) if is_last else (df['score'] >= low) & (df['score'] < high)
            c = len(df[mask])
            b = df[mask]['target'].sum()
            total = len(df)
            p = c / total
            br = b / c if c > 0 else 0
            
            props.append(p)
            if i < self.n_train:
                tr_cnt += c; tr_bad += b
            else:
                ts_cnt += c; ts_bad += b
                set_test_brs.append(br)
        
        return {
            'props': np.array(props),
            'agg_tr_br': tr_bad / tr_cnt if tr_cnt > 0 else 0,
            'agg_ts_br': ts_bad / ts_cnt if ts_cnt > 0 else 0,
            'avg_ts_br': np.mean(set_test_brs) if set_test_brs else 0
        }

    def check_all_constraints(self, cutoffs):
        """Returns the index of the first bin that violates any constraint."""
        groups = []
        for i in range(len(cutoffs)-1):
            groups.append(self.get_metrics(cutoffs[i], cutoffs[i+1], i == len(cutoffs)-2))

        for i in range(len(groups)):
            # 1. Proportion Check
            if np.any(groups[i]['props'] < self.a) or np.any(groups[i]['props'] > self.b):
                return i, "Proportion"
            
            # 2. Monotonicity & Adj Sum (requires comparison with previous)
            if i > 0:
                prev, curr = groups[i-1], groups[i]
                # Mono Check: Rates must decrease as score increases
                if curr['agg_tr_br'] >= prev['agg_tr_br'] or \
                   curr['agg_ts_br'] >= prev['agg_ts_br'] or \
                   curr['avg_ts_br'] >= prev['avg_ts_br']:
                    return i, "Monotonicity"
                
                # Adj Sum Check
                if np.any(prev['props'] + curr['props'] > self.C):
                    return i, "AdjSum"
                    
        return None, "All Clear"

    def repair(self, cutoffs):
        current_bins = list(cutoffs)
        max_attempts = 10
        
        for attempt in range(max_attempts):
            idx, error_type = self.check_all_constraints(current_bins)
            if idx is None:
                print(f"Repair successful on attempt {attempt}!")
                return current_bins
            
            print(f"Attempt {attempt+1}: Fixing {error_type} violation at Bin {idx+1}")
            
            # MERGE STRATEGY
            # We remove the boundary that defines the violating bin.
            # If idx is 0, we must merge with the next one.
            # If idx is last, we must merge with the previous one.
            
            # Candidate 1: Merge with Left (remove current_bins[idx])
            if idx > 0:
                candidate_left = current_bins[:idx] + current_bins[idx+1:]
                c_idx, _ = self.check_all_constraints(candidate_left)
                if c_idx is None: 
                    current_bins = candidate_left
                    continue
            
            # Candidate 2: Merge with Right (remove current_bins[idx+1])
            if idx < len(current_bins) - 2:
                candidate_right = current_bins[:idx+1] + current_bins[idx+2:]
                c_idx, _ = self.check_all_constraints(candidate_right)
                if c_idx is None:
                    current_bins = candidate_right
                    continue

            # Candidate 3: Extended Merge (Previous 2 or Next 2)
            # This is a 'greedy' jump to simplify the bin structure
            print("Level 1 neighbors failed. Trying extended window merge...")
            if idx > 1:
                current_bins = current_bins[:idx-1] + current_bins[idx+1:]
            elif idx < len(current_bins) - 3:
                current_bins = current_bins[:idx+1] + current_bins[idx+3:]
            else:
                # Last resort: just remove the boundary causing the most trouble
                current_bins.pop(idx if idx > 0 else 1)

        return current_bins