import pandas as pd
import numpy as np

class AdvancedBinningFramework:
    def __init__(self, train_dfs, test_dfs, a=0.05, b=0.25, C=0.50):
        self.all_dfs = train_dfs + test_dfs
        self.n_train, self.n_test = len(train_dfs), len(test_dfs)
        self.a, self.b, self.C = a, b, C
        self.eps = 1e-7 # Precision buffer

    # ==========================================
    # CORE METRIC CALCULATOR
    # ==========================================

    def _get_bin_metrics(self, low, high, is_last=False):
        """Calculates exact metrics for a score range."""
        props, bads, counts = [], [], []
        for i, df in enumerate(self.all_dfs):
            mask = (df['score'] >= low) & (df['score'] <= high) if is_last else (df['score'] >= low) & (df['score'] < high)
            c, b = len(df[mask]), df[mask]['target'].sum()
            props.append(c / len(df))
            counts.append(c); bads.append(b)
        
        props, counts, bads = np.array(props), np.array(counts), np.array(bads)
        
        # Check Proportion Bounds (a and b)
        if np.any(props < self.a - self.eps) or np.any(props > self.b + self.eps):
            return None

        tr_c, tr_b = counts[:self.n_train].sum(), bads[:self.n_train].sum()
        ts_c, ts_b = counts[self.n_train:].sum(), bads[self.n_train:].sum()
        test_brs = [bads[i]/counts[i] if counts[i] > 0 else 0 for i in range(self.n_train, len(self.all_dfs))]
        
        return {
            'props': props,
            'tr_br': tr_b/tr_c if tr_c > 0 else 0,
            'ts_br': ts_b/ts_c if ts_c > 0 else 0,
            'avg_ts_br': np.mean(test_brs)
        }

    def _get_all_bin_stats(self, cutoffs):
        """Helper to get stats for a full list of cutoffs."""
        stats = []
        for i in range(len(cutoffs) - 1):
            m = self._get_bin_metrics(cutoffs[i], cutoffs[i+1], i == len(cutoffs) - 2)
            if m is None: return None # Invalid cutoff set
            stats.append(m)
        return stats

    # ==========================================
    # STAGE 1: GLOBAL OPTIMIZATION
    # ==========================================

    def solve_stage1(self, n_micro=100):
        combined = pd.concat(self.all_dfs)[['score']]
        _, self.bin_edges = pd.qcut(combined['score'], q=n_micro, retbins=True, duplicates='drop')
        self.bin_edges = sorted(list(set(self.bin_edges)))
        n = len(self.bin_edges) - 1
        
        chains = {}
        for j in range(1, n + 1):
            for i in range(j):
                curr = self._get_bin_metrics(self.bin_edges[i], self.bin_edges[j], j == n)
                if curr is None: continue
                if i == 0:
                    chains[(i, j)] = (1, None)
                else:
                    best_len, best_pred = -1, None
                    for (pk, pi) in [k for k in chains if k[1] == i]:
                        prev = self._get_bin_metrics(self.bin_edges[pk], self.bin_edges[pi], False)
                        if curr['tr_br'] >= prev['tr_br'] - self.eps: continue
                        if curr['ts_br'] >= prev['ts_br'] - self.eps: continue
                        if curr['avg_ts_br'] >= prev['avg_ts_br'] - self.eps: continue
                        if np.any(prev['props'] + curr['props'] > self.C + self.eps): continue
                        
                        if chains[(pk, pi)][0] > best_len:
                            best_len, best_pred = chains[(pk, pi)][0], (pk, pi)
                    if best_pred: chains[(i, j)] = (best_len + 1, best_pred)

        final = max([k for k in chains if k[1] == n], key=lambda x: chains[x][0], default=None)
        if not final: return []
        path = []
        while final: path.append(final); final = chains[final][1]
        return [self.bin_edges[0]] + [self.bin_edges[k[1]] for k in reversed(path)]

    # ==========================================
    # STAGE 2: RECURSIVE REPAIR (BACKTRACKING)
    # ==========================================

    def repair_recursive(self, cutoffs):
        """
        The recursive core. Tries merging left and right at any violation point
        and returns the best valid sequence.
        """
        stats = self._get_all_bin_stats(cutoffs)
        
        # 1. Find the first violation
        violation_idx = None
        for i in range(1, len(stats)):
            prev, curr = stats[i-1], stats[i]
            # Check Monotonicity and Adj Sum
            mono_fail = (curr['tr_br'] >= prev['tr_br'] - self.eps) or \
                        (curr['ts_br'] >= prev['ts_br'] - self.eps) or \
                        (curr['avg_ts_br'] >= prev['avg_ts_br'] - self.eps)
            adj_fail = np.any(prev['props'] + curr['props'] > self.C + self.eps)
            
            if mono_fail or adj_fail:
                violation_idx = i
                break
        
        # 2. Base Case: No violations found
        if violation_idx is None:
            return cutoffs

        # 3. Recursive Branching
        # Option L: Merge violator with left neighbor (remove cutoffs[idx])
        # Option R: Merge violator with right neighbor (remove cutoffs[idx+1])
        
        results = []
        
        # Try Path L
        cuts_l = cutoffs[:violation_idx] + cutoffs[violation_idx+1:]
        if len(cuts_l) >= 2:
            res_l = self.repair_recursive(cuts_l)
            if res_l: results.append(res_l)
            
        # Try Path R
        if violation_idx < len(cutoffs) - 2:
            cuts_r = cutoffs[:violation_idx+1] + cutoffs[violation_idx+2:]
            if len(cuts_r) >= 2:
                res_r = self.repair_recursive(cuts_r)
                if res_r: results.append(res_r)

        # 4. Selection: Pick the path with the maximum number of bins
        if not results: return None
        return max(results, key=lambda x: len(x))
    
    def repair_recursive(self, cutoffs):
            """
            Recursively resolves violations by branching into three merge strategies:
            - Path L: Merge violation bin (i) with left (i-1)
            - Path R: Merge violation bin (i) with right (i+1)
            - Path LL: Merge the two bins preceding the violation (i-2 and i-1)
            Returns the valid path with the maximum number of bins.
            """
            stats = []
            for i in range(len(cutoffs) - 1):
                m = self._get_bin_metrics(cutoffs[i], cutoffs[i+1], i == len(cutoffs)-2)
                # If a single bin itself violates the [a, b] proportion rule, this branch is invalid
                if m is None: return None 
                stats.append(m)
            
            # 1. Identify the first violation
            violation_idx = None
            for i in range(1, len(stats)):
                prev, curr = stats[i-1], stats[i]
                # Check Monotonicity and Adjacent Sum
                mono_fail = (curr['tr_br'] >= prev['tr_br'] - self.eps) or \
                            (curr['ts_br'] >= prev['ts_br'] - self.eps) or \
                            (curr['avg_ts_br'] >= prev['avg_ts_br'] - self.eps)
                adj_fail = np.any(prev['props'] + curr['props'] > self.C + self.eps)
                
                if mono_fail or adj_fail:
                    violation_idx = i
                    break
            
            # 2. Base Case: No more violations found
            if violation_idx is None:
                return cutoffs
    
            results = []
    
            # Strategy Path L: Merge violating bin with its left neighbor
            # We remove the boundary between bin i-1 and bin i
            cuts_l = cutoffs[:violation_idx] + cutoffs[violation_idx+1:]
            res_l = self.repair_recursive(cuts_l)
            if res_l: results.append(res_l)
                
            # Strategy Path R: Merge violating bin with its right neighbor
            # We remove the boundary between bin i and bin i+1
            if violation_idx < len(cutoffs) - 2:
                cuts_r = cutoffs[:violation_idx+1] + cutoffs[violation_idx+2:]
                res_r = self.repair_recursive(cuts_r)
                if res_r: results.append(res_r)
    
            # Strategy Path LL: Merge row i-2 and row i-1
            # This changes the bad rate of the bin that 'curr' is being compared against.
            # We need at least two bins before the violation point to do this.
            if violation_idx >= 2:
                cuts_ll = cutoffs[:violation_idx-1] + cutoffs[violation_idx:]
                res_ll = self.repair_recursive(cuts_ll)
                if res_ll: results.append(res_ll)
    
            # 3. Decision: Return the valid path that yields the highest bin count
            if not results:
                return None
            return max(results, key=lambda x: len(x))

    # ==========================================
    # AUDIT AND OUTPUT
    # ==========================================

    def final_report(self, cutoffs):
        print(f"\n{'#'*80}\nFINAL CONSTRAINT AUDIT\n{'#'*80}")
        if not cutoffs:
            print("No valid binning solution found.")
            return
            
        stats = self._get_all_bin_stats(cutoffs)
        violations = 0
        
        for i in range(len(stats)):
            m = stats[i]
            print(f"Group {i+1} [{cutoffs[i]:.2f} to {cutoffs[i+1]:.2f}]:")
            
            # Check Prop
            p_fail = np.any(m['props'] < self.a) or np.any(m['props'] > self.b)
            print(f"  - Proportions ({self.a:.0%} to {self.b:.0%}): {'PASS' if not p_fail else 'FAIL !!'}")
            if p_fail: violations += 1
            
            # Check Mono
            if i > 0:
                prev = stats[i-1]
                mono_fail = (m['tr_br'] >= prev['tr_br']) or (m['ts_br'] >= prev['ts_br']) or (m['avg_ts_br'] >= prev['avg_ts_br'])
                adj_fail = np.any(prev['props'] + m['props'] > self.C)
                print(f"  - Monotonicity Trend:       {'PASS' if not mono_fail else 'FAIL !!'}")
                print(f"  - Adjacent Sum (Max {self.C:.0%}): {'PASS' if not adj_fail else 'FAIL !!'}")
                if mono_fail or adj_fail: violations += 1
                
            print(f"  - Bad Rates: Train={m['tr_br']:.4f}, Test={m['ts_br']:.4f}, AvgTest={m['avg_ts_br']:.4f}")
            print("-" * 40)

        print(f"\nFINAL VERDICT: {violations} Violations Detected.")
        return violations

# ==========================================
# RUNNING THE EXAMPLE
# ==========================================

def gen_data(n):
    s = np.random.randint(300, 850, n)
    # Basic logic: Higher score = Lower target probability
    p = 1 / (1 + np.exp((s - 550) / 65))
    return pd.DataFrame({'score': s, 'target': np.random.binomial(1, p)})

# Setup
train = [gen_data(1500) for _ in range(10)]
test = [gen_data(800) for _ in range(6)]
framework = AdvancedBinningFramework(train, test, a=0.05, b=0.25)

# Execute Stages
initial_cuts = framework.solve_stage1()
final_cuts = framework.repair_recursive(initial_cuts)

# Output
framework.final_report(final_cuts)