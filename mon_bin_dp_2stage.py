import pandas as pd
import numpy as np

class RobustBinningOptimizer:
    def __init__(self, train_dfs, test_dfs, a=0.05, b=0.25, C=0.50):
        self.train_dfs = train_dfs
        self.test_dfs = test_dfs
        self.all_dfs = train_dfs + test_dfs
        self.n_train, self.n_test = len(train_dfs), len(test_dfs)
        
        # Constraints
        self.a, self.b, self.C = a, b, C
        
        # Epsilon Buffer: Ensures solver is stricter than the auditor
        # This prevents "marginal" passes from failing the final check.
        self.eps = 1e-7 

    # ==========================================
    # CORE UTILITIES
    # ==========================================

    def _get_exact_metrics(self, low, high, is_last=False):
        """Calculates exact metrics for a score range across all 16 datasets."""
        props, bads, counts = [], [], []
        
        for i, df in enumerate(self.all_dfs):
            # Boundary logic: [low, high) or [low, high] for last bin
            if is_last:
                mask = (df['score'] >= low) & (df['score'] <= high)
            else:
                mask = (df['score'] >= low) & (df['score'] < high)
            
            c = len(df[mask])
            b = df[mask]['target'].sum()
            props.append(c / len(df))
            bads.append(b)
            counts.append(c)
        
        props = np.array(props)
        counts = np.array(counts)
        bads = np.array(bads)

        # Basic Condition 1 Check (Solver strictness)
        if np.any(props < self.a - self.eps) or np.any(props > self.b + self.eps):
            return None

        # Calculate Rates
        tr_c, tr_b = counts[:self.n_train].sum(), bads[:self.n_train].sum()
        ts_c, ts_b = counts[self.n_train:].sum(), bads[self.n_train:].sum()
        
        test_brs = [bads[i]/counts[i] if counts[i] > 0 else 0 for i in range(self.n_train, self.n_train + self.n_test)]
        
        return {
            'props': props,
            'tr_br': tr_b/tr_c if tr_c > 0 else 0,
            'ts_br': ts_b/ts_c if ts_c > 0 else 0,
            'avg_ts_br': np.mean(test_brs)
        }

    def _calculate_micro_stats(self, n_bins=100):
        """Discretizes scores into micro-bins to create 'building blocks' for the solver."""
        combined = pd.concat(self.all_dfs)[['score']]
        _, self.bin_edges = pd.qcut(combined['score'], q=n_bins, retbins=True, duplicates='drop')
        self.bin_edges = sorted(list(set(self.bin_edges)))
        self.n_micro = len(self.bin_edges) - 1

    # ==========================================
    # STAGE 1: GLOBAL OPTIMIZATION (DP)
    # ==========================================

    def solve_stage1(self):
        print("Starting Stage 1: Global Optimization...")
        self._calculate_micro_stats(n_bins=100)
        
        # dp[end_bin] = (max_groups, prev_key)
        chains = {}
        
        for j in range(1, self.n_micro + 1):
            for i in range(j):
                curr = self._get_exact_metrics(self.bin_edges[i], self.bin_edges[j], j == self.n_micro)
                if curr is None: continue
                
                if i == 0:
                    chains[(i, j)] = (1, None)
                else:
                    best_prev_len = -1
                    best_prev_key = None
                    for (pk, pi) in [k for k in chains if k[1] == i]:
                        prev = self._get_exact_metrics(self.bin_edges[pk], self.bin_edges[pi], False)
                        
                        # Monotonicity with Epsilon Buffer
                        if curr['tr_br'] >= (prev['tr_br'] - self.eps): continue
                        if curr['ts_br'] >= (prev['ts_br'] - self.eps): continue
                        if curr['avg_ts_br'] >= (prev['avg_ts_br'] - self.eps): continue
                        
                        # Adjacency Sum
                        if np.any(prev['props'] + curr['props'] > self.C + self.eps): continue
                        
                        if chains[(pk, pi)][0] > best_prev_len:
                            best_prev_len = chains[(pk, pi)][0]
                            best_prev_key = (pk, pi)
                    
                    if best_prev_key:
                        chains[(i, j)] = (best_prev_len + 1, best_prev_key)

        final_key = max([k for k in chains if k[1] == self.n_micro], key=lambda x: chains[x][0], default=None)
        if not final_key: return []
        
        path = []
        curr = final_key
        while curr:
            path.append(curr)
            curr = chains[curr][1]
        return [self.bin_edges[0]] + [self.bin_edges[k[1]] for k in reversed(path)]

    # ==========================================
    # STAGE 2: POST-OPTIMIZATION REPAIR
    # ==========================================

    def repair_stage2(self, cutoffs):
        """Identifies violations and merges bins greedily."""
        print("Starting Stage 2: Post-Optimization Repair...")
        current_cuts = list(cutoffs)
        
        for attempt in range(15):
            viol_idx, error = self.find_first_violation(current_cuts)
            if viol_idx is None:
                print(f"Repair complete. All constraints satisfied in {attempt} merges.")
                return current_cuts
            
            print(f"  > Attempt {attempt+1}: Fixing {error} at group {viol_idx+1}")
            
            # Strategy: Merge the violating bin with the best neighbor
            # If it's the first bin, merge with next. If last, merge with previous.
            if viol_idx == 0:
                current_cuts.pop(1)
            elif viol_idx == len(current_cuts) - 2:
                current_cuts.pop(-2)
            else:
                # Try previous 2 or next 2 logic: we remove the boundary that 
                # provides a better monotonicity flow.
                current_cuts.pop(viol_idx) # Default greedy merge with left neighbor
                
        return current_cuts

    def find_first_violation(self, cuts):
        groups = []
        for i in range(len(cuts)-1):
            m = self._get_exact_metrics(cuts[i], cuts[i+1], i == len(cuts)-2)
            if m is None: return i, "Prop Bounds"
            groups.append(m)
            
        for i in range(len(groups)):
            if i > 0:
                p, c = groups[i-1], groups[i]
                if c['tr_br'] >= p['tr_br'] or c['ts_br'] >= p['ts_br'] or c['avg_ts_br'] >= p['avg_ts_br']:
                    return i, "Monotonicity"
                if np.any(p['props'] + c['props'] > self.C):
                    return i, "Adj Sum"
        return None, None

    # ==========================================
    # AUDIT / REPORTING
    # ==========================================

    def run_audit(self, cuts):
        print(f"\n{'='*100}\nFINAL AUDIT REPORT\n{'='*100}")
        total_violations = 0
        groups = []
        for i in range(len(cuts)-1):
            m = self._get_exact_metrics(cuts[i], cuts[i+1], i == len(cuts)-2)
            groups.append(m)
            print(f"GROUP {i+1} [{cuts[i]:.2f} - {cuts[i+1]:.2f}]")
            
            # Print specific dataset check
            for s_idx, p in enumerate(m['props']):
                status = "!!" if p < self.a or p > self.b else "OK"
                if status == "!!": total_violations += 1
                if s_idx % 8 == 0 and s_idx > 0: print("")
                print(f" S{s_idx+1:02}:{p:5.1%} ({status})", end=" | ")
            
            print(f"\n Agg Tr BR: {m['tr_br']:.4f} | Agg Ts BR: {m['ts_br']:.4f} | Avg Ts BR: {m['avg_ts_br']:.4f}")
            
            if i > 0:
                p_m = groups[i-1]
                m_ok = (m['tr_br'] < p_m['tr_br']) and (m['ts_br'] < p_m['ts_br']) and (m['avg_ts_br'] < p_m['avg_ts_br'])
                adj_ok = np.all(p_m['props'] + m['props'] <= self.C)
                print(f" Monotonicity: {'PASS' if m_ok else 'FAIL !!'} | Adj Sum: {'PASS' if adj_ok else 'FAIL !!'}")
                if not m_ok or not adj_ok: total_violations += 1
            print("-" * 100)
            
        print(f"TOTAL REMAINING VIOLATIONS: {total_violations}")
        return total_violations

# ==========================================
# EXECUTION
# ==========================================

def get_dummy_data(n):
    np.random.seed(np.random.randint(0, 1000))
    s = np.random.randint(300, 850, n)
    # Ensure a basic trend for the solver to work with
    p = 1 / (1 + np.exp((s - 550) / 60))
    return pd.DataFrame({'score': s, 'target': np.random.binomial(1, p)})

# 1. Initialize
train = [get_dummy_data(1200) for _ in range(10)]
test = [get_dummy_data(700) for _ in range(6)]
opt = RobustBinningOptimizer(train, test, a=0.05, b=0.25, C=0.50)

# 2. Stage 1: Global Best
cuts_s1 = opt.solve_stage1()

# 3. Stage 2: Repair if needed
final_cuts = opt.repair_stage2(cuts_s1)

# 4. Detailed Results
opt.run_audit(final_cuts)