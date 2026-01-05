import pandas as pd
import numpy as np

class AdvancedBinningFramework:
    def __init__(self, train_dfs, test_dfs, a=0.05, b=0.25, C=0.50):
        self.train_dfs = train_dfs
        self.test_dfs = test_dfs
        self.all_dfs = train_dfs + test_dfs
        self.n_train, self.n_test = len(train_dfs), len(test_dfs)
        
        # Constraints
        self.a, self.b, self.C = a, b, C
        self.eps = 1e-7 # Precision buffer for floating point comparisons
        
        # Pre-calculate total populations for aggregate proportion calculation
        self.total_train_pop = sum(len(df) for df in train_dfs)
        self.total_test_pop = sum(len(df) for df in test_dfs)

    # ==========================================
    # CORE METRIC CALCULATOR (UPDATED)
    # ==========================================

    def _get_bin_metrics(self, low, high, is_last=False):
        """Calculates exact metrics, including aggregate proportions."""
        individual_props, bads, counts = [], [], []
        
        for i, df in enumerate(self.all_dfs):
            # Boundary handling: [low, high) vs [low, high]
            mask = (df['pd'] >= low) & (df['pd'] <= high) if is_last else (df['pd'] >= low) & (df['pd'] < high)
            c, b = len(df[mask]), df[mask]['target'].sum()
            
            individual_props.append(c / len(df))
            counts.append(c)
            bads.append(b)
        
        individual_props = np.array(individual_props)
        counts = np.array(counts)
        bads = np.array(bads)
        
        # 1. Check Condition 1: Every individual dataset must be between a and b
        if np.any(individual_props < self.a - self.eps) or np.any(individual_props > self.b + self.eps):
            return None

        # 2. Calculate Aggregate Statistics
        train_count = counts[:self.n_train].sum()
        train_bads = bads[:self.n_train].sum()
        test_count = counts[self.n_train:].sum()
        test_bads = bads[self.n_train:].sum()
        
        # Calculate Bad Rates
        agg_tr_br = train_bads / train_count if train_count > 0 else 0
        agg_ts_br = test_bads / test_count if test_count > 0 else 0
        
        # Per-test-set bad rates for the average calculation
        test_set_brs = [bads[i]/counts[i] if counts[i] > 0 else 0 for i in range(self.n_train, len(self.all_dfs))]
        
        return {
            'props': individual_props,
            'agg_tr_prop': train_count / self.total_train_pop, # NEW: Aggregate Train Proportion
            'agg_ts_prop': test_count / self.total_test_pop,   # NEW: Aggregate Test Proportion
            'tr_br': agg_tr_br,
            'ts_br': agg_ts_br,
            'avg_ts_br': np.mean(test_set_brs)
        }

    # ==========================================
    # STAGE 1: GLOBAL OPTIMIZATION (DP)
    # ==========================================

    def solve_stage1(self, n_micro=100):
        print("Stage 1: Running Dynamic Programming Solver...")
        print("It may take a while, plz wait.")
        combined = pd.concat(self.all_dfs)[['pd']]
        _, self.bin_edges = pd.qcut(combined['pd'], q=n_micro, retbins=True, duplicates='drop')
        self.bin_edges = sorted(list(set(self.bin_edges)))
        n = len(self.bin_edges) - 1
        
        # 
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
                        
                        # Monotonicity checks with epsilon
                        if curr['tr_br'] <= prev['tr_br'] - self.eps: continue
                        if curr['ts_br'] <= prev['ts_br'] - self.eps: continue
                        if curr['avg_ts_br'] <= prev['avg_ts_br'] - self.eps: continue
                        # Adj Sum check
                        if np.any(prev['props'] + curr['props'] > self.C + self.eps): continue
                        
                        if chains[(pk, pi)][0] > best_len:
                            best_len, best_pred = chains[(pk, pi)][0], (pk, pi)
                    
                    if best_pred: chains[(i, j)] = (best_len + 1, best_pred)

        final = max([k for k in chains if k[1] == n], key=lambda x: chains[x][0], default=None)
        if not final: return []
        
        path = []
        while final: 
            path.append(final)
            final = chains[final][1]
        return [self.bin_edges[0]] + [self.bin_edges[k[1]] for k in reversed(path)]

    # ==========================================
    # STAGE 2: RECURSIVE REPAIR (BACKTRACKING)
    # ==========================================

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
                mono_fail = (curr['tr_br'] <= prev['tr_br'] - self.eps) or \
                            (curr['ts_br'] <= prev['ts_br'] - self.eps) or \
                            (curr['avg_ts_br'] <= prev['avg_ts_br'] - self.eps)
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
        print(f"\n{'='*100}\nFINAL BINNING AUDIT REPORT\n{'='*100}")
        if not cutoffs:
            print("ERROR: No valid binning path exists under these constraints.")
            return
            
        stats = []
        for i in range(len(cutoffs) - 1):
            stats.append(self._get_bin_metrics(cutoffs[i], cutoffs[i+1], i == len(cutoffs)-2))

        for i, m in enumerate(stats):
            print(f"BIN {i+1}: Range [{cutoffs[i]:.2f}, {cutoffs[i+1]:.2f}]")
            print(f"  - AGGREGATE PROPORTION:  Train: {m['agg_tr_prop']:.2%} | Test: {m['agg_ts_prop']:.2%}")
            print(f"  - BAD RATES:            Train: {m['tr_br']:.4f}  | Test: {m['ts_br']:.4f} | AvgTest: {m['avg_ts_br']:.4f}")
            
            if i > 0:
                p = stats[i-1]
                m_ok = (m['tr_br'] > p['tr_br']) and (m['ts_br'] > p['ts_br']) and (m['avg_ts_br'] > p['avg_ts_br'])
                print(f"  - MONOTONICITY CHECK:   {'PASS' if m_ok else 'FAIL !!'}")
            print("-" * 100)

    def summary(self, cutoffs):
        
        train_list = [f"Tr{i+1}" for i in range(self.n_train)]
        test_list = [f"Te{i+1}" for i in range(self.n_test)]
        combined_list = train_list + test_list
        
        stats = []
        for i in range(len(cutoffs) - 1):
            stats.append(self._get_bin_metrics(cutoffs[i], cutoffs[i+1], i ==len(cutoffs)-2))
            
        df = pd.DataFrame()
        groups = []
        
        for i in range(len(stats)):
            m = stats[i]
            groups.append(m)
            
            if i > 0:
                p_m = groups[i-1] # get previous stats
                m_tr = m['tr_br'] > p_m['tr_br']
                m_ts = m['ts_br'] > p_m['ts_br']
                m_avg = m['avg_ts_br'] > p_m['avg_ts_br']
                adj_ok = np.all(p_m['props'] + m['props'] <= self.C)
            else:
                m_tr, m_ts, m_avg, adj_ok = None, None, None, None
            
            data = m
            data["pd_range"] = f"[{cutoffs[i]:.2f}, {cutoffs[i+1]:.2f})"
            
            df_tmp = pd.DataFrame({
                'pd_range':[data['pd_range']],
                **{f'prop_{i}':[data['props'][i]] for i in range(len(data['props']))},
                'tr_br':[data['tr_br']],
                'ts_br':[data['ts_br']],
                'avg_ts_br':[data['avg_ts_br']],
                'agg_tr_prop':[data['agg_tr_prop']],
                'agg_ts_prop':[data['agg_ts_prop']],
                'm_tr':[m_tr],
                'm_ts':[m_ts],
                'm_avg':[m_avg],
                'adj_ok':[adj_ok],
                'pd_left':[cutoffs[i]],
                'pd_right':[cutoffs[i+1]],          
                })
            
            col_next = list(df_tmp.columns)
            col_next = col_next[(1+len(combined_list)):]
            col_name = ['score_range'] + combined_list + col_next
            df_tmp.columns = col_name
            df = pd.concat([df, df_tmp], axis = 0)
            
        return df

# ==========================================
# Validate_the_cutoff results
# ==========================================

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
        df['bin'] = pd.cut(df['pd'], bins=cutoffs, include_lowest=True, labels=bin_labels)
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
        df['bin'] = pd.cut(df['pd'], bins=cutoffs, include_lowest=True, labels=bin_labels)
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
    train_merged['bin'] = pd.cut(train_merged['pd'], bins=cutoffs, include_lowest=True, labels=bin_labels)
    # Group statistics: total customers & bad sample count per bin
    merge_train_stats = train_merged.groupby('bin', observed=False).agg(
        total_customers=('pd', 'count'),
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
    val_merged['bin'] = pd.cut(val_merged['pd'], bins=cutoffs, include_lowest=True, labels=bin_labels)
    # Group statistics: total customers & bad sample count per bin
    merge_val_stats = val_merged.groupby('bin', observed=False).agg(
        total_customers=('pd', 'count'),
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
#%%
# ==========================================
# EXECUTION
# ==========================================

def gen_data(n, seed):
    np.random.seed(seed)
    s = np.random.randint(300, 850, n)
    # Target prob decreases as score increases
    p = 1 / (1 + np.exp((s - 580) / 50))
    return pd.DataFrame({'pd': p, 'target': np.random.binomial(1, p)})

# Generate 16 datasets
train_sets = [gen_data(2000, i) for i in range(10)]
test_sets = [gen_data(1000, i+10) for i in range(6)]

# Run Framework
model = AdvancedBinningFramework(train_sets, test_sets, a=0.05, b=0.25)
stage1_cuts = model.solve_stage1()

final_cuts = model.repair_recursive(stage1_cuts)
model.final_report(final_cuts)
df = model.summary(final_cuts)


#%%




