# Import pandas library for DataFrame creation, grouping, statistics and other data processing operations
import pandas as pd
# Import numpy library for numerical calculation and array operations (e.g., sum statistics, conditional judgment)
import numpy as np

# ===================== Global utility function: Generate simulated datasets =====================
def gen(n):
    """
    Generate simulated dataset for binning test
    Parameters:
        n (int): Number of samples in the dataset
    Returns:
        pd.DataFrame: DataFrame containing 'score' (random integer from 300 to 850) and 'target' (good/bad label)
    """
    # Generate random integer scores ranging from 300 to 850 for customers
    s = np.random.randint(300, 850, n)
    # Generate target labels: bad sample probability is negatively correlated with scores (higher score = lower bad rate)
    target = np.random.binomial(1, 1/(1+np.exp((s-580)/50)))
    return pd.DataFrame({'score': s, 'target': target})

# ===================== Function 1: Initialize global configuration dictionary (replace class __init__) =====================
def init_config(train_dfs, test_dfs, a=0.05, b=0.25, C=0.50):
    """
    Initialize all configurations, parameters and dataset information, stored in a unified dictionary
    (replaces the instance attributes of the original class)
    Parameters:
        train_dfs (list[pd.DataFrame]): List of training datasets
        test_dfs (list[pd.DataFrame]): List of test datasets
        a (float): Minimum proportion of customers in a single group
        b (float): Maximum proportion of customers in a single group
        C (float): Maximum sum of proportions of two adjacent groups
    Returns:
        dict: Global configuration dictionary with unified storage of all parameters, datasets and statistics
    """
    config = {}
    # Store dataset related information
    config['train_dfs'] = train_dfs
    config['test_dfs'] = test_dfs
    config['all_dfs'] = train_dfs + test_dfs
    config['n_train'] = len(train_dfs)
    config['n_test'] = len(test_dfs)
    # Store business constraint parameters
    config['a'] = a
    config['b'] = b
    config['C'] = C
    # Initialize placeholders for micro-binning related data (populated by subsequent functions)
    config['bin_edges'] = None
    config['n_micro_bins'] = None
    config['micro_stats'] = None
    return config

# ===================== Function 2: Calculate micro-binning statistics (replace _calculate_micro_stats) =====================
def calculate_micro_stats(config, n_bins=60):
    """
    Calculate statistical information of fine-grained micro-binning, lay the foundation for dynamic programming
    (completely consistent with the original private method logic)
    Parameters:
        config (dict): Global configuration dictionary (will be modified to store bin edges and micro statistics)
        n_bins (int): Number of micro-bins, default 60 (finer bins mean more flexible grouping)
    """
    # Merge score and target columns of all datasets to generate unified bin edges
    combined = pd.concat(config['all_dfs'])[['score', 'target']]
    # Quantile binning, generate equal-frequency bin edges and remove duplicates to avoid overlap
    _, bin_edges = pd.qcut(combined['score'], q=n_bins, retbins=True, duplicates='drop')
    config['bin_edges'] = sorted(list(set(bin_edges)))
    config['n_micro_bins'] = len(config['bin_edges']) - 1
    
    # Initialize micro-binning statistics array: [num_micro_bins, num_datasets, 2] 
    # Dimension 2 = [customer count, bad sample count]
    config['micro_stats'] = np.zeros((config['n_micro_bins'], len(config['all_dfs']), 2))
    
    # Iterate over each dataset and count customers & bad samples for each micro-bin
    for idx, df in enumerate(config['all_dfs']):
        # Bin the scores with pre-generated edges and return bin index for each sample
        cuts = pd.cut(df['score'], bins=config['bin_edges'], include_lowest=True, labels=False)
        # Group statistics: customer count (count), bad sample count (sum)
        stats = df.groupby(cuts, observed=False)['target'].agg(['count', 'sum'])
        # Populate statistical results into the array
        for b_idx in stats.index:
            if 0 <= b_idx < config['n_micro_bins']:
                config['micro_stats'][b_idx, idx, 0] = stats.loc[b_idx, 'count']  # Customer count
                config['micro_stats'][b_idx, idx, 1] = stats.loc[b_idx, 'sum']   # Bad sample count

# ===================== Function 3: Calculate aggregate metrics for specified bin range (replace _get_metrics) =====================
def get_metrics(config, start, end):
    """
    Calculate aggregate metrics for the micro-binning interval [start, end) and verify Constraint 1 (single group proportion)
    Parameters:
        config (dict): Global configuration dictionary
        start (int): Start index of micro-binning (closed interval)
        end (int): End index of micro-binning (open interval)
    Returns:
        dict/None: Return metric dictionary if constraints are satisfied, otherwise return None
    """
    # Aggregate statistical data of all micro-bins in the interval
    g_stats = config['micro_stats'][start:end].sum(axis=0)
    counts, bads = g_stats[:, 0], g_stats[:, 1]
    # Calculate total customers and proportion of current interval for each dataset
    totals = config['micro_stats'].sum(axis=0)[:, 0]
    props = counts / totals
    
    # Verify Constraint 1: Proportion of all datasets must be within [a, b]
    if np.any(props < config['a']) or np.any(props > config['b']):
        return None
    
    # Calculate bad rate metrics for total training set, total test set and average test set
    tr_c, tr_b = counts[:config['n_train']].sum(), bads[:config['n_train']].sum()
    ts_c, ts_b = counts[config['n_train']:].sum(), bads[config['n_train']:].sum()
    test_brs = [bads[i]/counts[i] if counts[i] > 0 else 0 
                for i in range(config['n_train'], len(config['all_dfs']))]
    
    return {
        'props': props,
        'tr_br': tr_b/tr_c if tr_c > 0 else 0,
        'ts_br': ts_b/ts_c if ts_c > 0 else 0,
        'avg_ts_br': np.mean(test_brs)
    }

# ===================== Function 4: Core solving function (DP for optimal binning, replace solve) =====================
def binning_solve(config):
    """
    Core function: Use dynamic programming to find the optimal score bin edges, maximize the number of groups
    while satisfying all 4 business constraints
    Parameters:
        config (dict): Global configuration dictionary (micro-binning statistics completed)
    Returns:
        list[float]: Optimal score bin edge list (ascending order), return empty list if no valid solution
    """
    # Execute micro-binning statistics first (80 fine-grained bins by default for higher flexibility)
    calculate_micro_stats(config, n_bins=80)
    
    # Initialize DP bin chain dictionary: key=(start, end), value=(chain length, predecessor key)
    chains = {}
    n_micro = config['n_micro_bins']
    
    # Double loop to traverse all possible micro-binning intervals
    for j in range(1, n_micro + 1):
        for i in range(j):
            # Calculate metrics for current interval and verify Constraint 1
            curr_metrics = get_metrics(config, i, j)
            if curr_metrics is None:
                continue
            
            # Case 1: Current interval is the first group (starts from 0)
            if i == 0:
                chains[(i, j)] = (1, None)
            # Case 2: Match legal predecessor groups and verify Constraints 2/3/4
            else:
                best_prev_len = -1
                best_prev_key = None
                # Traverse all predecessor groups ending at i
                for (pk, pi) in [k for k in chains if k[1] == i]:
                    prev_metrics = get_metrics(config, pk, pi)
                    # Verify Constraint 2 (sum of adjacent proportions), 3 (total bad rate monotonicity), 4 (avg test bad rate monotonicity)
                    if np.any(prev_metrics['props'] + curr_metrics['props'] > config['C']):
                        continue
                    if curr_metrics['tr_br'] >= prev_metrics['tr_br'] or curr_metrics['ts_br'] >= prev_metrics['ts_br']:
                        continue
                    if curr_metrics['avg_ts_br'] >= prev_metrics['avg_ts_br']:
                        continue
                    # Update the optimal predecessor
                    if chains[(pk, pi)][0] > best_prev_len:
                        best_prev_len = chains[(pk, pi)][0]
                        best_prev_key = (pk, pi)
                # Record legal bin chains
                if best_prev_key:
                    chains[(i, j)] = (best_prev_len + 1, best_prev_key)
    
    # Find the optimal bin chain (the longest chain ending at the last micro-bin)
    final_key = max(
        [k for k in chains if k[1] == n_micro],
        key=lambda x: chains[x][0],
        default=None
    )
    if not final_key:
        return []
    
    # Backtrack predecessors to generate the final bin edges
    path = []
    curr_key = final_key
    while curr_key:
        path.append(curr_key)
        curr_key = chains[curr_key][1]
    
    # Generate ordered score bin edges
    return [config['bin_edges'][0]] + [config['bin_edges'][k[1]] for k in reversed(path)]

# ===================== Function 5: Audit auxiliary - calculate raw metrics (replace _get_metrics_raw) =====================
def get_metrics_raw(config, low, high, last):
    """
    Calculate raw metrics based on the specified score interval (used for audit verification)
    Parameters:
        config (dict): Global configuration dictionary
        low (float): Lower bound of the score interval
        high (float): Upper bound of the score interval
        last (bool): Whether it is the last group (control whether to include the upper bound)
    Returns:
        dict: Metric dictionary containing proportions and bad rates
    """
    props, test_brs = [], []
    tr_b, tr_c, ts_b, ts_c = 0, 0, 0, 0
    
    for i, df in enumerate(config['all_dfs']):
        # Score interval mask: the last group includes the upper bound, others do not
        if last:
            mask = (df['score'] >= low) & (df['score'] <= high)
        else:
            mask = (df['score'] >= low) & (df['score'] < high)
        # Count customers and bad samples of the current dataset in the interval
        c, b = len(df[mask]), df[mask]['target'].sum()
        p = c / len(df)
        props.append(p)
        # Aggregate by training/test set classification
        if i < config['n_train']:
            tr_c += c
            tr_b += b
        else:
            ts_c += c
            ts_b += b
            test_brs.append(b/c if c > 0 else 0)
    
    return {
        'props': np.array(props),
        'tr_br': tr_b/tr_c if tr_c > 0 else 0,
        'ts_br': ts_b/ts_c if ts_c > 0 else 0,
        'avg_ts_br': np.mean(test_brs) if test_brs else 0
    }

# ===================== Function 6: Audit verification function (replace audit) =====================
def binning_audit(config, cutoffs):
    """
    Audit and verify the binning results, validate all 4 constraints, output detailed audit report and violation statistics
    Parameters:
        config (dict): Global configuration dictionary
        cutoffs (list[float]): Optimal bin edge list
    """
    # Print audit report title
    print(f"\n{'='*95}\nAUDIT REPORT: a={config['a']:.1%}, b={config['b']:.1%}, C={config['C']:.1%}\n{'='*95}")
    # Generate group intervals
    groups = [(cutoffs[i], cutoffs[i+1]) for i in range(len(cutoffs)-1)]
    history = []
    violations = 0
    
    # Traverse each group, verify constraints and print results
    for idx, (low, high) in enumerate(groups):
        print(f"\nGROUP {idx+1}: [{low:.2f}, {high:.2f}]")
        # Calculate metrics for current group
        metrics = get_metrics_raw(config, low, high, idx == len(groups)-1)
        
        # Print proportion of each dataset and mark Constraint 1 violations
        for s_idx, p in enumerate(metrics['props']):
            lbl = f"Tr {s_idx+1:02}" if s_idx < config['n_train'] else f"Ts {s_idx-config['n_train']+1:02}"
            err = "!!" if p < config['a'] or p > config['b'] else ""
            if err:
                violations += 1
            print(f"  {lbl} | Prop: {p:6.2%} {err}", end=" | ")
            if (s_idx + 1) % 4 == 0:
                print("")
        
        # Print aggregate bad rate metrics
        print(f"\n  Agg Train BR: {metrics['tr_br']:.4f} | Agg Test BR: {metrics['ts_br']:.4f} | Avg Test BR: {metrics['avg_ts_br']:.4f}")
        
        # Verify Constraints 2, 3, 4 (compared with the previous group)
        if history:
            prev = history[-1]
            m_tr = metrics['tr_br'] < prev['tr_br']
            m_ts = metrics['ts_br'] < prev['ts_br']
            m_avg = metrics['avg_ts_br'] < prev['avg_ts_br']
            adj_ok = np.all(metrics['props'] + prev['props'] <= config['C'])
            
            print(f"  Checks: Mono Train: {m_tr} | Mono Test: {m_ts} | Mono Avg: {m_avg} | Adj Sum OK: {adj_ok}")
            if not all([m_tr, m_ts, m_avg, adj_ok]):
                violations += 1
        
        history.append(metrics)
    
    # Print audit summary
    print(f"\n{'='*95}\nAUDIT COMPLETE. TOTAL VIOLATIONS: {violations}\n{'='*95}")

# ===================== Main execution entry (pure function sequential call, no Class) =====================
if __name__ == "__main__":
    # 1. Fix random seed to ensure reproducible results
    np.random.seed(42)
    
    # 2. Generate simulated data: 10 training sets (1500 samples each) and 6 test sets (800 samples each)
    train_datasets = [gen(1500) for _ in range(10)]
    test_datasets = [gen(800) for _ in range(6)]
    
    # 3. Initialize global configuration (constraint parameters can be modified as needed)
    bin_config = init_config(
        train_dfs=train_datasets,
        test_dfs=test_datasets,
        a=0.05,    # Minimum proportion of single group: 5%
        b=0.25,    # Maximum proportion of single group: 25%
        C=0.50     # Maximum sum of adjacent group proportions: 50%
    )
    
    # 4. Core solving: Get optimal score bin edges
    optimal_cutoffs = binning_solve(bin_config)
    
    # 5. Audit verification: Validate constraints and output report (execute only if there is a valid solution)
    if optimal_cutoffs:
        print("✅ Solved successfully! The optimal score bin edges are:")
        print([round(x, 2) for x in optimal_cutoffs])
        binning_audit(bin_config, optimal_cutoffs)
    else:
        print("❌ Solving failed! No binning scheme satisfies all constraints, please relax the parameters and try again.")