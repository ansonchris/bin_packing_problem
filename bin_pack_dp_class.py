# Import pandas library for DataFrame creation, grouping, statistics, and other data processing operations
import pandas as pd

# Import numpy library for numerical calculations and array operations (e.g., statistical summation, conditional judgment)
import numpy as np


class BinningOptimizer:
    """
    Customer Score Binning Optimizer Class
    Function: Calculate the optimal score binning boundaries to maximize the number of groups 
              while satisfying 4 business constraints based on training and test set data
    Core Logic: Find the longest valid bin chain through micro-binning statistics + Dynamic Programming (DP)
    """

    def __init__(self, train_dfs, test_dfs, a=0.05, b=0.25, C=0.50):
        """
        Class initialization method (constructor) to set the initial state and parameters of the instance
        
        Parameters:
            train_dfs (list[pd.DataFrame]): List of training datasets, each element is a DataFrame containing "score" (customer score) and "target" (good/bad label)
            test_dfs (list[pd.DataFrame]): List of test datasets with the same structure as training datasets
            a (float): Minimum proportion of customers in a single group (default 0.05, i.e., 5%)
            b (float): Maximum proportion of customers in a single group (default 0.25, i.e., 25%)
            C (float): Maximum sum of customer proportions of two adjacent groups (default 0.50, i.e., 50%)
        """
        # Save the list of training datasets to instance attribute
        self.train_dfs = train_dfs
        # Save the list of test datasets to instance attribute
        self.test_dfs = test_dfs
        # Merge training and test datasets to get the list of all datasets
        self.all_dfs = train_dfs + test_dfs
        # Record the number of training sets and test sets
        self.n_train, self.n_test = len(train_dfs), len(test_dfs)
        # Save business constraint parameters: proportion bounds, maximum sum of adjacent proportions
        self.a, self.b, self.C = a, b, C


    def _calculate_micro_stats(self, n_bins=60):
        """
        Private method: Calculate statistical information of "micro-binning" (first split scores into fine-grained bins to lay the foundation for subsequent dynamic programming)
        
        Parameters:
            n_bins (int): Number of micro-bins (default 60; finer bins mean more flexible subsequent grouping)
        """
        # 1. Merge all datasets and retain only "score" (customer score) and "target" (good/bad label) columns for unified bin edge generation
        combined = pd.concat(self.all_dfs)[['score', 'target']]
        
        # 2. Use quantile binning (pd.qcut) to bin the merged scores, return bin labels and bin edges (retbins=True)
        # duplicates='drop': Remove duplicate edges (avoid bin range overlap)
        _, self.bin_edges = pd.qcut(combined['score'], q=n_bins, retbins=True, duplicates='drop')
        
        # 3. Deduplicate and sort bin edges to ensure uniqueness and order (avoid duplicate bins)
        self.bin_edges = sorted(list(set(self.bin_edges)))
        
        # 4. Calculate the number of micro-bins: number of edges - 1 (e.g., edges [0,10,20] correspond to 2 bins)
        self.n_micro_bins = len(self.bin_edges) - 1
        
        # 5. Initialize micro-binning statistics array: dimension=(number of micro-bins, number of datasets, 2)
        # Third dimension index 0: number of customers in the corresponding dataset for this micro-bin; index 1: number of bad samples in the corresponding dataset for this micro-bin
        self.micro_stats = np.zeros((self.n_micro_bins, len(self.all_dfs), 2))
        
        # 6. Iterate over each dataset and calculate the number of customers and bad samples for each micro-bin
        for idx, df in enumerate(self.all_dfs):
            # 6.1 Use the previously obtained bin edges to bin the scores of the current dataset, return bin indices (labels=False)
            # include_lowest=True: Include the bin containing the minimum value (avoid unclassified scores equal to the minimum edge)
            cuts = pd.cut(df['score'], bins=self.bin_edges, include_lowest=True, labels=False)
            
            # 6.2 Group by bin indices and calculate the number of customers (count) and bad samples (sum, target=1 for bad samples) for the "target" column
            stats = df.groupby(cuts, observed=False)['target'].agg(['count', 'sum'])
            
            # 6.3 Fill the statistical results into the micro_stats array (ensure bin indices are within valid range)
            for b_idx in stats.index:
                if 0 <= b_idx < self.n_micro_bins:
                    self.micro_stats[b_idx, idx, 0] = stats.loc[b_idx, 'count']  # Number of customers
                    self.micro_stats[b_idx, idx, 1] = stats.loc[b_idx, 'sum']   # Number of bad samples


    def _get_metrics(self, start, end):
        """
        Private method: Calculate aggregate metrics from micro-bin start to end, and check Constraint 1 (single group proportion)
        
        Parameters:
            start (int): Start index of micro-bins (closed interval)
            end (int): End index of micro-bins (open interval, i.e., actually taking up to end-1)
        
        Returns:
            dict/None: Return a dictionary containing metrics if Constraint 1 is satisfied; otherwise return None
        """
        # 1. Aggregate micro-binning statistics within the interval [start, end): sum along the micro-bin dimension (axis=0)
        # Shape of g_stats: (number of datasets, 2), each row represents the total number of customers and bad samples of a dataset
        g_stats = self.micro_stats[start:end].sum(axis=0)
        # Extract the number of customers (index 0) and bad samples (index 1) for each dataset
        counts, bads = g_stats[:, 0], g_stats[:, 1]
        
        # 2. Calculate the total number of customers for each dataset (sum of customers in all micro-bins)
        totals = self.micro_stats.sum(axis=0)[:, 0]
        
        # 3. Calculate the customer proportion of the current interval in each dataset (number of customers in current interval / total number of customers)
        props = counts / totals
        
        # 4. Check Constraint 1: The proportion of a single group in all datasets must be between [a, b]
        # Return None if any dataset does not satisfy (current interval is invalid)
        if np.any(props < self.a) or np.any(props > self.b):
            return None
        
        # 5. Calculate aggregate metrics (total training set, total test set, test set average)
        # 5.1 Total training set: sum of customers and bad samples from all training sets
        tr_c, tr_b = counts[:self.n_train].sum(), bads[:self.n_train].sum()
        # 5.2 Total test set: sum of customers and bad samples from all test sets
        ts_c, ts_b = counts[self.n_train:].sum(), bads[self.n_train:].sum()
        # 5.3 Bad sample rate of each test set (avoid division by zero, set bad rate to 0 when number of customers is 0)
        test_brs = [bads[i]/counts[i] if counts[i] > 0 else 0 for i in range(self.n_train, len(self.all_dfs))]
        
        # 6. Return a valid metric dictionary
        return {
            'props': props,          # Customer proportion of each dataset (array)
            'tr_br': tr_b/tr_c if tr_c > 0 else 0,  # Bad sample rate of total training set
            'ts_br': ts_b/ts_c if ts_c > 0 else 0,  # Bad sample rate of total test set
            'avg_ts_br': np.mean(test_brs)          # Average bad sample rate of test sets
        }


    def solve(self):
        """
        Core method: Use Dynamic Programming (DP) to find the optimal bin chain and return the final score binning boundaries
        
        Returns:
            list[float]: List of binning boundaries (ascending order), return empty list if no valid binning exists
        """
        # 1. First calculate micro-binning statistics (n_bins=80, fine-grained bins improve grouping flexibility)
        self._calculate_micro_stats(n_bins=80)
        
        # 2. Initialize the DP bin chain dictionary: key=(start index of micro-bin, end index of micro-bin)
        # value=(current bin chain length, key of the previous bin chain) for backtracking the optimal path
        chains = {} 
        
        # 3. Double loop to traverse all possible micro-binning intervals (i as start, j as end, j > i)
        for j in range(1, self.n_micro_bins + 1):  # j: end index (open interval), from 1 to number of micro-bins
            for i in range(j):  # i: start index (closed interval), from 0 to j-1
                # 3.1 Calculate metrics for the current interval [i, j) and check Constraint 1
                curr = self._get_metrics(i, j)
                if curr is None:  # Skip current interval if Constraint 1 is not satisfied
                    continue
                
                # 3.2 Case 1: Current interval is the first group in the bin chain (i=0, starting from the first micro-bin)
                if i == 0:
                    # Bin chain length is 1, no previous group (None for the previous group)
                    chains[(i, j)] = (1, None)
                
                # 3.3 Case 2: Current interval is not the first group, need to match valid previous groups
                else:
                    # Initialize the length of the longest previous chain and its corresponding key
                    best_prev_len = -1
                    best_prev_key = None
                    
                    # Traverse all previous bin chains ending at i (end index of previous group equals start index i of current group)
                    for (pk, pi) in [k for k in chains if k[1] == i]:
                        # Calculate metrics of the previous group
                        prev = self._get_metrics(pk, pi)
                        
                        # Check Constraints 2, 3, 4:
                        # Constraint 2: Sum of proportions of previous group and current group ≤ C
                        # Constraint 3: Bad rate of total training set/total test set of current group < that of previous group (monotonically decreasing)
                        # Constraint 4: Average bad rate of test sets of current group < that of previous group (monotonically decreasing)
                        if np.any(prev['props'] + curr['props'] > self.C):
                            continue  # Skip if Constraint 2 is violated
                        if curr['tr_br'] >= prev['tr_br'] or curr['ts_br'] >= prev['ts_br']:
                            continue  # Skip if Constraint 3 is violated
                        if curr['avg_ts_br'] >= prev['avg_ts_br']:
                            continue  # Skip if Constraint 4 is violated
                        
                        # Update the longest previous chain (select the previous group with the maximum length)
                        if chains[(pk, pi)][0] > best_prev_len:
                            best_prev_len = chains[(pk, pi)][0]
                            best_prev_key = (pk, pi)
                    
                    # Update current bin chain if a valid previous group is found (length = previous length + 1, record previous key)
                    if best_prev_key:
                        chains[(i, j)] = (best_prev_len + 1, best_prev_key)

        # 4. Find the optimal bin chain: the longest chain ending at the last micro-bin (j=self.n_micro_bins)
        # Filter all keys ending at the last micro-bin, sort by chain length, and select the longest one
        final_key = max(
            [k for k in chains if k[1] == self.n_micro_bins], 
            key=lambda x: chains[x][0],  # Sort by chain length
            default=None  # Return None if no valid chain exists
        )
        
        # 5. Return empty list if no valid bin chain exists
        if not final_key:
            return []
        
        # 6. Backtrack the optimal bin chain to generate the bin boundary path
        path = []  # Store (start, end) indices of the bin chain
        curr_key = final_key
        while curr_key:
            path.append(curr_key)
            # Backtrack along the previous key until there is no previous group (curr_key becomes None)
            curr_key = chains[curr_key][1]
        
        # 7. Generate final bin boundaries: start with the first boundary, concatenate the end boundaries obtained from backtracking (need to reverse the path order)
        # Example: After reversing, path is [(0,20), (20,50), (50,80)], and boundaries are [bin_edges[0], bin_edges[20], bin_edges[50], bin_edges[80]]
        return [self.bin_edges[0]] + [self.bin_edges[k[1]] for k in reversed(path)]


    def audit(self, cutoffs):
        """
        Audit method: Verify all constraint conditions based on binning boundaries, generate an audit report, and count the number of violations
        
        Parameters:
            cutoffs (list[float]): List of binning boundaries (ascending order)
        """
        # 1. Print the audit report title (including constraint parameters)
        print(f"\n{'='*95}\nAUDIT REPORT: a={self.a:.1%}, b={self.b:.1%}, C={self.C:.1%}\n{'='*95}")
        
        # 2. Generate score intervals for each group (previous boundary as lower bound, next as upper bound)
        groups = [(cutoffs[i], cutoffs[i+1]) for i in range(len(cutoffs)-1)]
        
        # 3. Initialize historical metrics (store metrics of the previous group for checking Constraints 2, 3, 4) and violation count
        history = []
        violations = 0
        
        # 4. Traverse each group, verify constraints, and print results
        for idx, (low, high) in enumerate(groups):
            # Print current group number and score interval
            print(f"\nGROUP {idx+1}: [{low:.2f}, {high:.2f}]")
            
            # 4.1 Calculate raw metrics of the current group (distinguish whether it is the last group to handle boundary inclusion)
            metrics = self._get_metrics_raw(low, high, idx == len(groups)-1)
            
            # 4.2 Print the customer proportion of each dataset and mark violations (proportion not in [a,b])
            for s_idx, p in enumerate(metrics['props']):
                # Generate dataset labels: "Tr XX" for training sets, "Ts XX" for test sets
                lbl = f"Tr {s_idx+1:02}" if s_idx < self.n_train else f"Ts {s_idx-self.n_train+1:02}"
                # Mark violation: display "!!" if proportion < a or > b
                err = "!!" if p < self.a or p > self.b else ""
                if err:  # Accumulate violation count if violated
                    violations += 1
                # Print proportion (wrap line every 4 datasets for better readability)
                print(f"  {lbl} | Prop: {p:6.2%} {err}", end=" | ")
                if (s_idx + 1) % 4 == 0:
                    print("")
            
            # 4.3 Print aggregate bad rate metrics of the current group
            print(f"\n  Agg Train BR: {metrics['tr_br']:.4f} | Agg Test BR: {metrics['ts_br']:.4f} | Avg Test BR: {metrics['avg_ts_br']:.4f}")
            
            # 4.4 Check Constraints 2, 3, 4 if not the first group (compare with the previous group)
            if history:
                prev = history[-1]  # Metrics of the previous group
                # Check Constraint 3: Bad rate of current group < that of previous group (monotonically decreasing)
                m_tr = metrics['tr_br'] < prev['tr_br']  # Monotonicity of total training set
                m_ts = metrics['ts_br'] < prev['ts_br']  # Monotonicity of total test set
                m_avg = metrics['avg_ts_br'] < prev['avg_ts_br']  # Monotonicity of average test set
                # Check Constraint 2: Sum of proportions of current group and previous group ≤ C (must be satisfied for all datasets)
                adj_ok = np.all(metrics['props'] + prev['props'] <= self.C)
                
                # Print constraint check results
                print(f"  Checks: Mono Train: {m_tr} | Mono Test: {m_ts} | Mono Avg: {m_avg} | Adj Sum OK: {adj_ok}")
                # Accumulate violation count if any constraint is not satisfied
                if not all([m_tr, m_ts, m_avg, adj_ok]):
                    violations += 1
            
            # Add metrics of the current group to history for comparison with the next group
            history.append(metrics)
        
        # 5. Print audit summary (total number of violations)
        print(f"\n{'='*95}\nAUDIT COMPLETE. TOTAL VIOLATIONS: {violations}\n{'='*95}")


    def _get_metrics_raw(self, low, high, last):
        """
        Private method: Calculate raw metrics of each dataset based on the score interval [low, high] (used for audit)
        
        Parameters:
            low (float): Lower bound of the score interval
            high (float): Upper bound of the score interval
            last (bool): Whether it is the last group (the last group includes the upper bound, others do not)
        
        Returns:
            dict: Metric dictionary containing proportions of each dataset and aggregate bad rates
        """
        # Initialize to store proportions of each dataset and bad rates of each test set
        props, test_brs = [], []
        # Initialize the number of customers and bad samples for total training set and total test set
        tr_b, tr_c, ts_b, ts_c = 0, 0, 0, 0
        
        # Iterate over each dataset and calculate metrics
        for i, df in enumerate(self.all_dfs):
            # Generate score mask: determine if the score is within the current interval
            # Last group (last=True): include upper bound (score <= high); other groups: exclude upper bound (score < high)
            if last:
                mask = (df['score'] >= low) & (df['score'] <= high)
            else:
                mask = (df['score'] >= low) & (df['score'] < high)
            
            # Calculate the number of customers (c) and bad samples (b) of the current dataset in the interval
            c, b = len(df[mask]), df[mask]['target'].sum()
            # Calculate the customer proportion of the current dataset (number of customers in interval / total number of customers)
            p = c / len(df)
            props.append(p)
            
            # Accumulate the number of customers and bad samples by training set/test set classification
            if i < self.n_train:  # Training set
                tr_c += c
                tr_b += b
            else:  # Test set
                ts_c += c
                ts_b += b
                # Calculate the bad rate of the current test set (avoid division by zero)
                test_brs.append(b/c if c > 0 else 0)
        
        # Return metric dictionary: proportions, bad rate of total training set, bad rate of total test set, average bad rate of test sets
        return {
            'props': np.array(props),  # Proportions of each dataset (array)
            'tr_br': tr_b/tr_c,        # Bad rate of total training set (will report error if tr_c=0, but tr_c will not be 0 for valid binning)
            'ts_br': ts_b/ts_c,        # Bad rate of total test set
            'avg_ts_br': np.mean(test_brs)  # Average bad rate of test sets
        }


# ===================== Code Execution Entry =====================
if __name__ == "__main__":
    # 1. Set random seed (42) to ensure reproducibility of simulated data
    np.random.seed(42)
    
    # 2. Define data generation function gen: generate simulated datasets containing "score" and "target"
    def gen(n):
        """
        Generate simulated data
        
        Parameters:
            n (int): Number of samples in the dataset
        
        Returns:
            pd.DataFrame: DataFrame containing "score" (random integer from 300 to 850) and "target" (good/bad label)
        """
        # Generate random integers from 300 to 850 as scores
        s = np.random.randint(300, 850, n)
        # Generate target: bad sample probability is negatively correlated with score (higher score = lower bad sample probability)
        # 1/(1+np.exp((s-580)/50)): probability ≈ 0.5 when s=580, decreases as s increases, increases as s decreases
        target = np.random.binomial(1, 1/(1+np.exp((s-580)/50)))
        # Return DataFrame
        return pd.DataFrame({'score': s, 'target': target})
    
    # 3. Create BinningOptimizer instance: 10 training sets (1500 samples each), 6 test sets (800 samples each)
    opt = BinningOptimizer(
        train_dfs=[gen(1500) for _ in range(10)],  # 10 training sets, 1500 samples each
        test_dfs=[gen(800) for _ in range(6)],     # 6 test sets, 800 samples each
        a=0.05, b=0.25, C=0.50                    # Constraint parameters: 5% ≤ proportion ≤ 25%, sum of adjacent proportions ≤ 50%
    )
    
    # 4. Call solve method to find optimal binning boundaries
    cuts = opt.solve()
    
    # 5. Call audit method to generate audit report if valid binning boundaries exist
    if cuts:
        opt.audit(cuts)