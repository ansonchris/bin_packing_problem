import pandas as pd
import numpy as np

eps = 1e-8  # Avoid floating-point calculation errors and log(0) issues

def calculate_woe(non_default, default, total_non_default, total_default):
    """Calculate WOE (Weight of Evidence) for a single bin
    WOE = ln(Good sample proportion / Bad sample proportion)
    Good samples: Y=0 (non-default); Bad samples: Y=1 (default)
    """
    p_good = non_default / (total_non_default + eps)
    p_bad = default / (total_default + eps)
    return np.log((p_good + eps) / (p_bad + eps))

def calculate_iv(non_default, default, total_non_default, total_default):
    """Calculate IV (Information Value) for a single bin
    IV = (Good sample proportion - Bad sample proportion) * WOE
    """
    woe = calculate_woe(non_default, default, total_non_default, total_default)
    p_good = non_default / (total_non_default + eps)
    p_bad = default / (total_default + eps)
    return (p_good - p_bad) * woe

class DPBinner:
    def __init__(self, a=5, b=25, c=30, target_bins=5, monotonicity='increasing', eps=1e-8):
        """Initialize Binner
        Parameters:
            a: Minimum proportion per bin (percentage)
            b: Maximum proportion per bin (percentage)
            c: Maximum sum of proportions for adjacent bins (percentage)
            target_bins: Target number of bins (try first, decrement if not feasible)
            monotonicity: Default rate monotonicity ('increasing'/'decreasing')
            eps: Tolerance for floating-point calculation errors
        """
        self.a = a / 100  # Convert to ratio
        self.b = b / 100
        self.c = c / 100
        self.target_bins = target_bins  # Restore target bin count parameter
        self.monotonicity = monotonicity
        self.eps = eps
        self.bin_edges = []  # Micro-bin boundaries (numeric)
        self.total_samples = 0
        self.total_default = 0
        self.total_non_default = 0
        self.bin_result = {}  # Store final binning result
        self.all_feasible_solutions = {}  # Store all feasible solutions (k: {'iv': total_iv, 'edges': boundaries})

    def get_bin_metrics(self, left, right):
        """Calculate basic metrics for bin [left, right]
        Returns: Metric dictionary (None if single bin proportion constraint not met)
        """
        mask = (self.sorted_X >= left) & (self.sorted_X <= right)
        bin_size = mask.sum()
        bin_ratio = bin_size / self.total_samples

        # Check single bin proportion constraint (a% - b%)
        if bin_ratio < self.a - self.eps or bin_ratio > self.b + self.eps:
            return None

        # Calculate default-related metrics
        bin_default = self.sorted_Y[mask].sum()
        bin_non_default = bin_size - bin_default
        default_rate = bin_default / (bin_size + self.eps)

        # Calculate WOE and IV
        woe = calculate_woe(bin_non_default, bin_default, self.total_non_default, self.total_default)
        iv = calculate_iv(bin_non_default, bin_default, self.total_non_default, self.total_default)

        return {
            'left': left,
            'right': right,
            'bin_size': bin_size,
            'bin_ratio': bin_ratio,
            'non_default': bin_non_default,
            'default': bin_default,
            'default_rate': default_rate,
            'woe': woe,
            'iv': iv
        }

    def check_adjacent_and_monotonic(self, edges):
        """Check if the binning scheme meets:
        1. Sum of adjacent bin proportions ≤ c%
        2. Default rate monotonicity
        Returns: (Is valid, Total IV value, Detailed metric list)
        """
        metrics_list = []
        total_iv = 0.0
        default_rates = []
        adjacent_sums = []

        # Calculate basic metrics for each bin
        for idx in range(len(edges)-1):
            left = edges[idx]
            right = edges[idx+1]
            metrics = self.get_bin_metrics(left, right)
            if metrics is None:
                return False, 0.0, []  # Single bin proportion not met
            metrics_list.append(metrics)
            total_iv += metrics['iv']
            default_rates.append(metrics['default_rate'])

        # Check 1: Sum of adjacent bin proportions ≤ c%
        for idx in range(len(metrics_list)-1):
            sum_ratio = metrics_list[idx]['bin_ratio'] + metrics_list[idx+1]['bin_ratio']
            adjacent_sums.append(sum_ratio)
            if sum_ratio > self.c + self.eps:
                return False, 0.0, []

        # Check 2: Default rate monotonicity
        is_monotonic = True
        for idx in range(1, len(default_rates)):
            prev = default_rates[idx-1]
            curr = default_rates[idx]
            if self.monotonicity == 'increasing':
                if curr < prev - self.eps:
                    is_monotonic = False
                    break
            else:
                if curr > prev + self.eps:
                    is_monotonic = False
                    break
        if not is_monotonic:
            return False, 0.0, []

        return True, total_iv, metrics_list

    def find_best_solution_for_k(self, k):
        """Find the optimal solution with exactly k bins that meets all constraints and maximizes IV
        Parameter: k: Target number of bins
        Returns: (Is found, Maximum IV, Optimal boundaries)
        """
        if k < 1 or k > len(self.bin_edges)-1:
            return False, 0.0, []

        # DP initialization: dp[i][j] = (max_iv, prev_j) 
        # Represents max IV for first i micro-bins split into j bins and previous split point
        micro_bins_count = len(self.bin_edges) - 1  # Number of micro-bins
        dp = [[(-np.inf, None) for _ in range(k+1)] for __ in range(micro_bins_count+1)]
        
        # Initialize: Case with 1 bin
        for i in range(1, micro_bins_count+1):
            edges = [self.bin_edges[0], self.bin_edges[i]]
            is_valid, iv, _ = self.check_adjacent_and_monotonic(edges)
            if is_valid:
                dp[i][1] = (iv, None)

        # DP transition: j from 2 to k, i from j to micro_bins_count
        for j in range(2, k+1):
            for i in range(j, micro_bins_count+1):
                # Iterate all possible previous split points m
                for m in range(j-1, i):
                    if dp[m][j-1][0] == -np.inf:
                        continue
                    # Check if current bin (m+1 to i) + previous bins meet constraints
                    edges = [self.bin_edges[m], self.bin_edges[i]]
                    curr_metrics = self.get_bin_metrics(edges[0], edges[1])
                    if curr_metrics is None:
                        continue
                    # Concatenate previous boundaries + current boundaries and check overall constraints
                    prev_edges = []
                    temp_m = m
                    temp_j = j-1
                    # Backtrack to get previous boundaries
                    while temp_j > 0 and dp[temp_m][temp_j][1] is not None:
                        prev_edges.append(self.bin_edges[temp_m])
                        temp_m = dp[temp_m][temp_j][1]
                        temp_j -= 1
                    prev_edges.append(self.bin_edges[0])
                    prev_edges.reverse()
                    full_edges = prev_edges + [self.bin_edges[i]]
                    is_valid, total_iv, _ = self.check_adjacent_and_monotonic(full_edges)
                    if is_valid and (dp[m][j-1][0] + curr_metrics['iv']) > dp[i][j][0]:
                        dp[i][j] = (dp[m][j-1][0] + curr_metrics['iv'], m)

        # Find boundaries with maximum IV
        max_iv = -np.inf
        best_end = -1
        for i in range(k, micro_bins_count+1):
            if dp[i][k][0] > max_iv:
                max_iv = dp[i][k][0]
                best_end = i

        if max_iv == -np.inf:
            return False, 0.0, []

        # Backtrack to get boundaries
        edges = []
        current_i = best_end
        current_j = k
        while current_j > 0:
            edges.append(self.bin_edges[current_i])
            current_i = dp[current_i][current_j][1]
            current_j -= 1
        edges.append(self.bin_edges[0])
        edges.reverse()
        edges = sorted(list(set(edges)))  # Deduplicate

        return True, max_iv, edges

    def solve_stage1(self, X, Y, n_micro=100):
        """Stage 1: Core logic - Decrement from target_bins to find feasible solution with maximum IV
        Parameters:
            X: Binning variable
            Y: Target variable (0=non-default, 1=default)
            n_micro: Initial number of micro-bins
        Returns: Binning result dictionary
        """
        print(f"Starting binning: Priority target bin count = {self.target_bins}, decrement if not feasible...")
        print("Stage 1: Generating initial micro-bin boundaries...")

        # Preprocessing: Sort X and Y
        self.sorted_indices = np.argsort(X)
        self.sorted_X = X.iloc[self.sorted_indices].values if isinstance(X, pd.Series) else X[self.sorted_indices]
        self.sorted_Y = Y.iloc[self.sorted_indices].values if isinstance(Y, pd.Series) else Y[self.sorted_indices]
        self.total_samples = len(self.sorted_X)
        self.total_default = self.sorted_Y.sum()
        self.total_non_default = self.total_samples - self.total_default

        # Step 1: Generate micro-bin boundaries with qcut
        combined = pd.Series(self.sorted_X)
        _, self.bin_edges = pd.qcut(combined, q=n_micro, retbins=True, duplicates='drop')
        self.bin_edges = sorted(list(set(self.bin_edges)))
        if len(self.bin_edges) < 2:
            raise ValueError("Insufficient micro-bin boundaries to perform binning!")
        micro_bins_count = len(self.bin_edges) - 1
        print(f"Initial number of micro-bins: {micro_bins_count} (after deduplication)")

        # Step 2: Decrement from target_bins to find first feasible and maximum IV solution
        best_k = -1
        best_iv = -np.inf
        best_edges = []
        max_attempt_k = min(self.target_bins, micro_bins_count)  # Maximum attempt up to micro-bin count

        for k in range(max_attempt_k, 0, -1):
            print(f"Trying bin count k = {k}...")
            is_found, curr_iv, curr_edges = self.find_best_solution_for_k(k)
            if is_found and curr_iv > best_iv:
                best_k = k
                best_iv = curr_iv
                best_edges = curr_edges
                print(f"Bin count k = {k} meets all constraints, IV = {curr_iv:.6f} (current optimal)")
                break  # Terminate immediately after finding first feasible k (closest to target_bins)

        if best_k == -1:
            raise ValueError("No feasible solution even with 1 bin. Please adjust a/b/c/monotonicity parameters!")

        # Step 3: Calculate detailed metrics for final bins
        bin_metrics = []
        for idx in range(len(best_edges)-1):
            left = best_edges[idx]
            right = best_edges[idx+1]
            # Handle interval representation (last bin is closed interval)
            if idx == len(best_edges)-2:
                interval = f"[{left:.4f}, {right:.4f}]"
            else:
                interval = f"[{left:.4f}, {right:.4f})"
            
            metrics = self.get_bin_metrics(left, right)
            # Calculate sum of adjacent proportions
            adjacent_sum = "-"
            if idx < len(best_edges)-2:
                next_metrics = self.get_bin_metrics(best_edges[idx+1], best_edges[idx+2])
                adjacent_sum = round((metrics['bin_ratio'] + next_metrics['bin_ratio'])*100, 2)
            
            bin_metrics.append({
                'Bin Interval': interval,
                'Sample Count': metrics['bin_size'],
                'Ratio (%)': round(metrics['bin_ratio']*100, 2),
                'Non-Default Count': metrics['non_default'],
                'Default Count': metrics['default'],
                'Default Rate (%)': round(metrics['default_rate']*100, 2),
                'WOE Value': round(metrics['woe'], 6),
                'IV Value': round(metrics['iv'], 6),
                'Adjacent Ratio Sum (%)': adjacent_sum,
                'Raw Ratio': metrics['bin_ratio'],
                'Raw Default Rate': metrics['default_rate']
            })

        # Organize final result
        self.bin_result = {
            'Target Bin Count': self.target_bins,
            'Optimal Bin Count': best_k,
            'Total IV Value': round(best_iv, 6),
            'Initial Micro-Bin Count': micro_bins_count,
            'Bin Boundaries': [round(e, 4) for e in best_edges],
            'Constraint Parameters': {
                'Minimum Proportion per Bin a (%)': self.a * 100,
                'Maximum Proportion per Bin b (%)': self.b * 100,
                'Maximum Adjacent Ratio Sum c (%)': self.c * 100,
                'Default Rate Monotonicity': self.monotonicity
            },
            'Bin Detailed Metrics': pd.DataFrame(bin_metrics)
        }
        print(f"\nFinal Binning Result: Target Bin Count = {self.target_bins} → Optimal Bin Count = {best_k}, Total IV = {best_iv:.6f}")
        return self.bin_result

    def validate_bin_result(self):
        """Validate if binning result meets all constraints
        Validation items: 1. Single bin ≥ a% 2. Single bin ≤ b% 3. Adjacent sum ≤ c% 4. Default rate monotonicity
        Returns: Validation result dictionary
        """
        if not self.bin_result:
            raise ValueError("Please run solve_stage1 first to generate binning result!")
        
        metrics_df = self.bin_result['Bin Detailed Metrics']
        a = self.a * 100
        b = self.b * 100
        c = self.c * 100
        monotonicity = self.monotonicity

        # Initialize validation result
        validate_result = {
            'Constraint Item': [],
            'Validation Result': [],
            'Non-Compliant Bins': [],
            'Remarks': []
        }

        # Validation 1: Each bin proportion ≥ a%
        low_ratio_bins = metrics_df[metrics_df['Ratio (%)'] < a - self.eps]['Bin Interval'].tolist()
        validate_result['Constraint Item'].append('Minimum Single Bin Proportion ≥ a%')
        validate_result['Validation Result'].append('Pass' if not low_ratio_bins else 'Fail')
        validate_result['Non-Compliant Bins'].append(','.join(low_ratio_bins) if low_ratio_bins else '-')
        validate_result['Remarks'].append(f'a = {a}%')

        # Validation 2: Each bin proportion ≤ b%
        high_ratio_bins = metrics_df[metrics_df['Ratio (%)'] > b + self.eps]['Bin Interval'].tolist()
        validate_result['Constraint Item'].append('Maximum Single Bin Proportion ≤ b%')
        validate_result['Validation Result'].append('Pass' if not high_ratio_bins else 'Fail')
        validate_result['Non-Compliant Bins'].append(','.join(high_ratio_bins) if high_ratio_bins else '-')
        validate_result['Remarks'].append(f'b = {b}%')

        # Validation 3: Sum of adjacent bin proportions ≤ c%
        exceed_adjacent_bins = []
        for idx in range(len(metrics_df)-1):
            curr_ratio = metrics_df.iloc[idx]['Raw Ratio']
            next_ratio = metrics_df.iloc[idx+1]['Raw Ratio']
            sum_ratio = (curr_ratio + next_ratio) * 100
            if sum_ratio > c + self.eps:
                exceed_adjacent_bins.append(f"{metrics_df.iloc[idx]['Bin Interval']} + {metrics_df.iloc[idx+1]['Bin Interval']} (Sum = {sum_ratio:.2f}%)")
        
        validate_result['Constraint Item'].append('Maximum Adjacent Bin Ratio Sum ≤ c%')
        validate_result['Validation Result'].append('Pass' if not exceed_adjacent_bins else 'Fail')
        validate_result['Non-Compliant Bins'].append('; '.join(exceed_adjacent_bins) if exceed_adjacent_bins else '-')
        validate_result['Remarks'].append(f'c = {c}%')

        # Validation 4: Default rate monotonicity
        default_rates = metrics_df['Raw Default Rate'].tolist()
        is_monotonic = True
        non_monotonic_pos = []
        for idx in range(1, len(default_rates)):
            prev = default_rates[idx-1]
            curr = default_rates[idx]
            if monotonicity == 'increasing':
                if curr < prev - self.eps:
                    is_monotonic = False
                    non_monotonic_pos.append(f"Bin {idx} (Default Rate {curr:.4f} < Bin {idx-1} {prev:.4f})")
            else:
                if curr > prev + self.eps:
                    is_monotonic = False
                    non_monotonic_pos.append(f"Bin {idx} (Default Rate {curr:.4f} > Bin {idx-1} {prev:.4f})")
        
        validate_result['Constraint Item'].append('Default Rate Monotonicity')
        validate_result['Validation Result'].append('Pass' if is_monotonic else 'Fail')
        validate_result['Non-Compliant Bins'].append('; '.join(non_monotonic_pos) if non_monotonic_pos else '-')
        validate_result['Remarks'].append(f'Requirement: {monotonicity}')

        # Summary of overall validation result
        all_passed = all([res == 'Pass' for res in validate_result['Validation Result']])
        validate_result['Constraint Item'].append('All Constraints')
        validate_result['Validation Result'].append('Pass' if all_passed else 'Fail')
        validate_result['Non-Compliant Bins'].append('-')
        validate_result['Remarks'].append('All constraints met / Some constraints not met')

        self.validate_result = validate_result
        return validate_result

    def export_results(self):
        """Standardize output: Export bin metrics and validation results as DataFrames
        Returns:
            bin_metrics_df: Bin detailed metrics DataFrame
            validate_df: Validation result DataFrame
        """
        if not self.bin_result:
            raise ValueError("Please run solve_stage1 first to generate binning result!")
        if not hasattr(self, 'validate_result'):
            self.validate_bin_result()  # Automatically run validation
        
        # Bin metrics DataFrame (remove temporary fields)
        bin_metrics_df = self.bin_result['Bin Detailed Metrics'].drop(columns=['Raw Ratio', 'Raw Default Rate'], errors='ignore')
        
        # Validation result DataFrame
        validate_df = pd.DataFrame(self.validate_result)
        
        return bin_metrics_df, validate_df

# -------------------------- Test Code --------------------------
if __name__ == "__main__":
    # Generate test data (60,000 samples)
    np.random.seed(42)
    n_samples = 60000
    X = np.random.normal(100, 20, n_samples)
    default_prob = 1 / (1 + np.exp(-(X - 100) / 15))  # Default rate increases with X
    Y = np.random.binomial(1, default_prob)

    # Initialize binner (set target_bins=6, try 6 bins first)
    binner = DPBinner(
        a=5,          # Minimum single bin proportion 5%
        b=25,         # Maximum single bin proportion 25%
        c=30,         # Maximum adjacent ratio sum 30%
        target_bins=6, # Target bin count 6
        monotonicity='increasing'
    )

    # 1. Perform binning (core logic: decrement from target_bins=6 to find optimal)
    bin_result = binner.solve_stage1(X=pd.Series(X), Y=pd.Series(Y), n_micro=100)

    # 2. Perform constraint validation
    validate_result = binner.validate_bin_result()
    print("\n" + "="*120)
    print("Binning Constraint Validation Results")
    for k, v in validate_result.items():
        print(f"{k}: {v}")

    # 3. Standardize output as DataFrames
    bin_metrics_df, validate_df = binner.export_results()
    print("="*120)
    print("\nBin Detailed Metrics DataFrame:")
    print(bin_metrics_df.to_string(index=False))

    print("="*120)
    print("\nBinning Validation Result DataFrame:")
    print(validate_df.to_string(index=False))