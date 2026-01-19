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
    def __init__(self, a=5, b=25, c=30, target_bins=5, min_bad_samples=1, monotonicity='increasing', eps=1e-8):
        """Initialize Binner
        Parameters:
            a: Minimum proportion per bin (percentage)
            b: Maximum proportion per bin (percentage)
            c: Maximum sum of proportions for adjacent bins (percentage)
            target_bins: Target number of bins (try first, decrement if not feasible)
            min_bad_samples: Minimum number of bad samples (Y=1) required per bin (user-configurable)
            monotonicity: Default rate monotonicity ('increasing'/'decreasing')
            eps: Tolerance for floating-point calculation errors
        """
        # Validate min_bad_samples (must be positive integer)
        if not isinstance(min_bad_samples, int) or min_bad_samples < 1:
            raise ValueError("min_bad_samples must be a positive integer (≥1)")
        
        self.a = a / 100  # Convert to ratio
        self.b = b / 100
        self.c = c / 100
        self.target_bins = target_bins
        self.min_bad_samples = min_bad_samples  # New: User-configurable minimum bad samples per bin
        self.monotonicity = monotonicity
        self.eps = eps
        self.bin_edges = []  # Micro-bin boundaries (numeric)
        self.total_samples = 0
        self.total_default = 0
        self.total_non_default = 0
        self.bin_result = {}
        self.all_feasible_solutions = {}

    def get_bin_metrics(self, left, right):
        """Calculate basic metrics for bin [left, right]
        Returns: Metric dictionary (None if any constraint not met)
        Constraints checked here:
        1. Single bin proportion ∈ [a%, b%]
        2. At least min_bad_samples bad samples (Y=1) in the bin
        """
        mask = (self.sorted_X >= left) & (self.sorted_X <= right)
        bin_size = mask.sum()
        bin_ratio = bin_size / self.total_samples

        # Constraint 1: Single bin proportion ∈ [a%, b%]
        if bin_ratio < self.a - self.eps or bin_ratio > self.b + self.eps:
            return None

        # Calculate default-related metrics
        bin_default = self.sorted_Y[mask].sum()  # Number of bad samples in the bin
        bin_non_default = bin_size - bin_default

        # New Constraint: At least min_bad_samples bad samples
        if bin_default < self.min_bad_samples:
            return None

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

    def check_adjacent_and_monotonic(self, edges, verbose=False):
        """Check if the binning scheme meets:
        1. Sum of adjacent bin proportions ≤ c%
        2. Default rate monotonicity
        Returns: (Is valid, Total IV value, Detailed metric list)
        """
        metrics_list = []
        total_iv = 0.0
        default_rates = []
        adjacent_sums = []

        # Calculate basic metrics for each bin (ensure full sequence)
        for idx in range(len(edges)-1):
            left = edges[idx]
            right = edges[idx+1]
            metrics = self.get_bin_metrics(left, right)
            if metrics is None:
                if verbose:
                    print(f"Invalid bin: [{left:.4f}, {right:.4f}] (violates proportion or min bad samples constraint)")
                return False, 0.0, []
            metrics_list.append(metrics)
            total_iv += metrics['iv']
            default_rates.append(metrics['default_rate'])

        # Check 1: Sum of adjacent bin proportions ≤ c%
        for idx in range(len(metrics_list)-1):
            sum_ratio = metrics_list[idx]['bin_ratio'] + metrics_list[idx+1]['bin_ratio']
            adjacent_sums.append(sum_ratio)
            if sum_ratio > self.c + self.eps:
                if verbose:
                    print(f"Adjacent sum violation: Bin {idx} + Bin {idx+1} (sum={sum_ratio*100:.2f}% > c={self.c*100}%)")
                return False, 0.0, []

        # Check 2: Default rate monotonicity (strict check for full sequence)
        is_monotonic = True
        for idx in range(1, len(default_rates)):
            prev = default_rates[idx-1]
            curr = default_rates[idx]
            if self.monotonicity == 'increasing':
                if curr < prev - self.eps:
                    is_monotonic = False
                    if verbose:
                        print(f"Monotonicity violation (increasing): Bin {idx} (DR={curr:.4f}) < Bin {idx-1} (DR={prev:.4f})")
                    break
            else:
                if curr > prev + self.eps:
                    is_monotonic = False
                    if verbose:
                        print(f"Monotonicity violation (decreasing): Bin {idx} (DR={curr:.4f}) > Bin {idx-1} (DR={prev:.4f})")
                    break
        if not is_monotonic:
            return False, 0.0, []

        if verbose:
            print(f"All constraints satisfied: Default rates = {[round(dr,4) for dr in default_rates]}")
        return True, total_iv, metrics_list

    def find_best_solution_for_k(self, k):
        """Find the optimal solution with exactly k bins that meets all constraints and maximizes IV
        Parameter: k: Target number of bins
        Returns: (Is found, Maximum IV, Optimal boundaries)
        """
        if k < 1 or k > len(self.bin_edges)-1:
            return False, 0.0, []

        # DP initialization: dp[i][j] = (max_iv, prev_split_idx)
        # prev_split_idx: the end micro-bin index of (j-1)-th bin
        micro_bins_count = len(self.bin_edges) - 1
        dp = [[(-np.inf, None) for _ in range(k+1)] for __ in range(micro_bins_count+1)]

        # Initialize: Case with 1 bin (j=1)
        for i in range(1, micro_bins_count+1):
            edges = [self.bin_edges[0], self.bin_edges[i]]
            is_valid, iv, _ = self.check_adjacent_and_monotonic(edges)
            if is_valid:
                dp[i][1] = (iv, None)  # prev_split_idx = None for 1 bin

        # DP transition: j from 2 to k (number of bins), i from j to micro_bins_count (end micro-bin)
        for j in range(2, k+1):
            for i in range(j, micro_bins_count+1):
                # Iterate all possible previous split points (m = end of (j-1)-th bin)
                for m in range(j-1, i):
                    # Skip if (j-1) bins for first m micro-bins is infeasible
                    if dp[m][j-1][0] == -np.inf:
                        continue

                    # Check current bin (m -> i) is valid (proportion + min bad samples constraints)
                    curr_bin_edges = [self.bin_edges[m], self.bin_edges[i]]
                    curr_metrics = self.get_bin_metrics(curr_bin_edges[0], curr_bin_edges[1])
                    if curr_metrics is None:
                        continue

                    # Fully backtrack all boundaries of (j-1) bins
                    prev_edges = []
                    temp_split_idx = m  # Start from the end of (j-1)-th bin
                    temp_j = j - 1       # Current number of bins in previous step
                    
                    # Loop backtracking: until temp_j=0 (cover all j-1 bins)
                    while temp_j > 0:
                        prev_edges.append(self.bin_edges[temp_split_idx])
                        # Get previous split index (None if temp_j=1)
                        next_split_idx = dp[temp_split_idx][temp_j][1]
                        # Update for next iteration
                        temp_split_idx = next_split_idx
                        temp_j -= 1
                    
                    # Add start boundary and reverse to get complete (j-1) bin boundaries
                    prev_edges.append(self.bin_edges[0])
                    prev_edges.reverse()
                    # Concatenate full boundaries: (j-1) bins + current bin
                    full_edges = prev_edges + [self.bin_edges[i]]
                    # Deduplicate (avoid invalid bins from duplicate boundaries)
                    full_edges = sorted(list(set(full_edges)))
                    
                    # Validate all constraints for the full binning scheme
                    is_valid, total_iv, _ = self.check_adjacent_and_monotonic(full_edges, verbose=False)
                    if is_valid:
                        # Calculate total IV: IV of (j-1) bins + IV of current bin
                        candidate_iv = dp[m][j-1][0] + curr_metrics['iv']
                        # Update DP table (keep maximum IV)
                        if candidate_iv > dp[i][j][0]:
                            dp[i][j] = (candidate_iv, m)

        # Find the maximum IV solution for exactly k bins
        max_iv = -np.inf
        best_end_micro = -1
        for i in range(k, micro_bins_count+1):
            if dp[i][k][0] > max_iv:
                max_iv = dp[i][k][0]
                best_end_micro = i

        if max_iv == -np.inf:
            print(f"No feasible solution for k={k} bins (violates constraints: proportion, min bad samples, adjacent sum, or monotonicity)")
            return False, 0.0, []

        # Backtrack to generate final boundaries (ensure completeness)
        final_edges = []
        current_split_idx = best_end_micro
        current_j = k
        while current_j > 0:
            final_edges.append(self.bin_edges[current_split_idx])
            current_split_idx = dp[current_split_idx][current_j][1]
            current_j -= 1
        final_edges.append(self.bin_edges[0])
        final_edges.reverse()
        final_edges = sorted(list(set(final_edges)))

        # Double validation: Final boundaries must meet all constraints
        final_is_valid, final_total_iv, _ = self.check_adjacent_and_monotonic(final_edges, verbose=True)
        if not final_is_valid:
            print(f"Unexpected constraint violation for k={k} bins (rejecting solution)")
            return False, 0.0, []

        return True, final_total_iv, final_edges

    def solve_stage1(self, X, Y, n_micro=100):
        """Stage 1: Core logic - Decrement from target_bins to find feasible solution with maximum IV
        Parameters:
            X: Binning variable
            Y: Target variable (0=non-default, 1=default)
            n_micro: Initial number of micro-bins
        Returns: Binning result dictionary
        """
        print(f"Starting binning: Priority target bin count = {self.target_bins}, decrement if not feasible...")
        print(f"Constraints: min_bad_samples={self.min_bad_samples}, a={self.a*100}%, b={self.b*100}%, c={self.c*100}%, monotonicity={self.monotonicity}")
        print("Stage 1: Generating initial micro-bin boundaries...")

        # Preprocessing: Sort X and Y
        self.sorted_indices = np.argsort(X)
        self.sorted_X = X.iloc[self.sorted_indices].values if isinstance(X, pd.Series) else X[self.sorted_indices]
        self.sorted_Y = Y.iloc[self.sorted_indices].values if isinstance(Y, pd.Series) else Y[self.sorted_indices]
        self.total_samples = len(self.sorted_X)
        self.total_default = self.sorted_Y.sum()
        self.total_non_default = self.total_samples - self.total_default

        # Validate if total bad samples is sufficient for min_bad_samples per bin
        min_total_bad_needed = self.target_bins * self.min_bad_samples
        if self.total_default < min_total_bad_needed:
            print(f"Warning: Total bad samples ({self.total_default}) < minimum required ({min_total_bad_needed}) for target_bins={self.target_bins}. Will decrement bin count.")

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
        max_attempt_k = min(self.target_bins, micro_bins_count)

        for k in range(max_attempt_k, 0, -1):
            # Skip if total bad samples is insufficient for k bins
            if self.total_default < k * self.min_bad_samples:
                print(f"Skipping k={k}: Insufficient total bad samples ({self.total_default} < {k*self.min_bad_samples} required)")
                continue
            
            print(f"\nTrying bin count k = {k}...")
            is_found, curr_iv, curr_edges = self.find_best_solution_for_k(k)
            if is_found and curr_iv > best_iv:
                best_k = k
                best_iv = curr_iv
                best_edges = curr_edges
                print(f"Bin count k = {k} meets all constraints, IV = {curr_iv:.6f} (current optimal)")
                break

        if best_k == -1:
            raise ValueError(f"No feasible solution even with 1 bin. Please adjust parameters (e.g., reduce min_bad_samples, a, or b)")

        # Step 3: Calculate detailed metrics for final bins
        bin_metrics = []
        for idx in range(len(best_edges)-1):
            left = best_edges[idx]
            right = best_edges[idx+1]
            if idx == len(best_edges)-2:
                interval = f"[{left:.4f}, {right:.4f}]"
            else:
                interval = f"[{left:.4f}, {right:.4f})"
            
            metrics = self.get_bin_metrics(left, right)
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
            'Constraint Parameters': {
                'Minimum Proportion per Bin a (%)': self.a * 100,
                'Maximum Proportion per Bin b (%)': self.b * 100,
                'Maximum Adjacent Ratio Sum c (%)': self.c * 100,
                'Minimum Bad Samples per Bin': self.min_bad_samples,
                'Default Rate Monotonicity': self.monotonicity
            },
            'Bin Boundaries': [round(e, 4) for e in best_edges],
            'Bin Detailed Metrics': pd.DataFrame(bin_metrics)
        }
        print(f"\nFinal Binning Result: Target Bin Count = {self.target_bins} → Optimal Bin Count = {best_k}, Total IV = {best_iv:.6f}")
        return self.bin_result

    def validate_bin_result(self):
        """Validate if binning result meets all constraints
        Validation items: 
        1. Single bin ≥ a% 2. Single bin ≤ b% 3. Adjacent sum ≤ c% 
        4. Default rate monotonicity 5. At least min_bad_samples bad samples per bin
        Returns: Validation result dictionary
        """
        if not self.bin_result:
            raise ValueError("Please run solve_stage1 first to generate binning result!")
        
        metrics_df = self.bin_result['Bin Detailed Metrics']
        a = self.a * 100
        b = self.b * 100
        c = self.c * 100
        min_bad = self.min_bad_samples
        monotonicity = self.monotonicity

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

        # Validation 4: At least min_bad_samples bad samples per bin (NEW)
        insufficient_bad_bins = metrics_df[metrics_df['Default Count'] < min_bad]['Bin Interval'].tolist()
        validate_result['Constraint Item'].append(f'Minimum Bad Samples per Bin ≥ {min_bad}')
        validate_result['Validation Result'].append('Pass' if not insufficient_bad_bins else 'Fail')
        validate_result['Non-Compliant Bins'].append(','.join(insufficient_bad_bins) if insufficient_bad_bins else '-')
        validate_result['Remarks'].append(f'Required: ≥ {min_bad} bad samples')

        # Validation 5: Default rate monotonicity
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
            self.validate_bin_result()
        
        bin_metrics_df = self.bin_result['Bin Detailed Metrics'].drop(columns=['Raw Ratio', 'Raw Default Rate'], errors='ignore')
        validate_df = pd.DataFrame(self.validate_result)
        
        return bin_metrics_df, validate_df

# -------------------------- Test Code --------------------------
if __name__ == "__main__":
    # Generate test data (60,000 samples)
    np.random.seed(42)
    n_samples = 60000
    X = np.random.normal(100, 20, n_samples)
    # Artificially construct a scenario with potential monotonicity issues
    default_prob = np.where(
        X < 90, 0.1,  # Low X: 10% default rate
        np.where(X < 110, 0.08,  # Medium X: 8% default rate
                 0.3)  # High X: 30% default rate
    )
    Y = np.random.binomial(1, default_prob)

    # Initialize binner (user-configurable min_bad_samples=5)
    binner = DPBinner(
        a=5,              # Minimum single bin proportion 5%
        b=25,             # Maximum single bin proportion 25%
        c=30,             # Maximum adjacent ratio sum 30%
        target_bins=3,    # Target bin count 3
        min_bad_samples=5, # New: At least 5 bad samples per bin
        monotonicity='increasing'
    )

    # 1. Perform binning
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