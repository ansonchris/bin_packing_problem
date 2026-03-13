Readme_binpack.txt
The binning methodology is a two-stage optimization framework designed to find the maximum possible number of bins while strictly adhering to complex multi-dataset constraints.
1.    Atomic Partitioning and Metric Calculation
The process begins by dividing the entire PD range into a high number of "micro-bins" (defaulting to 100 quantiles). For any potential bin defined by these micro-edges, the framework calculates a comprehensive set of metrics, including proportions, bad rates, and average test bad rate across all datasets (training and test),for different constraints.
2.    Stage 1: Global Optimization via Dynamic Programming (DP)
The framework employs dynamic programming to find the globally optimal "chain" of bins.
Candidate Search: It evaluates all possible start and end points (i, j) among the micro-edges.
Constraint Filtering: A candidate bin is considered if it satisfies all the constraints.
Chain Building: After identifying all the possible candidate bins, the algorithm returns the longest valid path of bins found via the DP table.
3.    Stage 2: Recursive Repair (Backtracking)
Because the initial DP search might still contain edge-case violations or can be improved, the framework enters a recursive repair stage. If a violation is detected in the path from Stage 1, the algorithm branches into three different repair strategies to find the best resolution:
Path L: Merging the violating bin with its left neighbor.
Path R: Merging the violating bin with its right neighbor.
Path LL: Merging the two bins immediately preceding the violation (i.e., bins i-2 and i-1) to alter the comparison baseline. Since the algorithm processes bins sequentially from left to right, merging only the preceding bins preserves the monotonic trend already established among earlier bins.
The algorithm recursively explores these branches and selects the final path that yields the maximum number of valid bins.
