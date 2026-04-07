Objective
To obtain a binning result that maximizes the width of the first bin, subject to the following four constraints.
Constraints
The loss-adjusted return in the first bin is less than 0, where loss-adjusted return is defined as return minus loss.
The write-off expenses in the first bin account for at least 10% of total write-off expenses (this 10% threshold is adjustable later).
The default rate across all bins must be monotonically increasing.
Constraints 1 to 3 must hold simultaneously for 1 development sample and 6 test samples.
Additional Notes
A key issue to consider: as the first bin is gradually expanded, the proportion of write-off expenses within this bin relative to total write-off expenses keeps increasing. However, including more observations will cause the loss-adjusted return to turn positive. We can first identify the critical point, then proceed with further optimization to ensure the default rate is monotonically increasing across all bins for both the 1 development sample and the 6 test samples. Use the scores to cut the bins, customers with low scores will have higher default rate.