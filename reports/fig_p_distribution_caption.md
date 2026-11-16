**Fig. S9. Predicted-probability distribution on the external test set.**
Predicted probability (raw model output, no calibration) of every spectrum (protocol scoring; τ = 0.5 → 63 TP / 4 FP / 1 FN);
positives (top) and negatives (bottom) shown separately as density-normalised histograms, logarithmic density axis. Positives concentrate
at high probability (57/64 above 0.9; median 0.999); the single false negative (#12, p = 0.012)
sits deep in the negative region yet on a steep boundary (α ≈ 0.20). Among 4,438 negatives
only 4 exceed the default threshold (the four FPs) and 4,428/4,438 stay below 0.1. The decision
threshold was deliberately left at its default value of 0.5 and not tuned on the external set:
a carefully tuned threshold may fail to transfer to other screening scenarios, whereas the
default threshold requires no test-set-specific adjustment.
