"""Evaluation harness: the version-controlled test set, a runner that drives a running API,
and the scoring. Kept outside the `medrag` package on purpose: the system must never see
or import the test set (the graders use a hidden one)."""
