"""Drift monitoring: per-column metrics, the thresholds that turn them into an alert.

``metrics`` computes distances between distributions; ``detector`` decides whether they
matter; ``report`` renders the verdict. The split exists so the decision logic can be
tested without any rendering, and evaluated in bulk (session 4) without any I/O.
"""
