"""Independent FastWAM action-only integration (not Motus or Pi0-FAST).

Imports are deliberately lazy: configuration/preflight needs neither Torch,
the upstream FastWAM package, compiled extensions, nor a CUDA context.
"""
