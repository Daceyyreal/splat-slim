"""The four post-training stages, in pipeline order (paper section 3).

``prune`` (adaptive opacity pruning), ``outliers`` (spatial and scale clean-up),
``sh`` (spherical-harmonic degree reduction) and ``quantize`` (fp16 / int8 / mixed /
int8-subgroup).
"""
