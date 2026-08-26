"""EntroFlow runtime communication-entropy plug-in core."""

from .entropy import EdgeEntropyEvent, RollingEdgeEntropy, normalized_entropy

__all__ = ["EdgeEntropyEvent", "RollingEdgeEntropy", "normalized_entropy"]
__version__ = "0.2.0"
