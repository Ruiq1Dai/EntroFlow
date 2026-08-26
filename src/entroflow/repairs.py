"""Constrained runtime repair actions shared by all routing policies."""

from __future__ import annotations

from enum import Enum


class RepairAction(str, Enum):
    NONE = "none"
    INSERT_VALIDATION = "insert_validation"
    EVIDENCE_BYPASS = "evidence_bypass"
    RETRY_WITH_CONSTRAINTS = "retry_with_constraints"
    REORDER_VALIDATION = "reorder_validation"
    EXPAND_EVIDENCE = "expand_evidence"
    CROSS_CHECK = "cross_check"
    COMPRESS_CONTEXT = "compress_context"


ALLOWED_REPAIR_ACTIONS = frozenset(RepairAction)
