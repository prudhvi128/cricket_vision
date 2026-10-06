"""
pipeline — Orchestration. Decides order, not arithmetic.

`analysis_pipeline.py` (`UnifiedPipeline`) sequences the stages, owns the stage
list and progress reporting, and persists the final result document. It contains
no detection maths, no trajectory maths, and no rendering. If a rule about what a
clip *means* needed changing, it belongs in `segmentation`, not here.
"""

from .analysis_pipeline import PipelineResult, UnifiedPipeline

__all__ = ["PipelineResult", "UnifiedPipeline"]