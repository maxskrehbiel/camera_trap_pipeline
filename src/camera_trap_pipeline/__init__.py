"""Turn folders of camera-trap photos into a clean, queryable dataset.

Every stage checkpoints to the output folder, so an interrupted run resumes where it stopped.
"""

import importlib.metadata

__version__ = importlib.metadata.version("camera_trap_pipeline")
