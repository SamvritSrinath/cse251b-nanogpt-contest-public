"""Submission entrypoint shim for the contest evaluation script.

The evaluation script imports ``load_model`` from ``model.py`` at the root of the
submission directory. The actual implementation lives in ``src.model`` so the
same code path is used during local training, export, and evaluation.
"""

from src.model import GPT, load_model

__all__ = ["GPT", "load_model"]
