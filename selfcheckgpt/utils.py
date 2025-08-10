"""Minimal stubs for the original :mod:`selfcheckgpt` utilities.

This project only relies on a tiny subset of the real library.  The
functions defined here provide the interfaces expected by
``selfcheck_metrics`` so that the tests can run without the heavy
dependencies of the original project.  They intentionally implement only
the minimal behaviour required by the tests; the generation and
answering helpers merely raise :class:`NotImplementedError` if invoked.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MQAGConfig:
    """Model names used by the MQAG metric.

    The real project stores default HuggingFace model identifiers here.
    For the purposes of the tests we simply keep placeholder strings.
    """

    generation1_squad: str = "g1"
    generation2: str = "g2"
    answering: str = "qa"


def _not_impl(*args, **kwargs):  # pragma: no cover - defensive stub
    raise NotImplementedError("This stub does not implement the full logic")


prepare_qa_input = _not_impl
prepare_distractor_input = _not_impl
prepare_answering_input = _not_impl

