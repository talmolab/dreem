"""Module containing flag codes for frames and other data structures.

This module provides an extensible enum-like system for flagging frames and other
objects with specific reason codes. This allows for type-safe, scalable flagging
without passing around string literals.
"""

from enum import Enum


class FrameFlagCode(Enum):
    """Enumeration of flag codes for Frame objects.

    Each flag code represents a specific reason why a frame might be flagged.
    This enum can be extended with additional flag codes as needed.

    Attributes:
        LOW_CONFIDENCE: Frame contains instances with low confidence scores
            (below confidence threshold).
        HIGH_ENTROPY: Frame contains instances with high entropy in association
            scores, indicating uncertain tracking assignments.
        MISSING_DETECTIONS: Frame has no detected instances when some were expected.
        TRACKING_FAILURE: Frame failed to be assigned to any trajectory.
    """

    LOW_CONFIDENCE = "low_confidence"

    def __str__(self) -> str:
        """Return the string value of the flag code."""
        return self.value

    def __repr__(self) -> str:
        """Return the representation of the flag code."""
        return f"FrameFlagCode.{self.name}"
