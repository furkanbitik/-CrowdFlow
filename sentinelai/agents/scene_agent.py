"""SceneAgent — Motion analysis and rule engine for triggering LLM analysis."""

from __future__ import annotations

from sentinelai.core.config import DEFAULT_SENSITIVITY, FALLEN_ASPECT_RATIO
from sentinelai.core.models import SceneData


class SceneAgent:
    """Decides when to call the LLM and builds compact summaries."""

    def __init__(self, threshold: float = DEFAULT_SENSITIVITY) -> None:
        self.threshold = threshold

    def should_analyze(self, scene: SceneData) -> bool:
        """Return True if this scene warrants LLM analysis."""
        if scene.suspicious_score >= self.threshold:
            return True
        for person in scene.persons:
            if person.bbox_aspect_ratio > FALLEN_ASPECT_RATIO:
                return True
        return False

    def summarize(self, scene: SceneData) -> str:
        """Build a compact English summary of the scene for the LLM."""
        lines: list[str] = []
        lines.append(
            f"Frame {scene.frame_number} | {scene.person_count} persons detected. "
            f"Suspicious score: {scene.suspicious_score:.2f}"
        )

        for pair in scene.pairs:
            tags: list[str] = []
            if pair.approach_speed > 8:
                tags.append("CLOSING IN FAST")
            if pair.velocity_opposition < -0.5:
                tags.append("OPPOSING MOTION")
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            lines.append(
                f"Pair (ID{pair.id_a}, ID{pair.id_b}): "
                f"distance={pair.distance:.0f}px, "
                f"approach_speed={pair.approach_speed:.1f}px/frame, "
                f"opposition_score={pair.velocity_opposition:.2f}{tag_str}."
            )

        for person in scene.persons:
            flags: list[str] = []
            if person.bbox_aspect_ratio > FALLEN_ASPECT_RATIO:
                flags.append("FALLEN/LYING DOWN")
            if person.speed > 15:
                flags.append("FAST MOVING")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            lines.append(
                f"Person ID{person.id}: "
                f"speed={person.speed:.1f}px/frame, "
                f"aspect_ratio={person.bbox_aspect_ratio:.2f}{flag_str}."
            )

        for entry in scene.long_stationary:
            lines.append(
                f"Person ID{entry['id']}: stationary for "
                f"{entry['stationary_seconds']:.1f} seconds "
                f"at position ({entry['center'][0]:.0f}, {entry['center'][1]:.0f})."
            )

        return "\n".join(lines)
