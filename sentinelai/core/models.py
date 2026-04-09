"""Pydantic data models for SentinelAI."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PersonTrack(BaseModel):
    id: int
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    center: tuple[float, float]
    velocity: tuple[float, float] = (0.0, 0.0)  # vx, vy pixels/frame
    speed: float = 0.0
    bbox_aspect_ratio: float = 1.0  # width / height (high = fallen person)
    keypoints: list[tuple] | None = None


class PairAnalysis(BaseModel):
    id_a: int
    id_b: int
    distance: float  # pixels
    approach_speed: float = 0.0  # negative = moving apart
    velocity_opposition: float = 0.0  # dot product of normalized velocities


class SceneData(BaseModel):
    frame_number: int
    timestamp: float
    person_count: int
    persons: list[PersonTrack] = Field(default_factory=list)
    pairs: list[PairAnalysis] = Field(default_factory=list)
    long_stationary: list[dict] = Field(default_factory=list)
    suspicious_score: float = 0.0  # 0.0 to 1.0


class DetectionResult(BaseModel):
    frame_number: int
    timestamp: float
    crime_detected: bool = False
    crime_type: str | None = None  # FIGHT, MURDER, THEFT, ASSAULT, SUSPICIOUS, OTHER
    severity: str | None = None  # LOW, MEDIUM, HIGH, CRITICAL
    description: str | None = None  # Turkish explanation
    recommendation: str | None = None  # Turkish action to take
    auto_alert_sent: bool = False
