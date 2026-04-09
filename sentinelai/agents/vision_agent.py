"""VisionAgent — YOLO person detection + ByteTrack tracking + pose estimation."""

from __future__ import annotations

import math
from collections import deque
from itertools import combinations

import numpy as np

from sentinelai.core.config import (
    CLOSE_DISTANCE_PX,
    CONFIDENCE_THRESHOLD,
    FALLEN_ASPECT_RATIO,
    FAST_APPROACH_SPEED,
    OPPOSITION_THRESHOLD,
    PERSON_CLASS_ID,
    SCORE_CLOSE_APPROACH,
    SCORE_FALLEN,
    SCORE_OPPOSITION,
    STATIONARY_SPEED_THRESHOLD,
    VELOCITY_HISTORY_MAXLEN,
    YOLO_FALLBACK_MODEL,
    YOLO_POSE_MODEL,
)
from sentinelai.core.models import PairAnalysis, PersonTrack, SceneData


class VisionAgent:
    """Runs YOLO detection/pose + ByteTrack on each frame."""

    def __init__(self, fps: float = 30.0) -> None:
        self.fps = fps
        self.has_pose = False
        self._load_model()

        # Tracking state
        self.prev_centers: dict[int, tuple[float, float]] = {}
        self.prev_distances: dict[tuple[int, int], float] = {}
        self.velocity_history: dict[int, deque] = {}
        self.stationary_frames: dict[int, int] = {}

    def _load_model(self) -> None:
        from ultralytics import YOLO

        try:
            self.model = YOLO(YOLO_POSE_MODEL)
            self.has_pose = True
        except Exception:
            try:
                self.model = YOLO(YOLO_FALLBACK_MODEL)
                self.has_pose = False
            except Exception:
                self.model = YOLO("yolov8n-pose.pt")
                self.has_pose = True

    def process_frame(self, frame: np.ndarray, frame_number: int) -> SceneData:
        """Process a single video frame and return structured scene data."""
        timestamp = frame_number / self.fps if self.fps > 0 else 0.0

        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONFIDENCE_THRESHOLD,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )

        persons: list[PersonTrack] = []
        current_centers: dict[int, tuple[float, float]] = {}

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            ids = boxes.id
            if ids is None:
                return SceneData(
                    frame_number=frame_number,
                    timestamp=timestamp,
                    person_count=0,
                )

            track_ids = ids.int().cpu().tolist()
            xyxy = boxes.xyxy.cpu().numpy()

            keypoints_data = None
            if self.has_pose and results[0].keypoints is not None:
                keypoints_data = results[0].keypoints.data.cpu().numpy()

            for i, track_id in enumerate(track_ids):
                x1, y1, x2, y2 = map(int, xyxy[i])
                w = x2 - x1
                h = y2 - y1
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                center = (cx, cy)
                current_centers[track_id] = center

                # Velocity
                vx, vy = 0.0, 0.0
                if track_id in self.prev_centers:
                    px, py = self.prev_centers[track_id]
                    vx = cx - px
                    vy = cy - py

                speed = math.sqrt(vx * vx + vy * vy)

                # Velocity history (smoothing)
                if track_id not in self.velocity_history:
                    self.velocity_history[track_id] = deque(
                        maxlen=VELOCITY_HISTORY_MAXLEN
                    )
                self.velocity_history[track_id].append((vx, vy))

                # Aspect ratio: width / height
                aspect_ratio = w / h if h > 0 else 1.0

                # Keypoints
                kpts = None
                if keypoints_data is not None and i < len(keypoints_data):
                    raw = keypoints_data[i]
                    kpts = [tuple(kp) for kp in raw.tolist()]

                persons.append(
                    PersonTrack(
                        id=track_id,
                        bbox=(x1, y1, x2, y2),
                        center=center,
                        velocity=(vx, vy),
                        speed=speed,
                        bbox_aspect_ratio=aspect_ratio,
                        keypoints=kpts,
                    )
                )

                # Stationary tracking
                if speed < STATIONARY_SPEED_THRESHOLD:
                    self.stationary_frames[track_id] = (
                        self.stationary_frames.get(track_id, 0) + 1
                    )
                else:
                    self.stationary_frames[track_id] = 0

        # Pair analysis
        pairs: list[PairAnalysis] = []
        for p_a, p_b in combinations(persons, 2):
            dist = math.dist(p_a.center, p_b.center)
            pair_key = (min(p_a.id, p_b.id), max(p_a.id, p_b.id))

            approach_speed = 0.0
            if pair_key in self.prev_distances:
                approach_speed = self.prev_distances[pair_key] - dist
            self.prev_distances[pair_key] = dist

            vel_opp = self._velocity_opposition(p_a.velocity, p_b.velocity)

            pairs.append(
                PairAnalysis(
                    id_a=p_a.id,
                    id_b=p_b.id,
                    distance=round(dist, 1),
                    approach_speed=round(approach_speed, 2),
                    velocity_opposition=round(vel_opp, 3),
                )
            )

        # Long stationary persons
        stationary_threshold_frames = int(self.fps * 5)
        long_stationary: list[dict] = []
        for p in persons:
            frames_still = self.stationary_frames.get(p.id, 0)
            if frames_still >= stationary_threshold_frames:
                long_stationary.append(
                    {
                        "id": p.id,
                        "center": p.center,
                        "stationary_seconds": round(frames_still / self.fps, 1),
                    }
                )

        # Suspicious score
        suspicious_score = self._compute_suspicious_score(persons, pairs)

        # Update previous centers
        self.prev_centers = current_centers

        return SceneData(
            frame_number=frame_number,
            timestamp=round(timestamp, 2),
            person_count=len(persons),
            persons=persons,
            pairs=pairs,
            long_stationary=long_stationary,
            suspicious_score=round(suspicious_score, 3),
        )

    @staticmethod
    def _velocity_opposition(
        v1: tuple[float, float], v2: tuple[float, float]
    ) -> float:
        """Dot product of normalized velocity vectors. Negative = moving toward each other."""
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
        if mag1 < 1e-6 or mag2 < 1e-6:
            return 0.0
        n1 = (v1[0] / mag1, v1[1] / mag1)
        n2 = (v2[0] / mag2, v2[1] / mag2)
        return n1[0] * n2[0] + n1[1] * n2[1]

    @staticmethod
    def _compute_suspicious_score(
        persons: list[PersonTrack], pairs: list[PairAnalysis]
    ) -> float:
        score = 0.0
        for pair in pairs:
            if pair.distance < CLOSE_DISTANCE_PX and pair.approach_speed > FAST_APPROACH_SPEED:
                score += SCORE_CLOSE_APPROACH
            if pair.velocity_opposition < OPPOSITION_THRESHOLD:
                score += SCORE_OPPOSITION
        for person in persons:
            if person.bbox_aspect_ratio > FALLEN_ASPECT_RATIO:
                score += SCORE_FALLEN
        return min(score, 1.0)
