"""ReasoningAgent — GPT-5.4 Thinking integration for crime analysis."""

from __future__ import annotations

import json
import logging
import os
import time

from openai import OpenAI

from sentinelai.core.config import AUTO_ALERT_SEVERITIES, LLM_COOLDOWN_SECONDS, OPENAI_MODEL
from sentinelai.core.models import DetectionResult, SceneData

logger = logging.getLogger(__name__)


class ReasoningAgent:
    """Uses GPT-5.4 Thinking to classify detected scenes as criminal activity."""

    def __init__(self, user_prompt: str) -> None:
        self.user_prompt = user_prompt
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = OPENAI_MODEL
        self.call_count = 0
        self.total_tokens = 0
        self.last_result: DetectionResult | None = None
        self._last_call_time: float = 0.0

    def analyze(self, scene: SceneData, scene_summary: str) -> DetectionResult:
        """Send scene data to GPT-5.4 and return a DetectionResult."""
        # Cooldown enforcement
        elapsed = time.time() - self._last_call_time
        if elapsed < LLM_COOLDOWN_SECONDS:
            time.sleep(LLM_COOLDOWN_SECONDS - elapsed)

        system_prompt = f"""You are SentinelAI, an advanced security camera AI analyst.
The user wants you to watch for: {self.user_prompt}

You receive structured scene data from each video frame.
Your job is to determine if criminal activity matching the user's request is occurring.

Crime types you can detect:
- FIGHT: Two or more people in physical confrontation
- MURDER: Person lying motionless, others fleeing or standing over
- THEFT: Person grabbing object from another person
- ASSAULT: One-sided physical attack
- SUSPICIOUS: Unusual loitering or stalking behavior
- OTHER: Any other criminal activity the user requested

Respond ONLY in this exact JSON format, nothing else:
{{
  "crime_detected": true or false,
  "crime_type": "FIGHT" | "MURDER" | "THEFT" | "ASSAULT" | "SUSPICIOUS" | "OTHER" | null,
  "severity": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL" | null,
  "description": "Turkish: 2-sentence explanation of what is happening",
  "recommendation": "Turkish: specific action for security team"
}}

Severity guide:
- CRITICAL: Murder, severe assault, active weapon threat
- HIGH: Active fight, theft in progress
- MEDIUM: Suspicious behavior, potential threat
- LOW: Minor anomaly, worth monitoring"""

        user_message = f"Analyze this scene:\n{scene_summary}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
            )

            self._last_call_time = time.time()
            self.call_count += 1

            if response.usage:
                self.total_tokens += response.usage.total_tokens

            raw = response.choices[0].message.content or "{}"
            self.last_raw_response = raw
            data = json.loads(raw)

            crime_detected = bool(data.get("crime_detected", False))
            severity = data.get("severity")
            auto_alert = (
                crime_detected and severity in AUTO_ALERT_SEVERITIES
            )

            result = DetectionResult(
                frame_number=scene.frame_number,
                timestamp=scene.timestamp,
                crime_detected=crime_detected,
                crime_type=data.get("crime_type") if crime_detected else None,
                severity=severity if crime_detected else None,
                description=data.get("description"),
                recommendation=data.get("recommendation"),
                auto_alert_sent=auto_alert,
            )

        except Exception as e:
            logger.error("OpenAI API error: %s", e)
            self.last_raw_response = f"ERROR: {e}"
            result = DetectionResult(
                frame_number=scene.frame_number,
                timestamp=scene.timestamp,
                crime_detected=False,
                description=f"LLM hatası: {e}",
            )

        self.last_result = result
        return result
