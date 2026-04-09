"""AlertSystem — Automatic alert dispatch for high-severity detections."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from sentinelai.core.config import ALERTS_DIR, ALERT_LOG_FILE, AUTO_ALERT_SEVERITIES
from sentinelai.core.models import DetectionResult


class AlertSystem:
    """Saves alert frames, prints console alerts, and logs events."""

    def __init__(self) -> None:
        self.alerts_dir = Path(ALERTS_DIR)
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        self.alert_count = 0

    def send_alert(self, result: DetectionResult, frame: np.ndarray) -> None:
        """Process an alert for CRITICAL/HIGH severity detections."""
        if result.severity not in AUTO_ALERT_SEVERITIES:
            return

        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        crime_type = result.crime_type or "UNKNOWN"

        # 1. Save alert frame with red border
        alert_frame = self._draw_alert_frame(frame.copy(), result)
        filename = f"alert_{timestamp_str}_{crime_type}.jpg"
        filepath = self.alerts_dir / filename
        cv2.imwrite(str(filepath), alert_frame)

        # 2. Console alert
        self._print_console_alert(result, now)

        # 3. Log to file
        self._log_alert(result, now, filename)

        result.auto_alert_sent = True
        self.alert_count += 1

    @staticmethod
    def _draw_alert_frame(frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw a red border and crime info on the alert frame."""
        h, w = frame.shape[:2]
        border = 8
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), border)
        label = f"ALERT: {result.crime_type} - {result.severity}"
        cv2.putText(
            frame, label, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
        )
        return frame

    @staticmethod
    def _print_console_alert(result: DetectionResult, now: datetime) -> None:
        """Print a formatted alert to the console with ANSI colors."""
        red = "\033[91m"
        bold = "\033[1m"
        reset = "\033[0m"
        time_str = now.strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n{red}{bold}")
        print("\u2554" + "\u2550" * 50 + "\u2557")
        print(f"\u2551  \U0001f6a8 SENTINELAI OTOMATIK BILDIRI \U0001f6a8{' ' * 13}\u2551")
        print(f"\u2551  Zaman: {time_str}{' ' * (30 - len(time_str))}\u2551")
        print(f"\u2551  Su\u00e7 Tipi: {result.crime_type or 'N/A'}{' ' * (28 - len(result.crime_type or 'N/A'))}\u2551")
        print(f"\u2551  Risk: {result.severity or 'N/A'}{' ' * (31 - len(result.severity or 'N/A'))}\u2551")
        if result.description:
            desc = result.description[:46]
            print(f"\u2551  {desc}{' ' * (48 - len(desc))}\u2551")
        if result.recommendation:
            rec = result.recommendation[:42]
            print(f"\u2551  \u00d6neri: {rec}{' ' * (42 - len(rec))}\u2551")
        print("\u255a" + "\u2550" * 50 + "\u255d")
        print(f"{reset}")

    def _log_alert(
        self, result: DetectionResult, now: datetime, filename: str
    ) -> None:
        """Append alert details to the log file."""
        log_path = self.alerts_dir / ALERT_LOG_FILE
        time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        entry = (
            f"[{time_str}] "
            f"Frame={result.frame_number} | "
            f"Type={result.crime_type} | "
            f"Severity={result.severity} | "
            f"File={filename} | "
            f"Description={result.description}\n"
        )
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(entry)
