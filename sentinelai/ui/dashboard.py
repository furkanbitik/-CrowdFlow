"""Streamlit UI components for SentinelAI dashboard."""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

from sentinelai.core.models import DetectionResult, PersonTrack, SceneData


def draw_annotations(
    frame: np.ndarray,
    scene: SceneData,
    result: DetectionResult | None = None,
) -> np.ndarray:
    """Draw bounding boxes, labels, and status bar on a video frame."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Determine which person IDs are involved in a detection
    involved_ids: set[int] = set()
    if result and result.crime_detected:
        for pair in scene.pairs:
            involved_ids.add(pair.id_a)
            involved_ids.add(pair.id_b)

    # Draw bounding boxes for each person
    for person in scene.persons:
        x1, y1, x2, y2 = person.bbox
        is_involved = person.id in involved_ids and result and result.crime_detected
        color = (0, 0, 255) if is_involved else (0, 255, 0)
        thickness = 3 if is_involved else 2

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        label = f"ID:{person.id}"
        if person.speed > 1:
            label += f" v:{person.speed:.0f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(
            annotated,
            (x1, y1 - label_size[1] - 8),
            (x1 + label_size[0] + 4, y1),
            color,
            -1,
        )
        cv2.putText(
            annotated, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

    # Crime overlay: red border on entire frame
    if result and result.crime_detected:
        cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
        crime_label = f"DETECTED: {result.crime_type} [{result.severity}]"
        cv2.putText(
            annotated, crime_label, (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
        )

    # Bottom status bar
    bar_h = 32
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

    bar_text = (
        f"Kare: {scene.frame_number} | "
        f"Kisi: {scene.person_count} | "
        f"Suphe: {scene.suspicious_score:.0%}"
    )
    cv2.putText(
        annotated, bar_text, (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
    )

    return annotated


def render_sidebar(
    frame_count: int,
    llm_calls: int,
    detections: int,
    alerts: int,
) -> tuple[str, float, object, bool]:
    """Render the sidebar and return (user_prompt, sensitivity, uploaded_file, start)."""
    with st.sidebar:
        st.title("\U0001f3af SentinelAI")
        st.caption("Person of Interest tarzi suc tespit sistemi")

        st.divider()
        st.subheader("Tespit Ayarlari")

        user_prompt = st.text_area(
            "\U0001f50d Ne aramak istiyorsunuz?",
            value="Kavga, dovus ve fiziksel saldirilari tespit et",
            height=100,
            help="Turkce veya Ingilizce yazabilirsiniz",
        )

        sensitivity = st.slider(
            "Hassasiyet",
            min_value=0.1,
            max_value=1.0,
            value=0.45,
            step=0.05,
            help="Dusuk = daha fazla uyari, Yuksek = sadece kesin tespitler",
        )

        st.divider()
        uploaded_file = st.file_uploader(
            "\U0001f4f9 Video Yukle",
            type=["mp4", "avi", "mov", "mkv"],
        )

        start = st.button("\u25b6 Analizi Baslat", type="primary")

        st.divider()
        st.subheader("\U0001f4ca Istatistikler")
        col1, col2 = st.columns(2)
        col1.metric("Toplam Kare", frame_count)
        col2.metric("LLM Sorgusu", llm_calls)
        col3, col4 = st.columns(2)
        col3.metric("Tespit Sayisi", detections)
        col4.metric("Otomatik Bildirim", alerts)

    return user_prompt, sensitivity, uploaded_file, start


def render_detection_card(result: DetectionResult) -> None:
    """Render a single detection result as a colored card."""
    severity = result.severity or "LOW"
    colors = {
        "CRITICAL": "#ff4444",
        "HIGH": "#ff6b6b",
        "MEDIUM": "#ff9f43",
        "LOW": "#ffd93d",
    }
    bg = colors.get(severity, "#ffd93d")
    text_color = "#ffffff" if severity in ("CRITICAL", "HIGH") else "#333333"

    alert_badge = ""
    if result.auto_alert_sent:
        alert_badge = "<br><b>\u26a1 Otomatik bildirim gonderildi</b>"

    st.markdown(
        f"""
        <div style="
            background-color: {bg};
            color: {text_color};
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 10px;
        ">
            <b>\U0001f6a8 {result.crime_type} &mdash; {severity}</b><br>
            Kare: {result.frame_number} | Zaman: {result.timestamp:.1f}s<br>
            {result.description or ''}<br>
            <i>Oneri: {result.recommendation or 'N/A'}</i>
            {alert_badge}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_detections_panel(
    detections: list[DetectionResult],
    last_raw_response: str | None,
) -> None:
    """Render the right-side detections panel."""
    st.subheader("\U0001f6a8 Tespit Loglari")

    if not detections:
        st.info("Henuz tespit yok. Video analizi baslatildiginda sonuclar burada gorunecek.")
    else:
        for det in reversed(detections):
            render_detection_card(det)

    st.divider()
    st.subheader("\U0001f9e0 Son GPT-5.4 Yaniti")
    if last_raw_response:
        st.code(last_raw_response, language="json")
    else:
        st.caption("Henuz LLM yaniti yok.")
