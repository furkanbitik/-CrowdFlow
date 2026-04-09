"""SentinelAI — Streamlit entry point for crime detection system."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from dotenv import load_dotenv

from sentinelai.agents.reasoning_agent import ReasoningAgent
from sentinelai.agents.scene_agent import SceneAgent
from sentinelai.agents.vision_agent import VisionAgent
from sentinelai.core.alert import AlertSystem
from sentinelai.core.config import MAX_PROCESSING_FPS, PAGE_ICON, PAGE_LAYOUT, PAGE_TITLE
from sentinelai.core.models import DetectionResult
from sentinelai.ui.dashboard import (
    draw_annotations,
    render_detections_panel,
    render_sidebar,
)

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=PAGE_LAYOUT)

# --- Session state defaults ---
for key, default in {
    "frame_count": 0,
    "llm_calls": 0,
    "detection_count": 0,
    "alert_count": 0,
    "detections": [],
    "last_raw_response": None,
    "processing": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def main() -> None:
    user_prompt, sensitivity, uploaded_file, start = render_sidebar(
        frame_count=st.session_state.frame_count,
        llm_calls=st.session_state.llm_calls,
        detections=st.session_state.detection_count,
        alerts=st.session_state.alert_count,
    )

    # Main area — two columns
    col_video, col_detections = st.columns([3, 2])

    with col_video:
        st.subheader("\U0001f4f9 Video Analizi")
        video_placeholder = st.empty()

        if not uploaded_file:
            video_placeholder.info(
                "Lutfen sol panelden bir video dosyasi yukleyin ve 'Analizi Baslat' butonuna tiklayin."
            )

    with col_detections:
        detection_placeholder = st.empty()

    if not (start and uploaded_file):
        with col_detections:
            render_detections_panel(
                st.session_state.detections,
                st.session_state.last_raw_response,
            )
        return

    # --- Save uploaded video to temp file ---
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        st.error("Video dosyasi acilamadi. Lutfen gecerli bir video yukleyin.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = max(1, int(fps / MAX_PROCESSING_FPS))

    # --- Initialize agents ---
    vision_agent = VisionAgent(fps=fps)
    scene_agent = SceneAgent(threshold=sensitivity)
    reasoning_agent = ReasoningAgent(user_prompt=user_prompt)
    alert_system = AlertSystem()

    # Reset counters
    st.session_state.frame_count = 0
    st.session_state.llm_calls = 0
    st.session_state.detection_count = 0
    st.session_state.alert_count = 0
    st.session_state.detections = []
    st.session_state.last_raw_response = None

    progress_bar = col_video.progress(0, text="Analiz baslatiliyor...")

    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1

        # Skip frames to stay within MAX_PROCESSING_FPS
        if frame_no % frame_skip != 0:
            continue

        st.session_state.frame_count = frame_no

        # --- Vision processing ---
        scene = vision_agent.process_frame(frame, frame_no)

        if scene.person_count == 0:
            annotated = draw_annotations(frame, scene)
            video_placeholder.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
            )
            progress = min(frame_no / max(total_frames, 1), 1.0)
            progress_bar.progress(progress, text=f"Kare {frame_no}/{total_frames}")
            continue

        # --- Scene analysis ---
        current_result: DetectionResult | None = None
        if scene_agent.should_analyze(scene):
            summary = scene_agent.summarize(scene)
            current_result = reasoning_agent.analyze(scene, summary)
            st.session_state.llm_calls = reasoning_agent.call_count
            st.session_state.last_raw_response = getattr(
                reasoning_agent, "last_raw_response", None
            )

            if current_result.crime_detected:
                st.session_state.detection_count += 1
                st.session_state.detections.append(current_result)

                alert_system.send_alert(current_result, frame)
                st.session_state.alert_count = alert_system.alert_count

        # --- Draw annotated frame ---
        annotated = draw_annotations(frame, scene, current_result)
        video_placeholder.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True,
        )

        # --- Update detections panel ---
        with detection_placeholder.container():
            render_detections_panel(
                st.session_state.detections,
                st.session_state.last_raw_response,
            )

        # --- Progress ---
        progress = min(frame_no / max(total_frames, 1), 1.0)
        progress_bar.progress(progress, text=f"Kare {frame_no}/{total_frames}")

    cap.release()
    progress_bar.progress(1.0, text="Analiz tamamlandi!")

    # Final stats
    col_video.success(
        f"Analiz tamamlandi! {st.session_state.frame_count} kare islendi, "
        f"{st.session_state.detection_count} tespit yapildi, "
        f"{st.session_state.llm_calls} LLM sorgusu gonderildi."
    )

    # Final detections panel update
    with detection_placeholder.container():
        render_detections_panel(
            st.session_state.detections,
            st.session_state.last_raw_response,
        )


if __name__ == "__main__":
    main()
