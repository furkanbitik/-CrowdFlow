# SentinelAI 🎯
Person of Interest style crime detection system.
Analyzes uploaded video files using YOLO26 + GPT-5.4 Thinking.

## Setup
1. Add your OpenAI API key to `.env`: `OPENAI_API_KEY=your_key`
2. `pip install -r sentinelai/requirements.txt`
3. `streamlit run sentinelai/main.py`

## Features
- Upload any video file (mp4, avi, mov, mkv)
- Enter what to look for in Turkish or English
- Real-time crime detection with GPT-5.4 Thinking
- Automatic alerts for violence and murder detections
- No model training required
