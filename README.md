# Automated Exam Proctoring System

## Overview
This solution enables automated detection of student attributes (gaze direction, head position) and suspicious activities (phone usage, identity changes) during exams, replacing manual proctoring efforts. The system transforms raw videos automatically into concise reports using LLMs. By processing key frames and using efficient models, the solution can monitor large videos in acceptable time. The modular design allows seamless integration of new detection models.

## Key Features
- 🎭 Multi-model inference pipeline (gaze, head pose, identity, phone, persons)
- ⚡ Optimized frame processing with configurable skipping
- 📊 Hierarchical results storage with JSON metadata
- 🤖 LLM-powered report generation for behavioral analysis
- 🐳 Docker container support for easy deployment
- 📈 Frame comparison tool for change detection

## Project Structure
```
CV_INFERENCE_PROJECT/
├── creds/                     # Authentication credentials
│   ├── credentials.json
│   └── token.json
│
├── models/                    # Detection models
│   ├── gaze.py                # Gaze direction detection (MediaPipe)
│   ├── headpose.py            # Head position estimation (MediaPipe + PnP)
│   ├── identity.py            # Student identification (InsightFace)
│   ├── persons.py             # Person counting (YOLOv8)
│   └── phone.py               # Phone detection (YOLOv8)
│
├── out/                       # Processing outputs
│   └── {student_id}/
│       └── all_results.json   # Aggregated results
│
├── parser/                    # Analysis utilities
│   └── compare_frames.py      # Frame comparison tool
│
├── downloads/                 # Raw video storage
│   └── {student_id}/
│       ├── *.mkv              # Exam recordings
│       └── *.MOV
│
├── Dockerfile                 # Containerization
├── run_inference.py           # Main processing script
└── send_to_llm.py             # LLM report generation
```

## Installation
```bash
# Clone repository
git clone https://github.com/yourusername/cv_inference_project.git
cd cv_inference_project

# Install dependencies
pip install -r requirements.txt

# Initialize models (will download weights)
python -c "from models.identity import load_model; load_model()"
```

## Usage

### 1. Process Videos
```bash
python run_inference.py \
  --dataset-root downloads/ \
  --output-dir out/ \
  --models identity gaze headpose phone persons \
  --frame-skip 5 \
  --log-level INFO
```

### 2. Compare Frames
```bash
python parser/compare_frames.py \
  --summary out/student123/session_summary.json \
  --frame1 100 \
  --frame2 500 \
  --output comparison.json
```

### 3. Generate LLM Report
```bash
python send_to_llm.py \
  --input comparison.json \
  --output report.md
```

## Detection Models

| Model       | Algorithm              | Detection                           | Output Parameters              |
|-------------|------------------------|-------------------------------------|--------------------------------|
| **Gaze**    | MediaPipe Face Mesh    | Eye gaze direction                 | `gaze_away`, `gaze_angle`     |
| **HeadPose**| MediaPipe + PnP        | Head orientation                   | `yaw`, `pitch`, `roll`        |
| **Identity**| InsightFace            | Student identification             | `is_match`, `distance`        |
| **Phone**   | YOLOv8n                | Phone usage detection              | `phone_count`                 |
| **Persons** | YOLOv8n                | Person count in frame              | `person_count`                |

## Output Structure
Results are stored in hierarchical JSON format:
```json
{
  "gaze": [
    {
      "frame": 120,
      "timestamp": 4.32,
      "meta": {
        "gaze_away": true,
        "gaze_angle": 37.2
      }
    }
  ],
  "identity": [
    {
      "frame": 120,
      "timestamp": 4.32,
      "meta": {
        "is_match": false,
        "distance": 1.24
      }
    }
  ]
}
```

## LLM Report Generation
The `send_to_llm.py` script:
1. Loads frame comparison data
2. Generates technical prompt with:
   - Aggregate statistics (average/max deviations)
   - Frame-by-frame change analysis
   - Behavioral pattern summaries
3. Sends request to DeepSeek-R1 via OpenRouter API
4. Formats results into markdown report

Sample report sections:
```
## Technical Conclusions
- Significant head pose variations detected between frames 230-310 (avg yaw Δ=12.4°)
- 3 identity match changes indicate possible substitution attempts
- Phone appeared at 08:32 and remained visible for 47 seconds
```

## Docker
```bash
# Build image
docker build -t exam-proctor .

# Run processing
docker run -v $(pwd)/downloads:/app/downloads \
           -v $(pwd)/out:/app/out \
           exam-proctor \
           --dataset-root downloads \
           --output-dir out
```

## Performance Metrics
```markdown
| Video Length | Frame Skip | Processing Time | Hardware        |
|--------------|------------|-----------------|-----------------|  |
| 60 minutes   | 10         | ~18 minutes     | GPU (RTX 3080)  |
```

## License
MIT License - See [LICENSE](LICENSE) for details

