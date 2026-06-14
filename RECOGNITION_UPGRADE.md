# Drishti Face Recognition Pipeline Upgrade Plan

## Purpose

This document defines the model and architecture upgrade plan for the Drishti project.  
The current pipeline uses:

```text
OpenCV → MediaPipe Face Detection → FaceNet Embeddings → Cosine Similarity
```

The current system has major false-positive problems:

- wrong person matched as target
- multiple people in frame causing wrong identity match
- clothes, wall, background, or non-face crops getting matched
- very high confidence score even for incorrect matches
- single-frame decisions producing unstable results

The new goal is not only to improve recognition accuracy, but also to make the system capable of refusing uncertain matches.

The upgraded pipeline must follow this principle:

```text
Detect only valid faces → align face → reject poor-quality crops → generate strong embeddings → compare safely → confirm over multiple frames/tracks → return match or no reliable match
```

---

# Final Recommended Stack

## Best Practical Stack for Drishti

```text
OpenCV / RTSP Input
        ↓
Frame Sampler
        ↓
SCRFD / RetinaFace Face Detector
        ↓
Landmark-Based Face Alignment
        ↓
Face Quality Filter
        ↓
AdaFace / InsightFace ArcFace Embedding Model
        ↓
Similarity Matching
        ↓
Track-Level Aggregation
        ↓
Open-Set Decision: Match / No Reliable Match
        ↓
Frontend Result Display
```

---

# Why the Current Model Fails

## 1. MediaPipe is not enough for CCTV-style recognition

MediaPipe is lightweight and fast, but it is not ideal for crowded CCTV-like face search. It can work for simple face detection, but in low light, motion blur, side pose, or crowded scenes, it may produce weak or unstable crops.

If a bad crop reaches FaceNet, the embedding model will still generate a vector. It does not automatically know that the crop is useless.

So this can happen:

```text
Wall / Shirt / Blurry Face → FaceNet → Random Embedding → High Similarity Accidentally
```

That is why non-face objects may appear as high-confidence matches.

## 2. FaceNet is outdated for this use case

FaceNet was a strong classic model, but for current CCTV-style open-set search, better options exist.

Better modern alternatives:

- ArcFace
- AdaFace
- MagFace
- CurricularFace
- InsightFace model packs

For Drishti, FaceNet should be replaced.

## 3. Single-frame matching is unsafe

Current logic probably does this:

```text
One frame → detect face → embedding → compare → show result
```

This is risky. One bad frame can produce a false match.

New logic should be:

```text
Multiple frames of same person → multiple embeddings → average/majority decision → final match
```

This is much safer.

## 4. Confidence score is misleading

Cosine similarity is not a true “confidence percentage.”

Bad:

```text
similarity = 0.82
display = 82% confidence
```

This is misleading because similarity depends on model, threshold, image quality, lighting, crop quality, and dataset.

Better UI wording:

```text
Strong Match
Possible Match
No Reliable Match
```

Or:

```text
Similarity Score: 0.82
Decision: Strong Match
```

Never treat cosine similarity directly as human-readable confidence.

---

# Recommended Model Choices

## Option A: Best Fast Integration

Use:

```text
InsightFace buffalo_l / buffalo_m
```

This is the easiest strong upgrade.

### Why

InsightFace gives an end-to-end face analysis pipeline:

- face detection
- facial landmarks
- alignment
- face embedding
- ONNX Runtime support
- strong ArcFace-family recognition

### Recommended for

- fast project upgrade
- hackathon/demo
- fewer integration bugs
- stable Python implementation

### Suggested package

```bash
pip install insightface onnxruntime opencv-python numpy
```

For GPU:

```bash
pip install onnxruntime-gpu
```

### Expected benefit

This should immediately reduce:

- weak face crops
- poor landmark alignment
- FaceNet-based wrong matches
- unstable results

---

## Option B: Best Accuracy-Oriented Stack

Use:

```text
SCRFD Detector + AdaFace R100 Recognizer + Quality Filter
```

### Why

AdaFace is especially useful for low-quality face images, which is closer to CCTV footage.

SCRFD is a strong face detector with good speed/accuracy tradeoff.

### Recommended for

- best model quality
- CCTV-like search
- crowded scene matching
- final polished version

### Downside

More integration effort than InsightFace buffalo_l.

---

# Detector Recommendation

## Replace MediaPipe with SCRFD or RetinaFace

### Preferred detector

```text
SCRFD-10GF
```

Use this for live demo because it is fast and accurate.

### Higher accuracy option

```text
SCRFD-34GF
```

Use this if GPU performance is good.

### Easy packaged option

```text
RetinaFace through InsightFace buffalo_l
```

---

# Recognition Model Recommendation

## Replace FaceNet

### Best research-backed choice

```text
AdaFace R100
```

### Best plug-and-play choice

```text
InsightFace buffalo_l / buffalo_m recognizer
```

### Avoid for final version

```text
FaceNet
```

FaceNet can be kept only as fallback or comparison demo.

---

# Required Pipeline Improvements

## 1. Face Detection Gate

Before embedding generation, verify that the crop is actually a valid face.

Reject crop if:

```text
detection_score < 0.70
face_width < 50 px
face_height < 50 px
landmarks missing
face too blurry
face too dark
face too side-posed
```

Recommended first thresholds:

```python
MIN_DETECTION_SCORE = 0.70
MIN_FACE_SIZE = 50
```

For high precision demo:

```python
MIN_DETECTION_SCORE = 0.80
MIN_FACE_SIZE = 60
```

---

## 2. Face Alignment

Do not send raw face crop directly to the recognizer.

Use 5-point facial landmarks:

```text
left eye
right eye
nose
left mouth corner
right mouth corner
```

Then align the face to standard input size:

```text
112x112
```

Most ArcFace/AdaFace models expect aligned 112x112 face images.

Correct flow:

```text
Frame → Detector → Landmarks → Aligned Face → Embedding Model
```

Wrong flow:

```text
Frame → bounding box crop → Embedding Model
```

---

## 3. Face Quality Filtering

Add a quality gate before recognition.

Reject poor face crops if:

```text
blur score too low
brightness too low/high
face too small
landmark confidence weak
face angle too extreme
occlusion too high
```

Simple initial quality checks:

```python
MIN_BLUR_SCORE = 80.0
MIN_BRIGHTNESS = 40
MAX_BRIGHTNESS = 220
```

Blur can be estimated using Laplacian variance:

```python
blur_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()
```

If:

```python
blur_score < MIN_BLUR_SCORE
```

then reject the crop.

---

## 4. Open-Set Recognition

The system should not always return the closest person.

Current bad behavior:

```text
Always return top match
```

Correct behavior:

```text
Return match only if similarity passes strict threshold.
Otherwise return "No reliable match".
```

This is very important.

### First threshold suggestion

For cosine similarity with normalized embeddings:

```python
MATCH_THRESHOLD = 0.45  # start point for InsightFace-like embeddings
```

But this must be calibrated on your own demo environment.

Possible threshold zones:

```text
similarity >= 0.50 → Strong Match
0.42 to 0.50       → Possible Match
< 0.42             → No Reliable Match
```

For high precision, increase threshold.

For fewer false positives:

```python
MATCH_THRESHOLD = 0.50
```

For more recall:

```python
MATCH_THRESHOLD = 0.40
```

Use high precision for presentation/demo.

---

## 5. Multi-Frame Confirmation

Never declare match from one frame.

A person should be matched only if the same identity appears in multiple frames.

Example rule:

```text
At least 3 strong matches within 10 frames
```

or:

```text
At least 5 matching detections from same track in last 2 seconds
```

Recommended initial rule:

```python
MIN_MATCH_FRAMES = 3
MIN_AVG_SIMILARITY = 0.48
```

Decision:

```text
If same face track matches target in 3+ frames and average similarity > threshold:
    show match
else:
    no reliable match
```

---

## 6. Track-Level Matching

In a crowded scene, each detected person should get a temporary track ID.

Use:

```text
ByteTrack
```

or simpler version:

```text
IoU-based face box tracking
```

Tracking helps because the system can say:

```text
Track #4 matched target across 7 frames
```

instead of:

```text
Frame 81 random crop matched target
```

For demo, even simple tracking is enough.

### Simple tracking logic

For each frame:

1. Detect faces.
2. Compare current face boxes with previous face boxes using IoU.
3. Assign same track ID if IoU is high.
4. Store embeddings per track.
5. Decide identity at track level.

---

# New Backend Architecture

## Recommended Folder Structure

```text
backend/
│
├── app/
│   ├── main.py
│   ├── config.py
│   │
│   ├── api/
│   │   ├── routes_search.py
│   │   ├── routes_camera.py
│   │   └── routes_health.py
│   │
│   ├── vision/
│   │   ├── detector.py
│   │   ├── aligner.py
│   │   ├── recognizer.py
│   │   ├── quality.py
│   │   ├── matcher.py
│   │   ├── tracker.py
│   │   └── pipeline.py
│   │
│   ├── storage/
│   │   ├── uploads/
│   │   ├── frames/
│   │   ├── crops/
│   │   └── results/
│   │
│   └── utils/
│       ├── video.py
│       ├── image.py
│       └── time.py
│
├── models/
│   ├── insightface/
│   └── adaface/
│
├── requirements.txt
└── README.md
```

---

# New Search Flow

## Input

```text
target image
CCTV video or live rolling buffer
```

## Step-by-step flow

```text
1. Read target image
2. Detect face in target image
3. Reject target image if no clear face
4. Align target face
5. Generate target embedding
6. Read CCTV frames
7. Detect all faces in each frame
8. Reject weak detections
9. Align each face
10. Run quality filtering
11. Generate embeddings
12. Track faces over frames
13. Compare track embeddings with target embedding
14. Aggregate scores per track
15. Return only reliable matches
```

---

# Result Format

Backend should return results like this:

```json
{
  "status": "completed",
  "decision": "match_found",
  "matches": [
    {
      "track_id": 4,
      "timestamp_start": "00:00:18",
      "timestamp_end": "00:00:27",
      "best_frame": "/results/frame_540.jpg",
      "best_crop": "/results/crop_track4.jpg",
      "avg_similarity": 0.53,
      "best_similarity": 0.58,
      "matched_frames": 6,
      "decision": "strong_match"
    }
  ],
  "rejected": {
    "low_quality_faces": 21,
    "small_faces": 12,
    "low_detection_score": 8
  }
}
```

If no good match:

```json
{
  "status": "completed",
  "decision": "no_reliable_match",
  "matches": [],
  "message": "No reliable match found in the selected footage."
}
```

---

# Frontend Result Display Rules

Do not show random nearest matches.

Show sections:

```text
Strong Matches
Possible Matches
No Reliable Match
Rejected Low-Quality Faces
```

For demo:

```text
Target found in CCTV1
Time: 00:01:14 - 00:01:21
Match strength: Strong
Frames confirmed: 5
```

Avoid:

```text
Confidence: 98%
```

Use:

```text
Match Strength: Strong
Similarity: 0.53
Verified across 5 frames
```

---

# Implementation Plan for Claude Code

## Phase 1: Replace Current Face Model with InsightFace

### Goal

Get immediate improvement with minimum code risk.

### Tasks

1. Install dependencies:

```bash
pip install insightface onnxruntime opencv-python numpy scikit-learn
```

2. Create:

```text
app/vision/insightface_engine.py
```

3. Load model once at startup:

```python
from insightface.app import FaceAnalysis

face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=-1, det_size=(640, 640))
```

For GPU:

```python
face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))
```

4. Replace MediaPipe detection and FaceNet embedding with:

```python
faces = face_app.get(frame)
```

Each face object gives:

```python
face.bbox
face.kps
face.det_score
face.embedding
```

5. Add rejection:

```python
if face.det_score < MIN_DETECTION_SCORE:
    reject
if face size < MIN_FACE_SIZE:
    reject
if blur_score < MIN_BLUR_SCORE:
    reject
```

6. Normalize embeddings before matching.

7. Use cosine similarity.

8. Add open-set threshold.

9. Return `"no_reliable_match"` if threshold not passed.

---

## Phase 2: Add Quality Filter

Create:

```text
app/vision/quality.py
```

Functions:

```python
def compute_blur_score(face_crop):
    ...

def compute_brightness(face_crop):
    ...

def is_valid_face_crop(face_crop, bbox, det_score):
    ...
```

Reject bad crops before embedding comparison.

---

## Phase 3: Add Multi-Frame Aggregation

Create:

```text
app/vision/track_aggregator.py
```

For each detected face, store:

```python
{
  "track_id": int,
  "embeddings": [],
  "similarities": [],
  "timestamps": [],
  "best_frame": str,
  "best_crop": str
}
```

Final decision:

```python
avg_similarity = average(top_k_similarities)
matched_frames = count(similarity >= MATCH_THRESHOLD)

if matched_frames >= MIN_MATCH_FRAMES and avg_similarity >= MIN_AVG_SIMILARITY:
    return strong_match
else:
    reject
```

---

## Phase 4: Add AdaFace Later

After InsightFace pipeline is stable, add AdaFace as optional recognizer.

Config:

```python
RECOGNIZER_BACKEND = "insightface"  # or "adaface"
```

Then create:

```text
app/vision/adaface_engine.py
```

Important:

```text
AdaFace expects BGR input in official implementation.
InsightFace commonly uses RGB conventions internally.
Do not mix color formats blindly.
```

---

# Recommended Config File

Create:

```text
app/config.py
```

Suggested config:

```python
DETECTOR_BACKEND = "insightface"   # insightface, scrfd
RECOGNIZER_BACKEND = "insightface" # insightface, adaface

DET_SIZE = (640, 640)

MIN_DETECTION_SCORE = 0.75
MIN_FACE_SIZE = 55

MIN_BLUR_SCORE = 80.0
MIN_BRIGHTNESS = 40
MAX_BRIGHTNESS = 220

MATCH_THRESHOLD = 0.50
POSSIBLE_MATCH_THRESHOLD = 0.43

MIN_MATCH_FRAMES = 3
MIN_AVG_SIMILARITY = 0.48

FRAME_SKIP = 5
MAX_RESULTS = 10

ALLOW_NO_MATCH = True
```

For high precision demo, use:

```python
MIN_DETECTION_SCORE = 0.80
MATCH_THRESHOLD = 0.52
MIN_MATCH_FRAMES = 4
```

---

# Matching Logic

## Cosine Similarity

Use normalized embeddings.

```python
import numpy as np

def normalize_embedding(embedding):
    embedding = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return None
    return embedding / norm

def cosine_similarity(a, b):
    a = normalize_embedding(a)
    b = normalize_embedding(b)
    if a is None or b is None:
        return -1.0
    return float(np.dot(a, b))
```

---

# Safe Decision Logic

```python
def decide_track_match(similarities):
    strong_scores = [s for s in similarities if s >= MATCH_THRESHOLD]

    if len(strong_scores) >= MIN_MATCH_FRAMES:
        avg_top = sum(sorted(strong_scores, reverse=True)[:5]) / min(5, len(strong_scores))

        if avg_top >= MIN_AVG_SIMILARITY:
            return {
                "decision": "strong_match",
                "avg_similarity": avg_top,
                "matched_frames": len(strong_scores)
            }

    possible_scores = [s for s in similarities if s >= POSSIBLE_MATCH_THRESHOLD]

    if len(possible_scores) >= MIN_MATCH_FRAMES + 1:
        return {
            "decision": "possible_match",
            "avg_similarity": sum(possible_scores) / len(possible_scores),
            "matched_frames": len(possible_scores)
        }

    return {
        "decision": "no_reliable_match",
        "avg_similarity": max(similarities) if similarities else None,
        "matched_frames": 0
    }
```

---

# Critical Do Not Do Rules

## Do not always return nearest face

Bad:

```python
best_match = max(all_faces, key=lambda x: x.similarity)
return best_match
```

Good:

```python
if best_match.similarity >= threshold and confirmed_across_frames:
    return best_match
else:
    return no_reliable_match
```

## Do not compare unaligned faces

Bad:

```text
raw crop → embedding
```

Good:

```text
landmark aligned face → embedding
```

## Do not show cosine score as confidence

Bad:

```text
Confidence: 97%
```

Good:

```text
Match Strength: Strong
Similarity Score: 0.53
Confirmed Frames: 6
```

## Do not process background crops

Bad:

```text
Any crop from detector/tracker → embedding
```

Good:

```text
det_score + face size + landmarks + blur + brightness check → embedding
```

---

# Live Phone Camera Plan

This part can be implemented after model upgrade.

## Goal

Use phone camera as CCTV feed.

Demo flow:

```text
Phone placed in room corner
        ↓
Phone camera stream
        ↓
Laptop receives stream
        ↓
Frontend shows feed in CCTV1 slot
        ↓
Backend keeps last 2 minutes
        ↓
User uploads target image
        ↓
System searches last 2 minutes
        ↓
Matched person shown with timestamp/frame
```

---

## Recommended Method

Use:

```text
Phone IP camera app / RTSP app
        ↓
MediaMTX or direct RTSP
        ↓
OpenCV backend
        ↓
Frontend CCTV slot
```

Simpler first demo:

```text
Phone IP Webcam app → RTSP/HTTP URL → OpenCV VideoCapture
```

Example:

```python
cap = cv2.VideoCapture("rtsp://PHONE_IP:PORT/stream")
```

or:

```python
cap = cv2.VideoCapture("http://PHONE_IP:PORT/video")
```

---

# Rolling 2-Minute Buffer

## Purpose

Search only the last 2 minutes of live footage.

## Simple Backend Strategy

Maintain a circular frame buffer in memory or disk.

For each camera:

```python
camera_buffer = deque(maxlen=MAX_BUFFER_FRAMES)
```

If processing 5 FPS for search:

```python
MAX_BUFFER_FRAMES = 2 minutes * 60 seconds * 5 FPS
MAX_BUFFER_FRAMES = 600 frames
```

Store:

```python
{
  "frame": frame,
  "timestamp": datetime,
  "camera_id": "cctv1"
}
```

When user searches:

```text
take current buffer snapshot → run face search → return matches
```

---

# Suggested Camera API

## Connect camera

```http
POST /camera/connect
```

Body:

```json
{
  "camera_id": "cctv1",
  "stream_url": "http://192.168.1.10:8080/video"
}
```

## Camera status

```http
GET /camera/status/cctv1
```

## Search last 2 minutes

```http
POST /search/live-buffer
```

Body:

```json
{
  "camera_id": "cctv1",
  "target_image": "file",
  "duration_seconds": 120
}
```

---

# Final Claude Code Prompt

Use this prompt in Claude Code:

```text
We need to upgrade the Drishti face recognition backend to reduce false positives and stop wrong matches with other people, clothes, walls, or background regions.

Current pipeline uses OpenCV + MediaPipe + FaceNet + cosine similarity. Replace this with a safer open-set face search pipeline.

Implement Phase 1 using InsightFace buffalo_l or buffalo_m:
- Add InsightFace model loading once at backend startup.
- Use FaceAnalysis(name="buffalo_l") with ONNX Runtime.
- Use CPUExecutionProvider by default and CUDAExecutionProvider if available.
- Replace MediaPipe face detection and FaceNet embedding calls with InsightFace face_app.get(frame).
- Use face.bbox, face.kps, face.det_score, and face.embedding.
- Add strict rejection before matching:
  - detection score below config.MIN_DETECTION_SCORE
  - face box width/height below config.MIN_FACE_SIZE
  - blur score below config.MIN_BLUR_SCORE
  - brightness outside config.MIN_BRIGHTNESS and config.MAX_BRIGHTNESS
  - missing landmarks
- Normalize embeddings before similarity comparison.
- Use cosine similarity but do not treat it as percentage confidence.
- Add open-set threshold logic:
  - return strong_match only if similarity passes MATCH_THRESHOLD and appears across multiple frames
  - return possible_match only if it passes POSSIBLE_MATCH_THRESHOLD across multiple frames
  - otherwise return no_reliable_match
- Do not always return nearest face.
- Add multi-frame aggregation:
  - group detections by simple IoU-based track IDs initially
  - store similarities per track
  - final decision should be track-level, not single-frame-level
- Add config.py with all thresholds.
- Add result JSON fields:
  - decision
  - track_id
  - timestamp_start
  - timestamp_end
  - best_frame
  - best_crop
  - avg_similarity
  - best_similarity
  - matched_frames
  - rejected counts
- Update frontend result labels:
  - Strong Match
  - Possible Match
  - No Reliable Match
  - Similarity Score
  - Confirmed Frames
- Remove or avoid “Confidence: XX%” wording.
- Keep existing OpenCV video reading and frame extraction if it works.
- Keep the current frontend layout unless required by API changes.
- Ensure the backend returns no_reliable_match instead of showing random closest faces.

After Phase 1 is stable, prepare clean interfaces so AdaFace R100 can later be added as recognizer backend with RECOGNIZER_BACKEND="adaface".

Also add a future-ready camera module structure:
- routes_camera.py
- camera_manager.py
- rolling_buffer.py
But do not fully implement phone camera streaming yet unless existing architecture already supports it.
```

---

# Final Recommendation

For fastest improvement:

```text
Use InsightFace buffalo_l first.
```

For best final version:

```text
SCRFD-10GF / SCRFD-34GF + AdaFace R100 + quality gate + track-level aggregation.
```

Most important fix:

```text
Do not force a match.
Return "No Reliable Match" when evidence is weak.
```
