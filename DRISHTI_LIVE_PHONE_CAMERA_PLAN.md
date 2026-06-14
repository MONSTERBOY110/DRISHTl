# Drishti Live Phone Camera / CCTV Integration Plan

## Goal

Add a live phone camera feed to Drishti and use it like a CCTV camera.

Demo scenario:

```text
Phone placed in one corner of the room
        ↓
Phone camera streams to laptop
        ↓
Drishti frontend shows it in a CCTV slot
        ↓
Backend keeps a rolling recent buffer
        ↓
User uploads target image
        ↓
System searches recent live footage
        ↓
Result shows camera, timestamp, best frame, crop, similarity, and match strength
```

Important performance constraint:

Current static clips are only around 3–4 seconds each, but search still takes around 1–1.5 minutes on CPU.  
So for live camera, we should **not** wait until the user searches and then process the entire last 2 minutes from scratch.

The better approach is:

```text
Process live camera continuously in the background at low FPS
Store face embeddings/tracks in a rolling buffer
When user uploads target image, compare against already-processed embeddings
Return result quickly
```

---

## Recommended Live Architecture

```text
Phone Camera
   ↓
HTTP/MJPEG or RTSP Stream
   ↓
OpenCV VideoCapture in backend
   ↓
Camera Capture Thread
   ↓
Latest Frame Cache
   ↓
Frontend Live Preview
   ↓
Low-FPS Face Processing Worker
   ↓
Rolling Embedding Buffer
   ↓
Search Target Image vs Buffer
   ↓
Fast Recent Match Result
```

---

## Why not process the last 2 minutes only after clicking Search?

Bad approach:

```text
User clicks search
        ↓
Backend takes last 2 minutes of raw frames
        ↓
Runs InsightFace on all frames
        ↓
Takes too long on CPU
```

If 2 minutes is processed at even 1 FPS:

```text
120 seconds × 1 frame/sec = 120 frames
```

On CPU, InsightFace inference over 120 frames can still take a long time.

Better approach:

```text
Camera is connected
        ↓
Backend continuously processes 1 frame/sec
        ↓
Embeddings are already ready
        ↓
Search only compares vectors
        ↓
Result comes much faster
```

---

## Recommended Demo Settings

For live phone camera on CPU:

```python
LIVE_PROCESS_FPS = 1
LIVE_BUFFER_SECONDS = 60
LIVE_DET_SIZE = 480
LIVE_MATCH_THRESHOLD = 0.55
LIVE_POSSIBLE_MATCH_THRESHOLD = 0.45
LIVE_MIN_MATCH_FRAMES = 2
```

Suggested first implementation:

```text
1 camera only
60-second rolling buffer
1 FPS processing
480 detector size
Search already-indexed embeddings
```

---

## Phone Connection Options

### Option A — Easiest Android method: IP Webcam app

Install on phone:

```text
IP Webcam
```

Typical flow:

1. Connect phone and laptop to the same Wi-Fi network.
2. Open the IP Webcam app.
3. Start server.
4. The app shows an address like:

```text
http://192.168.1.34:8080
```

5. Video stream usually becomes:

```text
http://192.168.1.34:8080/video
```

6. Use this URL in Drishti's Connect CCTV popup.

Backend uses:

```python
cv2.VideoCapture("http://192.168.1.34:8080/video")
```

### Option B — RTSP Camera app

Some apps expose RTSP directly:

```text
rtsp://192.168.1.34:8554/live
```

Backend uses:

```python
cv2.VideoCapture("rtsp://192.168.1.34:8554/live")
```

### Option C — DroidCam / Iriun

These are useful for turning phone into webcam, but for Drishti backend, direct HTTP/RTSP URL is simpler.

Recommended for Drishti:

```text
Use IP Webcam or any app that gives HTTP MJPEG / RTSP stream URL.
```

---

## Backend Routes Needed

### Connect camera

```http
POST /api/camera/connect
```

Request:

```json
{
  "camera_id": "phone1",
  "name": "Phone Camera",
  "stream_url": "http://192.168.1.34:8080/video",
  "process_fps": 1,
  "buffer_seconds": 60
}
```

### Camera status

```http
GET /api/camera/status/{camera_id}
```

### Live preview stream

```http
GET /api/camera/stream/{camera_id}
```

Use MJPEG output:

```html
<img src="/api/camera/stream/phone1" />
```

This is often easier than forcing `<video>` to play arbitrary phone MJPEG streams.

### Disconnect camera

```http
POST /api/camera/disconnect/{camera_id}
```

### Search live buffer

```http
POST /api/search/live-buffer
```

Request:

```text
multipart/form-data:
- file: target image
- camera_id: phone1
- duration_seconds: 60
```

---

## Backend Module Design

Recommended files:

```text
src/backend/camera/
├── __init__.py
├── camera_manager.py
├── camera_worker.py
├── rolling_buffer.py
└── live_search.py
```

Existing placeholder files can be extended.

### camera_manager.py

Responsibilities:

```text
- manage connected cameras
- start/stop worker threads
- store camera status
- expose latest frame
- expose rolling embedding buffer
```

### camera_worker.py

Responsibilities:

```text
- open stream with cv2.VideoCapture
- continuously read frames
- update latest frame
- process selected frames at LIVE_PROCESS_FPS
- run InsightFace detection/embedding
- apply quality gate
- push valid embeddings into rolling buffer
```

Important:

```text
Do not process every frame.
Only process 1 FPS initially.
```

### rolling_buffer.py

Store recent processed face data.

Recommended structure:

```python
{
  "timestamp": datetime,
  "frame": frame_or_saved_path,
  "faces": [
    {
      "bbox": [x1, y1, x2, y2],
      "embedding": "np.ndarray",
      "det_score": 0.91,
      "quality": {}
    }
  ]
}
```

Buffer should be time-based:

```python
remove entries older than buffer_seconds
```

Do not store unlimited frames.

### live_search.py

Responsibilities:

```text
- detect target face from uploaded photo
- generate target embedding
- compare target embedding with embeddings in camera rolling buffer
- aggregate matches
- return match_found / no_reliable_match
```

For first version, simple aggregation is acceptable:

```text
All recent face embeddings from same camera
        ↓
compare with target
        ↓
keep hits above threshold
        ↓
if at least MIN_MATCH_FRAMES hits, return match
```

Later improvement:

```text
IoU-based live tracking across buffer frames
```

---

## Frontend UI Plan

Use the existing Connect CCTV button and popup.

Fields needed:

```text
Camera Name
Stream URL
Process FPS
Buffer Seconds
Connect Button
```

Default values:

```text
Camera Name: Phone Camera
Process FPS: 1
Buffer Seconds: 60
```

Example URL placeholder:

```text
http://192.168.1.34:8080/video
```

Add helper text:

```text
Install IP Webcam on your phone, connect phone and laptop to the same Wi-Fi, start server, then paste the /video URL here.
```

When connected:

```text
Phone Camera should appear as one CCTV feed card.
```

Preview source:

```html
<img src="/api/camera/stream/phone1" />
```

Search mode:

```text
Search Source:
[Uploaded CCTV Clips] [Live Phone Camera]
```

Simpler first implementation:

```text
If a phone camera is connected, show a "Search Live Camera" button.
```

---

## Expected Demo Flow

1. Connect laptop and phone to the same Wi-Fi.
2. Open IP Webcam app on phone.
3. Start server.
4. Copy URL:

```text
http://PHONE_IP:8080/video
```

5. Start Drishti:

```bash
python run.py
```

6. Open:

```text
http://localhost:8000
```

7. Click:

```text
Connect CCTV
```

8. Paste phone stream URL.
9. Phone feed appears in CCTV panel.
10. Wait 15–20 seconds so the rolling buffer fills.
11. Upload target image.
12. Click:

```text
Search Live Camera
```

13. Result should come from already processed live buffer.

---

## Performance Expectation

Current CPU static-video performance:

```text
3–4 sec clips can take around 1–1.5 minutes
```

Live approach should reduce perceived search time because processing happens continuously before search.

Expected:

```text
Camera warm-up / buffer fill: 15–60 seconds
Actual search after clicking: a few seconds to around 15 seconds
```

If search is slow:

```text
lower LIVE_PROCESS_FPS
lower DET_SIZE to 480 or 320
use shorter LIVE_BUFFER_SECONDS
use GPU later
```

---

## Recommended First Version Scope

Do this first:

```text
One phone camera only
HTTP/MJPEG URL support
Backend latest-frame preview
Rolling embedding buffer
Search live buffer
Frontend connect popup wired to backend
```

Do not do yet:

```text
multi-phone camera
cloud camera support
authentication
database persistence
HLS transcoding
WebRTC
GPU optimization
AdaFace integration
```

---

## Important Safety Rules

1. Do not break current local CCTVS video search.
2. Do not remove static CCTV clips feature.
3. Live camera should be additional.
4. If camera disconnects, UI should show disconnected status.
5. If buffer is empty, return:

```json
{
  "decision": "no_reliable_match",
  "message": "Live camera buffer is still warming up. Please wait a few seconds."
}
```

6. Never search unprocessed raw 2-minute video on CPU unless explicitly requested.
7. Prefer already-indexed live embeddings.

---

## Claude Code Execution Prompt

```text
We need to implement the live phone camera / CCTV module for Drishti.

Context:
The current static CCTV search is working well with InsightFace, but it is CPU-heavy. Even short 3–4 second clips can take around 1–1.5 minutes. Therefore, for live phone camera we should NOT process the full last 2 minutes from raw frames after the user clicks search.

Required design:
Process the live stream continuously in the background at low FPS, store recent face embeddings in a rolling buffer, and when the user uploads a target image, compare the target embedding against the already processed rolling buffer.

Goal:
Use a phone camera as CCTV for demo.

Phone connection method:
Support HTTP/MJPEG or RTSP URL from phone camera apps such as IP Webcam.
Example stream URL:
http://192.168.1.34:8080/video
or
rtsp://192.168.1.34:8554/live

Backend requirements:
1. Extend the existing camera placeholder module instead of creating a completely separate system.
2. Implement:
   - src/backend/camera/camera_manager.py
   - src/backend/camera/camera_worker.py
   - src/backend/camera/rolling_buffer.py
   - src/backend/camera/live_search.py
   - src/backend/routes_camera.py
3. Add API endpoints:
   - POST /api/camera/connect
   - GET /api/camera/status/{camera_id}
   - POST /api/camera/disconnect/{camera_id}
   - GET /api/camera/stream/{camera_id}
   - POST /api/search/live-buffer
4. Use OpenCV VideoCapture to read the stream URL.
5. Keep latest frame for live preview.
6. Expose latest frames as MJPEG stream from /api/camera/stream/{camera_id}.
7. Process only selected frames at low FPS:
   - default LIVE_PROCESS_FPS = 1
   - default LIVE_BUFFER_SECONDS = 60
   - default LIVE_DET_SIZE = 480 if supported by current config
8. Use existing InsightFace engine, quality gate, matcher, thresholds, and snapshot saving logic where possible.
9. Store recent processed embeddings in a time-based rolling buffer.
10. Do not store unlimited raw frames in memory.
11. When /api/search/live-buffer is called:
    - detect and embed target face
    - compare against embeddings in the selected camera buffer
    - aggregate matches
    - return match_found / no_reliable_match
    - include best_frame and best_crop snapshots
12. If buffer is empty or warming up, return:
    decision = "no_reliable_match"
    message = "Live camera buffer is still warming up. Please wait a few seconds."
13. If stream disconnects, update status and do not crash backend.
14. Keep existing static CCTVS folder search fully working.
15. Do not touch current static video recognition thresholds unless absolutely required.

Frontend requirements:
1. Use the existing Connect CCTV button/popup if available.
2. Wire it to POST /api/camera/connect.
3. Popup fields:
   - Camera Name
   - Stream URL
   - Process FPS default 1
   - Buffer Seconds default 60
4. Add helper text:
   "Install IP Webcam on your phone, connect phone and laptop to the same Wi-Fi, start server, then paste the /video URL here."
5. When connected, show the phone camera feed in the CCTV grid.
6. Use /api/camera/stream/{camera_id} for preview if possible.
7. Add a button or mode:
   - Search Uploaded Clips
   - Search Live Camera
8. Search Live Camera should call POST /api/search/live-buffer with uploaded target image and camera_id.
9. Show live search results using the same result card style:
   - Match Strength
   - Similarity Score
   - Confirmed Frames
   - Time Range
   - Snapshot
10. Do not remove or break local CCTVS video feed cards.

Performance:
- Use 1 FPS live processing first.
- Search should use already-indexed embeddings, not raw full-buffer processing.
- Expected actual search after buffer is warm should be much faster than static clip search.

Acceptance criteria:
1. User can paste phone stream URL and connect.
2. Phone feed appears in CCTV section.
3. Backend status shows connected and processed frame count increasing.
4. After 15–30 seconds, uploading a target image and searching live buffer returns results.
5. Static CCTVS search still works.
6. Backend does not crash if phone disconnects.
7. No Twilio/SMS feature is reintroduced.
8. No Cloudinary CCTV dependency is reintroduced.

After implementation:
- Tell me files changed.
- Tell me exact phone setup steps.
- Tell me exact test steps.
- Tell me the endpoints added.
- Tell me expected runtime/search time.
- Tell me any limitations.
```
