"""
Live-camera package (SKELETON — not implemented yet).

Future design:
    Phone/IP camera RTSP/HTTP stream -> OpenCV VideoCapture -> rolling 2-minute
    in-memory buffer -> on search, run the open-set face pipeline over the last
    ~2 minutes of frames for that camera.

Only placeholders exist today; nothing streams.
"""
