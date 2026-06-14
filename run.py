import os
import socket
import sys
from contextlib import closing
from urllib.error import URLError
from urllib.request import urlopen

import uvicorn

# Get port from environment or default to 8000
PORT = int(os.getenv('PORT', 8000))
HOST = os.getenv('HOST', '0.0.0.0')


def is_port_in_use(host: str, port: int) -> bool:
    """Return True when the target port already accepts TCP connections."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


def existing_drishti_url(port: int) -> str | None:
    """Return the local DRISHTI URL when an existing healthy instance is running."""
    candidates = [
        f"http://127.0.0.1:{port}",
        f"http://localhost:{port}",
    ]

    for base_url in candidates:
        try:
            with urlopen(f"{base_url}/api/health", timeout=2) as response:
                payload = response.read().decode("utf-8")
            if response.status == 200 and '"status":"healthy"' in payload:
                return base_url
        except (URLError, TimeoutError, OSError):
            continue

    return None

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('data/uploads', exist_ok=True)
    os.makedirs('data/results', exist_ok=True)
    os.makedirs('CCTVS', exist_ok=True)

    if is_port_in_use('127.0.0.1', PORT):
        running_url = existing_drishti_url(PORT)
        if running_url:
            print(f"DRISTI is already running at {running_url}")
            sys.exit(0)
        print(
            f"Port {PORT} is already in use by another process. "
            "Set a different PORT environment variable or stop the conflicting process."
        )
        sys.exit(1)
    
    # Run the app
    uvicorn.run(
        'src.backend.app:app',
        host=HOST,
        port=PORT,
        reload=False  # Disable reload in production
    )
