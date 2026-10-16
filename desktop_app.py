# ==============================================================================
#      _    _ ______ __   __ _____ _______ ______ _____  ______ ____  
#     | |  | |  ____|\ \ / // ____|__   __|  ____|  __ \|  ____/ __ \ 
#     | |__| | |__    \ V /| (___    | |  | |__  | |__) | |__ | |  | |
#     |  __  |  __|    > <  \___ \   | |  |  __| |  _  /|  __|| |  | |
#     | |  | | |____  / . \ ____) |  | |  | |____| | \ \| |___| |__| |
#     |_|  |_|______|/_/ \_\_____/   |_|  |______|_|  \_\______\____/ 
#
#                         NeuralForge Studio
#                        Created by HEXSTEREO
# ==============================================================================
"""NeuralForge Desktop Application Runner.

Runs NeuralForge in a dedicated, native desktop application window.
Ensures backend services are active, and provides a frameless/chromeless
native window experience using Microsoft Edge WebView2 via pywebview,
with automatic fallback to Edge App Mode if needed.

Author: hexstereo (https://github.com/HexStereo)
"""

HEXSTEREO_ASCII = r"""
  _    _ ______ __   __ _____ _______ ______ _____  ______ ____  
 | |  | |  ____|\ \ / // ____|__   __|  ____|  __ \|  ____/ __ \ 
 | |__| | |__    \ V /| (___    | |  | |__  | |__) | |__ | |  | |
 |  __  |  __|    > <  \___ \   | |  |  __| |  _  /|  __|| |  | |
 | |  | | |____  / . \ ____) |  | |  | |____| | \ \| |___| |__| |
 |_|  |_|______|/_/ \_\_____/   |_|  |______|_|  \_\______\____/ 
                  --- Crafted by hexstereo ---
"""

import sys
import os
import time
import subprocess
import urllib.request
import threading
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent
ICON_PATH = ROOT_DIR / "scripts" / "neuralforge.ico"
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8008
DEV_PORT = 5173

def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a local TCP port is already open."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0

def wait_for_service(url: str, timeout: float = 8.0) -> bool:
    """Poll an HTTP endpoint until it returns 200 or timeout expires."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.3)
    return False

def start_backend_process():
    """Start FastAPI uvicorn backend if not already listening."""
    if is_port_in_use(BACKEND_PORT):
        print(f"[NeuralForge] Backend already running on port {BACKEND_PORT}")
        return None

    python_exe = sys.executable
    print(f"[NeuralForge] Spawning backend server on port {BACKEND_PORT}...")
    
    # Hide console window on Windows if using pythonw/subprocess
    startupinfo = None
    creationflags = 0
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        creationflags = subprocess.CREATE_NO_WINDOW

    proc = subprocess.Popen(
        [
            python_exe,
            "-m",
            "uvicorn",
            "backend.app.main:app",
            "--host",
            BACKEND_HOST,
            "--port",
            str(BACKEND_PORT),
        ],
        cwd=str(ROOT_DIR),
        startupinfo=startupinfo,
        creationflags=creationflags,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc

def get_target_url() -> str:
    """Determine the best URL to open (Vite dev server if running, else FastAPI SPA)."""
    if is_port_in_use(DEV_PORT):
        return f"http://localhost:{DEV_PORT}/"
    return f"http://{BACKEND_HOST}:{BACKEND_PORT}/"

def launch_pywebview(target_url: str):
    """Launch the native pywebview desktop window."""
    import webview

    title = "NeuralForge — AI Research Studio | hexstereo"
    icon = str(ICON_PATH) if ICON_PATH.exists() else None

    # Create native window
    window = webview.create_window(
        title=title,
        url=target_url,
        width=1440,
        height=900,
        min_size=(1024, 680),
        background_color="#0a0b10",
        text_select=True,
    )

    webview.start(gui="edgechromium", debug=False)

def launch_edge_app_fallback(target_url: str):
    """Fallback: open standalone Edge App window (looks like a native desktop app)."""
    print("[NeuralForge] Launching standalone Edge App window fallback...")
    cmd = [
        "msedge.exe",
        f"--app={target_url}",
        "--window-size=1440,900",
        "--enable-features=msEdgeSidebar",
    ]
    subprocess.Popen(cmd, shell=True)

def main():
    print(HEXSTEREO_ASCII)
    print("=" * 66)
    print("   NEURALFORGE - AI RESEARCH STUDIO  [Created by HEXSTEREO]")
    print("=" * 66)


    # 1. Start backend if necessary
    backend_proc = start_backend_process()

    # 2. Wait for backend to be healthy
    health_url = f"http://{BACKEND_HOST}:{BACKEND_PORT}/api/health"
    if not wait_for_service(health_url, timeout=10.0):
        print(f"[WARNING] Backend did not respond at {health_url} in time.")

    # 3. Determine target URL
    target_url = get_target_url()
    print(f"[NeuralForge] Connecting desktop UI to: {target_url}")

    # 4. Launch Desktop Window
    try:
        launch_pywebview(target_url)
    except Exception as e:
        print(f"[NeuralForge] pywebview failed ({e}). Falling back to Edge App Mode...")
        launch_edge_app_fallback(target_url)

    # 5. Cleanup if we spawned backend
    if backend_proc and backend_proc.poll() is None:
        print("[NeuralForge] Shutting down backend...")
        backend_proc.terminate()

if __name__ == "__main__":
    main()
