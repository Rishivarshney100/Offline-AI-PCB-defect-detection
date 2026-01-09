import argparse
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description="PCB Defect AI Agent")
    parser.add_argument("--port", type=int, default=7860, help="Port to run UI")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    
    args = parser.parse_args()
    
    print(f"Starting PCB Defect AI Agent...")
    print(f"Server: http://{args.host}:{args.port}")
    
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "ui/streamlit_app.py",
        "--server.port", str(args.port),
        "--server.address", args.host
    ]
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
