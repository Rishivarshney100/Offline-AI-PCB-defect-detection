import argparse
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description="PCB Defect AI Agent")
    parser.add_argument("--port", type=int, default=7860, help="Port to run UI")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind (use 0.0.0.0 for network access)")
    
    args = parser.parse_args()
    
    print(f"Starting PCB Defect AI Agent...")
    
    # For display, show localhost even if binding to 0.0.0.0
    display_host = "localhost" if args.host == "0.0.0.0" else args.host
    print(f"Server: http://{display_host}:{args.port}")
    if args.host == "0.0.0.0":
        print(f"  (Also accessible at: http://127.0.0.1:{args.port})")
    
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "ui/streamlit_app.py",
        "--server.port", str(args.port),
        "--server.address", args.host
    ]
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
