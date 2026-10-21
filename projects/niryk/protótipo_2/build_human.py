"""Kept for compatibility: builds the human male with build_characters.py (same flags).

    python projects/niryk/protótipo_2/build_human.py [--fast] [--no-review] ...
    ==  python projects/niryk/protótipo_2/build_characters.py human_m [--fast] [--no-review] ...
"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "build_characters.py"
    sys.exit(subprocess.call([sys.executable, str(script), "human_m"] + sys.argv[1:]))
