from __future__ import annotations
import argparse
import subprocess
from pathlib import Path

REPOS = {
    "autogad": "https://github.com/ZhongLIFR/AutoGAD2024.git",
    "anemone": "https://github.com/TrustAGI-Lab/ANEMONE.git",
    "cola": "https://github.com/TrustAGI-Lab/CoLA.git",
    "gradate": "https://github.com/FelixDJC/GRADATE.git",
    "sl_gad": "https://github.com/KimMeen/SL-GAD.git",
    "sub_cr": "https://github.com/Zjer12/Sub.git",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="third_party/checkouts")
    p.add_argument("--name", choices=sorted(REPOS), nargs="+")
    args = p.parse_args()
    root = Path(args.root); root.mkdir(parents=True, exist_ok=True)
    for name in args.name:
        dest = root / name
        if dest.exists():
            print(f"skip existing: {dest}")
            continue
        print(f"cloning {name} -> {dest}")
        subprocess.run(["git","clone","--depth","1",REPOS[name],str(dest)], check=True)


if __name__ == "__main__":
    main()
