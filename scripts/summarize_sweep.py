"""Print the test scores of each checkpoint step scored by scripts/eval_sweep.slurm.

    python scripts/summarize_sweep.py C50k_lmbda0.0373 [C50k_lmbda0.1493 ...]

With no argument, every run that has sweep results is listed.
"""
import json
import re
import sys
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parents[1] / "outputs" / "test"
# A resume folder (_r1, _r2, ...) belongs to the same run as the original.
NAME = re.compile(r"^sweep_(?P<run>.+?)(_r\d+)?_s(?P<step>\d+)$")


def main():
    found = {}
    for d in TEST_DIR.glob("sweep_*"):
        m = NAME.match(d.name)
        scores = d / "scores_all_avg.json"
        if m and scores.exists():
            found.setdefault(m["run"], {})[int(m["step"])] = json.loads(scores.read_text())

    runs = [NAME.match(f"sweep_{r}_s0")["run"] for r in sys.argv[1:]] or sorted(found)
    for run in runs:
        rows = sorted(found.get(run, {}).items())
        print(f"== {run}")
        if not rows:
            print("   no sweep results yet")
            continue
        print(f"   {'step':>6} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'KiB':>8}  scenes")
        for step, s in rows:
            n = s.get("encoder", [None])[0]
            print(f"   {step:>6} {s.get('psnr', float('nan')):7.3f} {s.get('ssim', float('nan')):7.4f} "
                  f"{s.get('lpips', float('nan')):7.4f} {s.get('codec_kib', float('nan')):8.2f}  {n}")


if __name__ == "__main__":
    main()
