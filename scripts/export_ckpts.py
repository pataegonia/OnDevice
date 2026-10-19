"""Make shareable copies of training checkpoints by dropping the optimizer state.

    python scripts/export_ckpts.py release outputs/codec/C50k_lmbda*_r*/checkpoints/*step_50000.ckpt

A training checkpoint is ~390 MB, two thirds of it Adam's moment estimates. This keeps
everything an evaluation needs (state_dict, global_step, Lightning version, loop and
callback state) and drops `optimizer_states`, leaving ~130 MB. The copies work as CKPT for
scripts/eval_codec.slurm but cannot be resumed with RESUME=true.

Each copy is named after its run with the resume suffix removed, e.g.
outputs/codec/C50k_lmbda0.0373_r2/checkpoints/epoch_37-step_50000.ckpt
-> <out>/C50k_lmbda0.0373_step50000.ckpt. A SHA256SUMS file is written next to them.
"""
import hashlib
import re
import sys
from pathlib import Path

import torch


def release_name(path: Path) -> str:
    run = re.sub(r"_r\d+$", "", path.parent.parent.name)
    step = re.search(r"step_(\d+)", path.name)
    return f"{run}_step{step.group(1)}.ckpt" if step else f"{run}_{path.stem}.ckpt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    sums = []
    for src in map(Path, sys.argv[2:]):
        dst = out_dir / release_name(src)
        if dst.exists():
            raise SystemExit(f"{dst} already exists; pick another output folder")
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
        ckpt.pop("optimizer_states", None)
        torch.save(ckpt, dst)
        sums.append(f"{sha256(dst)}  {dst.name}")
        print(f"{src} ({src.stat().st_size / 2**20:.0f} MiB) -> {dst} "
              f"({dst.stat().st_size / 2**20:.0f} MiB, step {ckpt.get('global_step')})")
    with (out_dir / "SHA256SUMS").open("a") as f:
        f.write("\n".join(sums) + "\n")


if __name__ == "__main__":
    main()
