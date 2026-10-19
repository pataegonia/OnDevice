"""Rate-distortion curves of the feature codec, read straight from evaluation logs.

    python scripts/plot_rd.py                        # C50k @ 50k steps, logs in slurm/
    python scripts/plot_rd.py --logs logs/slurm --runs "C50k_lmbda*_s36k" --out out/rd_c50k_36k

Every number comes from an `eval_codec.slurm` stdout log: the header line names the run
(`run=...`), and the final lines hold psnr / ssim / lpips / bitstream_kib. If a run was
evaluated more than once, the newest log wins. The x axis is the whole bitstream file per
scene (y+z streams plus the camera/shape header), on a log scale.
Writes <out>.png (300 dpi) and <out>.pdf and prints the values it plotted.
"""
import argparse
import fnmatch
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

# 2 context views x 256 x 256 pixel-aligned Gaussians, 86 floats each at sh_degree 4
# (75 SH + 3 scale + 4 rotation + 3 mean + 1 opacity), stored as float32.
RAW_KIB = 2 * 256 * 256 * 86 * 4 / 1024

# Reference palette (dataviz skill), light mode.
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SERIES = "#e1e0d9", "#c3c2b7", "#2a78d6"


def parse_eval(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    head = re.search(r"(?m)^experiment=(\S+).*?ckpt=(\S+)", text)
    if not head:
        return None
    run = re.search(r"run=(\S+)", text.splitlines()[0] if text else "")
    scores = {k: float(v) for k, v in re.findall(
        r"(?m)^(psnr|ssim|lpips|codec_kib|bitstream_kib) ([\d.eE+-]+)$", text)}
    if "psnr" not in scores:
        return None
    scenes = re.search(r"(?m)^decompress: (\d+) calls", text) or re.search(r"(?m)^encoder: (\d+) calls", text)
    return {
        "run": run.group(1) if run else None,
        "experiment": head.group(1),
        "ckpt": head.group(2),
        "scenes": int(scenes.group(1)) if scenes else None,
        "log": path,
        **scores,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="slurm", help="folder holding the eval .out logs")
    ap.add_argument("--runs", default="C50k_lmbda*_s50k", help="glob over the run= names to plot")
    ap.add_argument("--baseline-log", default="slurm/slurm-mvsplat-codec-eval-430844.out",
                    help="full-test-set eval of the released checkpoint without codec")
    ap.add_argument("--out", default="out/rd_c50k_s50k")
    args = ap.parse_args()

    latest = {}
    for path in sorted(Path(args.logs).glob("*eval*.out"), key=lambda p: p.stat().st_mtime):
        r = parse_eval(path)
        if r and r["run"] and fnmatch.fnmatch(r["run"], args.runs):
            latest[r["run"]] = r
    points = []
    for r in latest.values():
        lam = re.search(r"lmbda([\d.]+?)(?:_r\d+)?_s", r["run"])
        if lam:
            points.append({**r, "lmbda": float(lam.group(1))})
    if not points:
        raise SystemExit(f"no eval logs in {args.logs} with run= matching {args.runs}")
    points.sort(key=lambda p: p["bitstream_kib"])
    base = parse_eval(Path(args.baseline_log))

    print(f"{'lambda':>7} {'file KiB':>9} {'x smaller':>9} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'scenes':>6}  log")
    for p in points:
        print(f"{p['lmbda']:7.4g} {p['bitstream_kib']:9.2f} {RAW_KIB / p['bitstream_kib']:9.0f} "
              f"{p['psnr']:7.3f} {p['ssim']:7.4f} {p['lpips']:7.4f} {p['scenes']:>6}  {p['log'].name}")
    if base:
        print(f"{'none':>7} {'(raw)':>9} {'1':>9} {base['psnr']:7.3f} {base['ssim']:7.4f} {base['lpips']:7.4f} "
              f"{base['scenes']:>6}  {Path(args.baseline_log).name}")

    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 9,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
    })
    panels = [("psnr", "PSNR (dB) ↑"), ("ssim", "SSIM ↑"), ("lpips", "LPIPS ↓")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), facecolor=SURFACE)
    xs = [p["bitstream_kib"] for p in points]
    for ax, (key, label) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(True, which="major", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(1.6, 800)
        ax.set_xlabel("Bitstream per scene (KiB, log scale)")
        ax.set_ylabel(label)

        if base:
            ax.axhline(base[key], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
            ax.annotate(f"MVSplat, no compression ({base[key]:.3f})" if key != "psnr"
                        else f"MVSplat, no compression ({base[key]:.2f} dB)",
                        xy=(1.7, base[key]), xytext=(0, 4 if key != "lpips" else -11),
                        textcoords="offset points", color=INK_2, fontsize=8)

        ys = [p[key] for p in points]
        ax.plot(xs, ys, color=SERIES, linewidth=2, solid_capstyle="round", solid_joinstyle="round",
                marker="o", markersize=6.5, markerfacecolor=SERIES, markeredgecolor=SURFACE,
                markeredgewidth=1.6, zorder=3)

        if key == "psnr":
            for i, p in enumerate(points):
                last = i == len(points) - 1  # rightmost point: label to its left, inside the panel
                ax.annotate(f"λ={p['lmbda']:g}\n{RAW_KIB / p['bitstream_kib']:,.0f}× smaller",
                            xy=(p["bitstream_kib"], p["psnr"]), xytext=(-6 if last else 6, -22),
                            textcoords="offset points", ha="right" if last else "left",
                            color=INK, fontsize=7.5, linespacing=1.1)
        lo = min(ys + ([base[key]] if base else []))
        hi = max(ys + ([base[key]] if base else []))
        pad = (hi - lo) * 0.12
        ax.set_ylim(lo - (0.55 if key == "psnr" else pad), hi + pad)

    n = points[0]["scenes"]
    fig.suptitle("Feature codec (cut C, after the refinement U-Net) — rate–distortion at 50k steps",
                 x=0.012, ha="left", color=INK, fontsize=11.5, fontweight="semibold")
    fig.text(0.012, 0.885,
             f"RE10K test set, {n:,} scenes, 2 context views at 256×256. x = whole bitstream file "
             f"(y+z streams + camera header). “× smaller” is against the raw Gaussians "
             f"({RAW_KIB * 1024 / 1e6:.0f} MB as float32).",
             ha="left", color=INK_2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300, facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    print(f"wrote {out.with_suffix('.png')} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
