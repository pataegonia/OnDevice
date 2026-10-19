from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import moviepy.editor as mpy
import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim
import numpy as np
import json

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..dataset import DatasetCfg
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization import layout
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    cosine_lr: bool


@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    save_image: bool
    save_video: bool
    eval_time_skip_steps: int
    compress: bool = False


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    lmbda: float
    # Drop a batch whose loss exceeds this instead of backpropagating it. None = off.
    skip_loss_above: float | None = None
    # If this many batches in a row are dropped, the model has diverged: stop the run
    # instead of idling on the GPU with every batch skipped.
    max_consecutive_skips: int = 64


def rate_denominator(harmonics_shape) -> int:
    """Number of scalars the uncompressed 3DGS representation would store.

    NOTE: rates divided by this are not image bpp. The denominator counts the SH
    coefficients plus scales(3) + rotations(4) + means(3) + opacity(1) of every
    Gaussian, so the value is not comparable to an image codec's bpp, and lmbda is
    tied to this definition. It also scales with sh_degree: 59 per Gaussian at
    sh_degree 3, 86 at sh_degree 4.
    """
    n, k, c, l = harmonics_shape
    return n * k * (c * l + 3 + 4 + 3 + 1)


def rate_loss(estimated_bits: dict, harmonics_shape) -> Tensor:
    """Estimated bits per scalar of the uncompressed 3DGS representation."""
    num_values = rate_denominator(harmonics_shape)
    return sum(bits / num_values for bits in estimated_bits.values())


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    # Allow checkpoint mismatch: a base checkpoint has no codec weights, and a codec
    # checkpoint carries keys for modules that are not ported here.
    strict_loading: bool = False

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.eval_cnt = 0

        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.time_skip_steps_dict = {"encoder": 0, "decoder": 0, "decompress": 0}

    def training_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape

        # Run the model.
        enc_outputs = self.encoder(
            batch["context"], self.global_step, False, scene_names=batch["scene"]
        )
        gaussians = enc_outputs["gaussians"]
        output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )
        target_gt = batch["target"]["image"]

        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean())

        # Compute and log loss.
        total_loss = 0
        for loss_fn in self.losses:
            loss = loss_fn.forward(output, batch, gaussians, self.global_step)
            self.log(f"loss/{loss_fn.name}", loss)
            total_loss = total_loss + loss

        # Rate-distortion: add the estimated bitrate of the feature codec.
        bpp = None
        if enc_outputs["estimated_bits"] is not None:
            bpp = rate_loss(enc_outputs["estimated_bits"], gaussians.harmonics.shape)
            self.log("loss/bpp", bpp)
            total_loss = total_loss + self.train_cfg.lmbda * bpp
        self.log("loss/total", total_loss)

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"bound = [{batch['context']['near'].detach().cpu().numpy().mean()} "
                f"{batch['context']['far'].detach().cpu().numpy().mean()}]; "
                f"loss = {total_loss:.6f}"
                # LocalLogger.log_metrics is a no-op, so self.log("loss/bpp") is
                # invisible without wandb; print it where it can actually be read.
                + ("" if bpp is None else f"; bpp = {bpp:.6f}")
            )
        self.log("info/near", batch["context"]["near"].detach().cpu().numpy().mean())
        self.log("info/far", batch["context"]["far"].detach().cpu().numpy().mean())
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        # One blown-up batch (loss in the hundreds, seen around step 7k-12k in several
        # codec runs) was enough to collapse the whole model to a constant render, and
        # norm clipping does not help because the direction itself is garbage. Skip it.
        limit = self.train_cfg.skip_loss_above
        if limit is not None:
            # Under DDP every rank must take the same branch: a rank that skipped its
            # backward would leave the others waiting in the gradient all-reduce.
            bad = torch.tensor(
                float(not torch.isfinite(total_loss) or total_loss > limit),
                device=self.device,
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(bad)
            if bad.item() > 0:
                self._consecutive_skips = getattr(self, "_consecutive_skips", 0) + 1
                print(
                    f"train step {self.global_step}; loss = {float(total_loss):.4g}; "
                    f"a batch on some rank exceeded {limit}, skipping this batch "
                    f"({self._consecutive_skips} in a row)"
                )
                if self._consecutive_skips >= self.train_cfg.max_consecutive_skips:
                    raise RuntimeError(
                        f"{self._consecutive_skips} consecutive batches above "
                        f"skip_loss_above={limit}: the model has diverged, stopping."
                    )
                # A zero loss tied to every parameter instead of None: the batch adds
                # nothing to the gradient, yet backward and the DDP all-reduce still run
                # on every rank, so replicas neither hang nor drift apart.
                return sum(p.sum() for p in self.parameters() if p.requires_grad) * 0.0
            self._consecutive_skips = 0

        return total_loss

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        # Resuming restores the optimizer's saved lr schedule. Rescale the whole schedule
        # so its peak becomes optimizer.lr from the config, so a run that keeps diverging
        # can continue at a lower rate. Lightning restores the optimizer from this same
        # dict afterwards, so editing it suffices.
        target = self.optimizer_cfg.lr
        ratios = []
        for state in checkpoint.get("optimizer_states", []):
            for group in state["param_groups"]:
                # The peak lives in max_lr under OneCycleLR (cosine_lr) and in
                # initial_lr under the LinearLR warm-up.
                peak = group.get("max_lr", group.get("initial_lr", group["lr"]))
                ratio = target / peak
                ratios.append(ratio)
                for key in ("lr", "initial_lr", "max_lr", "min_lr"):
                    if key in group:
                        group[key] *= ratio
        for scheduler in checkpoint.get("lr_schedulers", []):
            for key in ("base_lrs", "_last_lr"):
                if key in scheduler:
                    scheduler[key] = [v * r for v, r in zip(scheduler[key], ratios)]
        if ratios:
            print(f"resume: lr schedule rescaled by {ratios[0]:.3g} to peak {target}")

    def on_test_start(self) -> None:
        # Build the CDF tables the range coder needs. MeanScaleHyperpriorBase.compress
        # does not do this itself, and calling it without update() raises at runtime.
        codec = getattr(self.encoder, "gauss_feature_codec", None)
        if codec is not None:
            codec.update()

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1

        (scene,) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name

        # Render Gaussians.
        codec = getattr(self.encoder, "gauss_feature_codec", None)
        estimated_bits = None
        bitstream_bytes = codec_bytes = None
        if self.test_cfg.compress and codec is not None:
            # Real bitstream: encode to a file, then decode from that file only. Going
            # through the file is what keeps the decoder from quietly reading an
            # encoder-side tensor.
            from .encoder.compression.ops import (
                read_nfcgs_bitstream,
                write_nfcgs_bitstream,
            )

            with self.benchmarker.time("encoder"):
                payload = self.encoder.compress(batch["context"], self.global_step)
            bitstream_path = path / "bitstream" / f"{scene}.bin"
            bitstream_bytes = write_nfcgs_bitstream(bitstream_path, **payload)
            codec_bytes = sum(
                len(s) for group in payload["codec_out"]["strings"] for s in group
            )
            del payload
            with self.benchmarker.time("decompress"):
                gaussians = self.encoder.decompress(
                    **read_nfcgs_bitstream(bitstream_path, device=self.device)
                )
        else:
            with self.benchmarker.time("encoder"):
                enc_outputs = self.encoder(
                    batch["context"],
                    self.global_step,
                    deterministic=False,
                )
            gaussians = enc_outputs["gaussians"]
            estimated_bits = enc_outputs["estimated_bits"]
        with self.benchmarker.time("decoder", num_calls=v):
            output = self.decoder.forward(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode=None,
            )

        images_prob = output.color[0]
        rgb_gt = batch["target"]["image"][0]

        # Save images.
        if self.test_cfg.save_image:
            for index, color in zip(batch["target"]["index"][0], images_prob):
                save_image(color, path / scene / f"color/{index:0>6}.png")

        # save video
        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            save_video(
                [a for a in images_prob],
                path / "video" / f"{scene}_frame_{frame_str}.mp4",
            )

        # compute scores
        if self.test_cfg.compute_scores:
            if batch_idx < self.test_cfg.eval_time_skip_steps:
                self.time_skip_steps_dict["encoder"] += 1
                self.time_skip_steps_dict["decoder"] += v
            rgb = images_prob

            if f"psnr" not in self.test_step_outputs:
                self.test_step_outputs[f"psnr"] = []
            if f"ssim" not in self.test_step_outputs:
                self.test_step_outputs[f"ssim"] = []
            if f"lpips" not in self.test_step_outputs:
                self.test_step_outputs[f"lpips"] = []

            # Rate, so that a test run yields a whole R-D point rather than just the
            # distortion half.
            denominator = rate_denominator(gaussians.harmonics.shape)
            if estimated_bits is not None:
                # What the training loss optimises, not the size of a real bitstream.
                self.test_step_outputs.setdefault("bpp", []).append(
                    rate_loss(estimated_bits, gaussians.harmonics.shape).item()
                )
            if bitstream_bytes is not None:
                # codec_*: the y/z streams alone, comparable to the estimated bpp.
                # bitstream_*: the whole file, which also carries the camera
                # parameters and shapes the decoder needs.
                self.test_step_outputs.setdefault("codec_bpp", []).append(
                    codec_bytes * 8 / denominator
                )
                self.test_step_outputs.setdefault("codec_kib", []).append(
                    codec_bytes / 1024
                )
                self.test_step_outputs.setdefault("bitstream_bpp", []).append(
                    bitstream_bytes * 8 / denominator
                )
                self.test_step_outputs.setdefault("bitstream_kib", []).append(
                    bitstream_bytes / 1024
                )

            self.test_step_outputs[f"psnr"].append(
                compute_psnr(rgb_gt, rgb).mean().item()
            )
            self.test_step_outputs[f"ssim"].append(
                compute_ssim(rgb_gt, rgb).mean().item()
            )
            self.test_step_outputs[f"lpips"].append(
                compute_lpips(rgb_gt, rgb).mean().item()
            )

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        out_dir = self.test_cfg.output_path / name
        saved_scores = {}
        if self.test_cfg.compute_scores:
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.dump(out_dir / "benchmark.json")

            for metric_name, metric_scores in self.test_step_outputs.items():
                avg_scores = sum(metric_scores) / len(metric_scores)
                saved_scores[metric_name] = avg_scores
                print(metric_name, avg_scores)
                with (out_dir / f"scores_{metric_name}_all.json").open("w") as f:
                    json.dump(metric_scores, f)
                metric_scores.clear()

            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(self.time_skip_steps_dict[tag]) :]
                saved_scores[tag] = [len(times), np.mean(times)]
                print(
                    f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call"
                )
                self.time_skip_steps_dict[tag] = 0

            with (out_dir / f"scores_all_avg.json").open("w") as f:
                json.dump(saved_scores, f)
            self.benchmarker.clear_history()
        else:
            self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
            self.benchmarker.dump_memory(
                self.test_cfg.output_path / name / "peak_memory.json"
            )
            self.benchmarker.summarize()

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {[a[:20] for a in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1
        gaussians_softmax = self.encoder(
            batch["context"],
            self.global_step,
            deterministic=False,
        )["gaussians"]
        output_softmax = self.decoder.forward(
            gaussians_softmax,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        )
        rgb_softmax = output_softmax.color[0]

        # Compute validation metrics.
        rgb_gt = batch["target"]["image"][0]
        for tag, rgb in zip(
            ("val",), (rgb_softmax,)
        ):
            psnr = compute_psnr(rgb_gt, rgb).mean()
            self.log(f"val/psnr_{tag}", psnr)
            lpips = compute_lpips(rgb_gt, rgb).mean()
            self.log(f"val/lpips_{tag}", lpips)
            ssim = compute_ssim(rgb_gt, rgb).mean()
            self.log(f"val/ssim_{tag}", ssim)

        # Construct comparison image.
        comparison = hcat(
            add_label(vcat(*batch["context"]["image"][0]), "Context"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_softmax), "Target (Softmax)"),
        )
        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )

        # Render projections and construct projection image.
        projections = hcat(*render_projections(
                                gaussians_softmax,
                                256,
                                extra_label="(Softmax)",
                            )[0])
        self.logger.log_image(
            "projection",
            [prep_image(add_border(projections))],
            step=self.global_step,
        )

        # Draw cameras.
        cameras = hcat(*render_cameras(batch, 256))
        self.logger.log_image(
            "cameras", [prep_image(add_border(cameras))], step=self.global_step
        )

        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                batch["context"], self.global_step
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=self.global_step)

        # Run video validation step.
        self.render_video_interpolation(batch)
        self.render_video_wobble(batch)
        if self.train_cfg.extended_visualization:
            self.render_video_interpolation_exaggerated(batch)

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians_prob = self.encoder(batch["context"], self.global_step, False)[
            "gaussians"
        ]
        # gaussians_det = self.encoder(batch["context"], self.global_step, True)

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.view(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output_prob = self.decoder.forward(
            gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images_prob = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_prob.color[0], depth_map(output_prob.depth[0]))
        ]
        # output_det = self.decoder.forward(
        #     gaussians_det, extrinsics, intrinsics, near, far, (h, w), "depth"
        # )
        # images_det = [
        #     vcat(rgb, depth)
        #     for rgb, depth in zip(output_det.color[0], depth_map(output_det.depth[0]))
        # ]
        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Softmax"),
                    # add_label(image_det, "Deterministic"),
                )
            )
            for image_prob, _ in zip(images_prob, images_prob)
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        fps = 30
        key = f"video/{name}"

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log({key: wandb.Video(video[None], fps=fps, format="mp4")})
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            # Build the clip straight from the array: wandb.Video's _prepare_video and
            # _fps are private and are not present in every wandb version.
            clip = mpy.ImageSequenceClip(list(video.transpose(0, 2, 3, 1)), fps=fps)
            dir = LOG_PATH / key
            dir.mkdir(exist_ok=True, parents=True)
            clip.write_videofile(
                str(dir / f"{self.global_step:0>6}.mp4"), logger=None
            )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr)
        if self.optimizer_cfg.cosine_lr:
            warm_up = torch.optim.lr_scheduler.OneCycleLR(
                            optimizer, self.optimizer_cfg.lr,
                            self.trainer.max_steps + 10,
                            pct_start=0.01,
                            cycle_momentum=False,
                            anneal_strategy='cos',
                        )
        else:
            warm_up_steps = self.optimizer_cfg.warm_up_steps
            warm_up = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                1 / warm_up_steps,
                1,
                total_iters=warm_up_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warm_up,
                "interval": "step",
                "frequency": 1,
            },
        }
