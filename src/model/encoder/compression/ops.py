# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import struct
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
from PIL import Image
from torch import Tensor

# https://github.com/InterDigitalInc/CompressAI/blob/master/examples/codec.py


def pad(x, p=2**6):
    h, w = x.size(2), x.size(3)
    pad, _ = compute_padding(h, w, min_div=p)
    return F.pad(x, pad, mode="constant", value=0)


def crop(x, size):
    H, W = x.size(2), x.size(3)
    h, w = size
    _, unpad = compute_padding(h, w, out_h=H, out_w=W)
    return F.pad(x, unpad, mode="constant", value=0)


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def load_image(filepath: str) -> Image.Image:
    return Image.open(filepath).convert("RGB")


def img2torch(img: Image.Image) -> torch.Tensor:
    ToTensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    return ToTensor(img).unsqueeze(0)


def torch2img(x: torch.Tensor) -> Image.Image:
    return v2.ToPILImage()(x.clamp_(0, 1).squeeze())


def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return 0
    fd.write(struct.pack(fmt.format(len(values)), values))
    return len(values) * 1


def write_body(fd, shape, out_strings):
    bytes_cnt = write_uints(fd, (shape[0], shape[1], len(out_strings)))
    for stream_group in out_strings:
        bytes_cnt += write_uints(fd, (len(stream_group),))
        for stream in stream_group:
            bytes_cnt += write_uints(fd, (len(stream),))
            bytes_cnt += write_bytes(fd, stream)
    return bytes_cnt


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 1


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def read_body(fd):
    lstrings = []
    shape = read_uints(fd, 2)
    n_stream_groups = read_uints(fd, 1)[0]
    for _ in range(n_stream_groups):
        stream_group = []
        n_streams = read_uints(fd, 1)[0]
        for _ in range(n_streams):
            stream = read_bytes(fd, read_uints(fd, 1)[0])
            stream_group.append(stream)
        lstrings.append(stream_group)

    return lstrings, shape


NFCGS_BITSTREAM_MAGIC = b"NFCGSBS\x00"
NFCGS_BITSTREAM_VERSION = 1

_DTYPE_TO_CODE = {
    torch.float16: 0,
    torch.float32: 1,
    torch.float64: 2,
    torch.bfloat16: 3,
    torch.int8: 4,
    torch.uint8: 5,
    torch.int16: 6,
    torch.int32: 7,
    torch.int64: 8,
    torch.bool: 9,
}
_CODE_TO_DTYPE = {code: dtype for dtype, code in _DTYPE_TO_CODE.items()}


def _read_exact(fd, n):
    data = fd.read(n)
    if len(data) != n:
        raise EOFError(f"Unexpected end of NFC-GS bitstream ({len(data)}/{n} bytes).")
    return data


def _write_int_tuple(fd, values):
    values = tuple(int(value) for value in values)
    write_uints(fd, (len(values),))
    for value in values:
        fd.write(struct.pack(">q", value))


def _read_int_tuple(fd):
    length = read_uints(fd, 1)[0]
    return tuple(struct.unpack(">q", _read_exact(fd, 8))[0] for _ in range(length))


def _write_tensor(fd, tensor):
    tensor = tensor.detach().cpu().contiguous()
    try:
        dtype_code = _DTYPE_TO_CODE[tensor.dtype]
    except KeyError as error:
        raise TypeError(f"Unsupported tensor dtype: {tensor.dtype}") from error

    write_uchars(fd, (dtype_code,))
    _write_int_tuple(fd, tensor.shape)
    payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    fd.write(struct.pack(">Q", len(payload)))
    fd.write(payload)


def _read_tensor(fd, device):
    dtype_code = read_uchars(fd, 1)[0]
    try:
        dtype = _CODE_TO_DTYPE[dtype_code]
    except KeyError as error:
        raise ValueError(f"Unknown tensor dtype code: {dtype_code}") from error

    shape = _read_int_tuple(fd)
    payload_size = struct.unpack(">Q", _read_exact(fd, 8))[0]
    expected_size = int(torch.empty((), dtype=dtype).element_size())
    for dimension in shape:
        if dimension < 0:
            raise ValueError(f"Negative tensor dimension in bitstream: {shape}")
        expected_size *= dimension
    if payload_size != expected_size:
        raise ValueError(
            f"Invalid tensor payload size: expected {expected_size}, got {payload_size}."
        )

    payload = bytearray(_read_exact(fd, payload_size))
    tensor = torch.frombuffer(payload, dtype=dtype).clone().reshape(shape)
    return tensor.to(device)


def write_nfcgs_bitstream(
    filepath,
    *,
    resolution,
    codec_out,
    extrinsics,
    intrinsics,
    feature_shape,
    infeature_shape,
    near,
    far,
    global_step,
    keep_ratio=None,
):
    # Write every data-dependent input required by Encoder.decompress.
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("wb") as fd:
        fd.write(NFCGS_BITSTREAM_MAGIC)
        write_uints(fd, (NFCGS_BITSTREAM_VERSION,))
        _write_int_tuple(fd, resolution)
        fd.write(struct.pack(">q", int(global_step)))
        write_uchars(fd, (keep_ratio is not None,))
        if keep_ratio is not None:
            fd.write(struct.pack(">d", float(keep_ratio)))
        _write_int_tuple(fd, feature_shape)
        _write_int_tuple(fd, infeature_shape)
        _write_tensor(fd, extrinsics)
        _write_tensor(fd, intrinsics)
        _write_tensor(fd, near)
        _write_tensor(fd, far)
        write_body(fd, codec_out["shape"], codec_out["strings"])
    return filesize(filepath)


def read_nfcgs_bitstream(filepath, *, device):
    # Read a self-contained NFC-GS bitstream for decoder-only inference.
    filepath = Path(filepath)
    with filepath.open("rb") as fd:
        magic = _read_exact(fd, len(NFCGS_BITSTREAM_MAGIC))
        if magic != NFCGS_BITSTREAM_MAGIC:
            raise ValueError(f"Invalid NFC-GS bitstream magic in {filepath}.")
        version = read_uints(fd, 1)[0]
        if version != NFCGS_BITSTREAM_VERSION:
            raise ValueError(
                f"Unsupported NFC-GS bitstream version {version}; "
                f"expected {NFCGS_BITSTREAM_VERSION}."
            )

        resolution = _read_int_tuple(fd)
        if len(resolution) != 2:
            raise ValueError(f"Invalid resolution in bitstream: {resolution}")
        global_step = struct.unpack(">q", _read_exact(fd, 8))[0]
        has_keep_ratio = read_uchars(fd, 1)[0]
        if has_keep_ratio not in (0, 1):
            raise ValueError(f"Invalid keep-ratio flag: {has_keep_ratio}")
        keep_ratio = (
            struct.unpack(">d", _read_exact(fd, 8))[0] if has_keep_ratio else None
        )
        feature_shape = _read_int_tuple(fd)
        infeature_shape = _read_int_tuple(fd)
        extrinsics = _read_tensor(fd, device)
        intrinsics = _read_tensor(fd, device)
        near = _read_tensor(fd, device)
        far = _read_tensor(fd, device)
        strings, shape = read_body(fd)
        if fd.read(1):
            raise ValueError(f"Trailing data found in NFC-GS bitstream {filepath}.")

    return {
        "resolution": resolution,
        "global_step": global_step,
        "keep_ratio": keep_ratio,
        "feature_shape": feature_shape,
        "infeature_shape": infeature_shape,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "near": near,
        "far": far,
        "codec_out": {"strings": strings, "shape": shape},
    }


# https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/ops/ops.py
def quantize_ste(x: Tensor) -> Tensor:
    """
    Rounding with non-zero gradients. Gradients are approximated by replacing
    the derivative by the identity function.

    Used in `"Lossy Image Compression with Compressive Autoencoders"
    <https://arxiv.org/abs/1703.00395>`_

    .. note::

        Implemented with the pytorch `detach()` reparametrization trick:

        `x_round = x_round - x.detach() + x`
    """
    return (torch.round(x) - x).detach() + x


def compute_padding(in_h: int, in_w: int, *, out_h=None, out_w=None, min_div=1):
    """Returns tuples for padding and unpadding.

    Args:
        in_h: Input height.
        in_w: Input width.
        out_h: Output height.
        out_w: Output width.
        min_div: Length that output dimensions should be divisible by.
    """
    if out_h is None:
        out_h = (in_h + min_div - 1) // min_div * min_div
    if out_w is None:
        out_w = (in_w + min_div - 1) // min_div * min_div

    if out_h % min_div != 0 or out_w % min_div != 0:
        raise ValueError(
            f"Padded output height and width are not divisible by min_div={min_div}."
        )

    left = (out_w - in_w) // 2
    right = out_w - in_w - left
    top = (out_h - in_h) // 2
    bottom = out_h - in_h - top

    pad = (left, right, top, bottom)
    unpad = (-left, -right, -top, -bottom)

    return pad, unpad


# https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/ops/bound_ops.py
def lower_bound_fwd(x: Tensor, bound: Tensor) -> Tensor:
    return torch.max(x, bound)


def lower_bound_bwd(x: Tensor, bound: Tensor, grad_output: Tensor):
    pass_through_if = (x >= bound) | (grad_output < 0)
    return pass_through_if * grad_output, None


class LowerBoundFunction(torch.autograd.Function):
    """Autograd function for the `LowerBound` operator."""

    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x, bound)
        return lower_bound_fwd(x, bound)

    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        return lower_bound_bwd(x, bound, grad_output)


class LowerBound(nn.Module):
    """Lower bound operator, computes `torch.max(x, bound)` with a custom
    gradient.

    The derivative is replaced by the identity function when `x` is moved
    towards the `bound`, otherwise the gradient is kept to zero.
    """

    bound: Tensor

    def __init__(self, bound: float):
        super().__init__()
        self.register_buffer("bound", torch.Tensor([float(bound)]))

    @torch.jit.unused
    def lower_bound(self, x):
        return LowerBoundFunction.apply(x, self.bound)

    def forward(self, x):
        if torch.jit.is_scripting():
            return torch.max(x, self.bound)
        return self.lower_bound(x)


# https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/ops/parametrizers.py
class NonNegativeParametrizer(nn.Module):
    """
    Non negative reparametrization.

    Used for stability during training.
    """

    pedestal: Tensor

    def __init__(self, minimum: float = 0, reparam_offset: float = 2**-18):
        super().__init__()

        self.minimum = float(minimum)
        self.reparam_offset = float(reparam_offset)

        pedestal = self.reparam_offset**2
        self.register_buffer("pedestal", torch.Tensor([pedestal]))
        bound = (self.minimum + self.reparam_offset**2) ** 0.5
        self.lower_bound = LowerBound(bound)

    def init(self, x: Tensor) -> Tensor:
        return torch.sqrt(torch.max(x + self.pedestal, self.pedestal))

    def forward(self, x: Tensor) -> Tensor:
        out = self.lower_bound(x)
        out = out**2 - self.pedestal
        return out


# https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/layers/gdn.py
class GDN(nn.Module):
    r"""Generalized Divisive Normalization layer.

    Introduced in `"Density Modeling of Images Using a Generalized Normalization
    Transformation" <https://arxiv.org/abs/1511.06281>`_,
    by Balle Johannes, Valero Laparra, and Eero P. Simoncelli, (2016).

    .. math::

       y[i] = \frac{x[i]}{\sqrt{\beta[i] + \sum_j(\gamma[j, i] * x[j]^2)}}

    """

    def __init__(
        self,
        in_channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
        max_gain: float | None = None,
    ):
        super().__init__()

        beta_min = float(beta_min)
        gamma_init = float(gamma_init)
        self.inverse = bool(inverse)
        self.max_gain = None if max_gain is None else float(max_gain)

        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)

        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(in_channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)

    def forward(self, x: Tensor) -> Tensor:
        _, C, _, _ = x.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(x**2, gamma, beta)

        if self.inverse:
            if self.max_gain is not None:
                # IGDN's gain sqrt(norm) grows with |x|; chained in g_s, a modest drift
                # of the latent got amplified 1e7x and blew up training. Cap the gain.
                norm = norm.clamp(max=self.max_gain ** 2)
            norm = torch.sqrt(norm)
        else:
            norm = torch.rsqrt(norm)

        out = x * norm

        return out

        
def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

def ste_round(x: torch.Tensor) -> torch.Tensor:
    # Straight-Through Estimator for rounding
    return torch.round(x) - x.detach() + x
