"""Compile vector addition and save its IR without a GPU or CUDA driver."""

import argparse
from pathlib import Path

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.runtime import MockTensor, mock_device


ARCHITECTURES = {
    75: "Turing (T4)",
    80: "Ampere (A100, A30)",
    86: "Ampere (A40, RTX 3090)",
    87: "Ampere (Jetson AGX Orin)",
    89: "Ada (L40, RTX 4090)",
    90: "Hopper (H100, H200, GH200)",
    100: "Blackwell (B200, GB200)",
    103: "Blackwell (B300, GB300)",
    110: "Thor (Jetson T5000, T4000)",
    120: "Blackwell (RTX 5090, RTX PRO 6000)",
    121: "Blackwell (GB10 / DGX Spark)",
}


@triton.jit
def add_kernel(X, Y, Z, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(X + offsets, mask=mask, other=0)
    y = tl.load(Y + offsets, mask=mask, other=0)
    tl.store(Z + offsets, x + y, mask=mask)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Supported NVIDIA targets:\n" + "\n".join(
            f"  {arch:3d}  {name}" for arch, name in ARCHITECTURES.items()))
    parser.add_argument("--arch", type=int, choices=ARCHITECTURES, default=103,
                        help="CUDA compute capability (default: 103, GB300)")
    parser.add_argument("--output", type=Path, default=Path("debug_agent/kernel-ir"), help="directory for IR files")
    args = parser.parse_args()

    target = GPUTarget("cuda", args.arch, 32)
    with mock_device(target), triton.knobs.compilation.scope():
        # Regenerate optional pass dumps even when the disk cache is populated.
        triton.knobs.compilation.always_compile = True
        triton.knobs.compilation.store_binary_only = False
        tensor = MockTensor(tl.float32)
        kernel = add_kernel.warmup(tensor, tensor, tensor, 1000, BLOCK_SIZE=128, grid=(8, ))

    args.output.mkdir(parents=True, exist_ok=True)
    for stage in ("ttir", "ttgir", "llir", "ptx"):
        path = args.output / f"{kernel.name}.{stage}"
        path.write_text(kernel.asm[stage])
        print(path)


if __name__ == "__main__":
    main()
