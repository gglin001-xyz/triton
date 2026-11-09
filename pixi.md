# Pixi development environment

Use this environment for local builds and tests on Linux. Pixi supplies Python 3.14,
uv, CMake, Ninja, Clang, and mold. Python packages are installed into each Git
worktree's own `.venv`.

## Setup and build

First build the LLVM commit specified by `cmake/llvm-info.json` in the local
`llvm-triton` checkout, using its `scripts/llvm_download.sh` and
`scripts/llvm_build.sh`. A symlink at `./llvm-triton` is supported. Set
`LLVM_SYSPATH` when configuring to select a different prebuilt LLVM dependency.

Run from the Triton repository root:

```bash
pixi install
pixi run venv
pixi run configure
pixi run build
pixi run install
```

`venv` installs build/test dependencies and CPU PyTorch into `.venv`.
`configure` and `build` invoke the `llvm-triton` CMake preset directly, writing
native outputs to `build/llvm-triton` and `python/triton/_C`. `install` uses
`uv run --no-project` with `TRITON_SKIP_BUILD=1` to register the editable Python
package without invoking native build tools.

The CMake preset enables `TRITON_STABLE_ABI` (abi3, Python 3.12+) and defaults
to 64 parallel build jobs. To use more jobs, run
`pixi run build --parallel 128`.

When changing Python minor versions, move the existing `.venv` aside and rerun
`pixi run venv`. Clear the native build's CMake cache before configuring again.

After C++ changes, rerun `pixi run build` before testing. Python edits take
effect immediately. Test commands and the demo do not rebuild or reinstall.

## Local tests

Run tests through Pixi and the local `.venv`, from the repository root:

```bash
pixi run uv run --no-project --python .venv/bin/python python -m pytest -s --tb=short python/test/unit/language/test_frontend.py
pixi run uv run --no-project --python .venv/bin/python lit -v build/llvm-triton/test --filter tmem_layouts.mlir
```

Replace the pytest path with `file.py::test_name` to run a single test. This
environment includes CPU-only PyTorch; tests that require GPU execution need
compatible GPU dependencies and hardware.

## Offline IR dump demo

```bash
pixi run demo
```

See `examples/dump_kernel_ir.py` for using `triton.runtime.mock_device` with
`MockTensor` and `kernel.warmup()`. The implementation lives in
`python/triton/runtime/mock_device.py`. The demo compiles without a GPU or CUDA
driver and writes:

- TTIR, TTGIR, LLVM IR, and PTX to `build/runtime/demo`.
- Automatic compilation-stage dumps to `build/runtime/dump`.
- The MLIR pass trace to `build/runtime/passes.mlir`.

The default target is GB300 (SM103). Run
`pixi run uv run --no-project --python .venv/bin/python python examples/dump_kernel_ir.py --help`
to list supported targets, and use `--arch` to select one.

The demo uses existing `TRITON_KERNEL_DUMP`, `MLIR_ENABLE_DUMP`, and
`MLIR_DUMP_PATH` options. It validates compilation, not device execution.

## Concurrent Git worktrees

Run the setup commands separately in each worktree. Each has its own `.venv`,
`build/deps`, native build directory, and runtime cache/dumps under
`build/runtime`, so worktrees can install and test concurrently.

Worktrees without a local `llvm-triton` use the main worktree's prebuilt LLVM
dependency read-only. Branches with different LLVM pins must configure with
`LLVM_SYSPATH` pointing to their matching build. Do not share `.venv` or native
build directories between worktrees.
