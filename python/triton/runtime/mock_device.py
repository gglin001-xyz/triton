"""Compile kernels without initializing a physical device driver."""

from contextlib import contextmanager

from ..backends.driver import DriverBase
from ..backends.compiler import GPUTarget
from .driver import driver


class _MockDriver(DriverBase):
    """Supply a compilation target without loading a device runtime."""

    def __init__(self, target: GPUTarget):
        self.target = target

    @classmethod
    def is_active(cls):
        return False

    def get_current_target(self):
        return self.target

    def get_current_device(self):
        # JIT caches are indexed by device, so mock targets need distinct keys
        # from real device ordinals and from other mock architectures.
        return ("mock", self.target)

    def get_current_stream(self, device):
        return None

    def get_active_torch_device(self):
        import torch
        return torch.device("cpu")

    def map_python_to_cpp_type(self, ty: str) -> str:
        raise RuntimeError("Mock devices only support compilation; use kernel.warmup().")

    def get_benchmarker(self):
        raise RuntimeError("Mock devices cannot benchmark kernels.")

    @staticmethod
    def launcher_cls(*args, **kwargs):
        raise RuntimeError("Mock devices cannot launch kernels; use kernel.warmup().")


@contextmanager
def mock_device(target: GPUTarget):
    """Compile kernels with ``warmup`` without a GPU or device runtime.

    ``target`` selects the real compiler backend and architecture. Backend
    compilation tools, such as CUDA's ``ptxas``, are still required. Use
    :class:`MockTensor` for tensor arguments and inspect the returned kernel's
    ``asm`` dictionary. Kernel execution and benchmarking are unsupported.

    Like ``driver.set_active``, this changes the process-wide active driver;
    do not use it concurrently with other driver changes or kernel launches.
    """
    if not isinstance(target, GPUTarget):
        raise TypeError("target must be a GPUTarget")
    previous = driver._active
    driver.set_active(_MockDriver(target))
    try:
        yield driver.active
    finally:
        # Do not access driver.default here: the previous driver may never
        # have been initialized, especially on a machine without a GPU.
        driver._active = previous
