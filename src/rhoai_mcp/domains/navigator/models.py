"""Pydantic models for CUDA compatibility matrix."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RuntimeImageMapping(BaseModel):
    """Mapping of RHOAI serving runtime image to CUDA versions."""

    image: str = Field(..., description="Container image reference")
    cuda_version: list[str] = Field(..., description="Required CUDA toolkit versions")
    notes: str | None = Field(None, description="Additional notes about this mapping")


class CudaDriverMapping(BaseModel):
    """Mapping of CUDA toolkit version to minimum driver version."""

    cuda_version: list[str] = Field(..., description="CUDA toolkit version")
    min_driver_version: list[str] = Field(..., description="Minimum required driver version")


class GpuComputeMapping(BaseModel):
    """Mapping of GPU compute capability to supported CUDA versions."""

    compute_capability: str = Field(..., description="GPU compute capability (e.g., '8.0')")
    supported_cuda_versions: list[str] = Field(..., description="Supported CUDA versions")


class CudaCompatibilityMatrix(BaseModel):
    """Complete CUDA compatibility matrix with all three mappings."""

    runtime_images: list[RuntimeImageMapping] = Field(
        ..., alias="RHOAI serving runtime image", description="Runtime image to CUDA mappings"
    )
    cuda_drivers: list[CudaDriverMapping] = Field(
        ..., alias="CUDA toolkit version", description="CUDA to driver version mappings"
    )
    gpu_compute: list[GpuComputeMapping] = Field(
        ..., alias="GPU compute capability", description="GPU compute to CUDA version mappings"
    )

    model_config = ConfigDict(populate_by_name=True)
