"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    """Text generation parameters.

    ``max_length`` is the number of *new* tokens to generate, matching the
    documented behaviour of the API. Bounds are enforced here so bad input is
    rejected with a 422 instead of exploding inside the model.
    """

    prompt: str = Field(..., min_length=1, max_length=20000)
    max_length: int = Field(100, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, gt=0.0, le=1.0)
    num_return_sequences: int = Field(1, ge=1, le=10)
    #: Upper bound matches what ``torch.manual_seed`` accepts; anything larger
    #: raises deep inside torch and would surface as a 500.
    seed: Optional[int] = Field(None, ge=0, le=2**64 - 1)


class GenerationResponse(BaseModel):
    generated_text: List[str]
    inference_time: float
    tokens_per_second: float
    model_type: str
    prompt_tokens: int
    generated_tokens: int
    device: str


class MemoryUsage(BaseModel):
    ram_used_gb: float
    ram_total_gb: float
    ram_used_percent: float
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    gpu_temperature_c: Optional[float] = None


class HardwareInfo(BaseModel):
    device: str
    torch_version: str
    cpu_count: Optional[int] = None
    cpu_freq_mhz: Optional[float] = None
    gpu_name: Optional[str] = None
    cuda_version: Optional[str] = None


class BenchmarkResponse(BaseModel):
    model_type: str
    iterations: int
    samples: int
    avg_latency: float
    p50_latency: float
    p95_latency: float
    throughput: float
    total_tokens: int
    #: Weight footprint of this variant alone, which is what makes variants
    #: comparable; ``memory_usage`` below is whole-process.
    model_memory_mb: float
    peak_gpu_memory_mb: Optional[float] = None
    memory_usage: MemoryUsage
    hardware_info: HardwareInfo


class BenchmarkAllResponse(BaseModel):
    iterations: int
    results: Dict[str, BenchmarkResponse]
    #: Latency/throughput of each variant relative to the ``original`` baseline.
    comparison: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    errors: Dict[str, str] = Field(default_factory=dict)


class VariantInfo(BaseModel):
    name: str
    description: str
    bits: int
    requires_device: Optional[str] = None
    supported: bool
    loaded: bool
    relative_weight_memory: float


class ModelsResponse(BaseModel):
    device: str
    model_name: str
    loaded: List[str]
    variants: List[VariantInfo]


class HealthResponse(BaseModel):
    status: str
    device: str
    model_name: str
    loaded_models: List[str]
    tokenizer_ready: bool


class SystemInfoResponse(BaseModel):
    timestamp: str
    device: str
    model_name: str
    loaded_models: List[str]
    cpu: Dict[str, Optional[float]]
    memory: Dict[str, float]
    gpu: Optional[Dict[str, Optional[float]]] = None
    gpu_name: Optional[str] = None


class SaveResponse(BaseModel):
    model_type: str
    path: str
    size_mb: float
