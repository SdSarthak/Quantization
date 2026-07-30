import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import time
import psutil
from typing import Optional, List, Dict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
import json
from datetime import datetime

# Optional imports with fallbacks
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    print("GPUtil not available. GPU monitoring will be limited.")

try:
    from optimum.onnxruntime import ORTModelForCausalLM
    from optimum.exporters import TasksManager
    OPTIMUM_AVAILABLE = True
    print("Optimum library loaded successfully.")
except ImportError as e:
    OPTIMUM_AVAILABLE = False
    print(f"Optimum not available: {str(e)}")
    print("ONNX runtime features will be disabled.")
except Exception as e:
    OPTIMUM_AVAILABLE = False
    print(f"Error loading Optimum (likely protobuf version issue): {str(e)}")
    print("ONNX runtime features will be disabled.")
    print("Try: pip install protobuf==3.20.3 or set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python")

# Configuration
MODEL_NAME = "ai4bharat/Airavata"
CACHE_DIR = "./model_cache"
QUANTIZED_MODEL_PATH = "./quantized_models"

# Create directories
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(QUANTIZED_MODEL_PATH, exist_ok=True)

# FastAPI app
app = FastAPI(title="Airavata Model Service")

# Request/Response models
class GenerationRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    num_return_sequences: Optional[int] = 1

class GenerationResponse(BaseModel):
    generated_text: List[str]
    inference_time: float
    tokens_per_second: float
    model_type: str

class BenchmarkResponse(BaseModel):
    model_type: str
    avg_latency: float
    throughput: float
    memory_usage: Dict[str, float]
    hardware_info: Dict[str, str]

# Model Manager Class
class ModelManager:
    def __init__(self):
        self.models = {}
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def load_tokenizer(self):
        """Load tokenizer once for all model variants"""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                cache_dir=CACHE_DIR,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_original_model(self):
        """Load original FP32/FP16 model"""
        if "original" not in self.models:
            try:
                print("Loading original model...")
                self.models["original"] = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    cache_dir=CACHE_DIR,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    device_map="auto" if self.device.type == "cuda" else None,
                    trust_remote_code=True
                )
                if self.device.type == "cpu":
                    self.models["original"] = self.models["original"].to(self.device)
            except Exception as e:
                print(f"Failed to load original model: {str(e)}")
                raise  # Re-raise as this is critical for the service
    
    def load_int8_model(self):
        """Load INT8 quantized model using BitsAndBytes"""
        if "int8" not in self.models and self.device.type == "cuda":
            try:
                print("Loading INT8 quantized model...")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                )
                self.models["int8"] = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    cache_dir=CACHE_DIR,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"Failed to load INT8 model: {str(e)}")
    
    def load_int4_model(self):
        """Load INT4 quantized model using BitsAndBytes"""
        if "int4" not in self.models and self.device.type == "cuda":
            try:
                print("Loading INT4 quantized model...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                self.models["int4"] = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    cache_dir=CACHE_DIR,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"Failed to load INT4 model: {str(e)}")
    
    def quantize_dynamic(self):
        """Apply dynamic quantization for CPU inference"""
        if "dynamic_quant" not in self.models and self.device.type == "cpu":
            try:
                print("Applying dynamic quantization...")
                self.load_original_model()
                self.models["dynamic_quant"] = torch.quantization.quantize_dynamic(
                    self.models["original"],
                    {nn.Linear},
                    dtype=torch.qint8
                )
            except Exception as e:
                print(f"Failed to apply dynamic quantization: {str(e)}")
    
    def generate_text(self, model_type: str, request: GenerationRequest):
        """Generate text using specified model"""
        if model_type not in self.models:
            raise ValueError(f"Model type {model_type} not loaded")
        
        model = self.models[model_type]
        
        # Tokenize input
        inputs = self.tokenizer(
            request.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        if self.device.type == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generation parameters
        gen_kwargs = {
            "max_length": len(inputs["input_ids"][0]) + request.max_length,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "num_return_sequences": request.num_return_sequences,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": True
        }
        
        # Measure inference time
        start_time = time.time()
        
        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model.generate(**inputs, **gen_kwargs)
            else:
                outputs = model.generate(**inputs, **gen_kwargs)
        
        inference_time = time.time() - start_time
        
        # Decode outputs
        generated_texts = []
        total_tokens = 0
        
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text[len(request.prompt):].strip())
            total_tokens += len(output) - len(inputs["input_ids"][0])
        
        tokens_per_second = total_tokens / inference_time if inference_time > 0 else 0
        
        return generated_texts, inference_time, tokens_per_second
    
    def benchmark_model(self, model_type: str, num_iterations: int = 10):
        """Benchmark model performance"""
        if model_type not in self.models:
            raise ValueError(f"Model type {model_type} not loaded")
        
        # Test prompts of varying lengths
        test_prompts = [
            "The future of artificial intelligence is",
            "In the context of machine learning, transformer models have revolutionized",
            "Climate change is one of the most pressing issues of our time, requiring immediate action from governments and individuals alike to mitigate its effects on"
        ]
        
        latencies = []
        throughputs = []
        
        for _ in range(num_iterations):
            for prompt in test_prompts:
                request = GenerationRequest(
                    prompt=prompt,
                    max_length=50,
                    temperature=0.7
                )
                
                _, latency, tps = self.generate_text(model_type, request)
                latencies.append(latency)
                throughputs.append(tps)
        
        # Get memory usage
        memory_info = {}
        if self.device.type == "cuda":
            if GPUTIL_AVAILABLE:
                try:
                    gpu = GPUtil.getGPUs()[0]
                    memory_info["gpu_memory_used_mb"] = gpu.memoryUsed
                    memory_info["gpu_memory_total_mb"] = gpu.memoryTotal
                    memory_info["gpu_utilization"] = gpu.load * 100
                except (IndexError, Exception):
                    # Fallback to torch CUDA memory info
                    memory_info["gpu_memory_used_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
                    memory_info["gpu_memory_total_mb"] = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                    memory_info["gpu_utilization"] = "N/A"
            else:
                # Use torch CUDA memory info as fallback
                memory_info["gpu_memory_used_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
                memory_info["gpu_memory_total_mb"] = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                memory_info["gpu_utilization"] = "N/A"
        
        memory_info["ram_used_gb"] = psutil.virtual_memory().used / (1024**3)
        memory_info["ram_total_gb"] = psutil.virtual_memory().total / (1024**3)
        
        # Hardware info
        hardware_info = {
            "device": str(self.device),
            "cpu_count": psutil.cpu_count(),
            "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else "N/A"
        }
        
        if self.device.type == "cuda":
            hardware_info["gpu_name"] = torch.cuda.get_device_name(0)
            hardware_info["cuda_version"] = torch.version.cuda
        
        return {
            "model_type": model_type,
            "avg_latency": np.mean(latencies),
            "throughput": np.mean(throughputs),
            "memory_usage": memory_info,
            "hardware_info": hardware_info
        }

# Initialize model manager
model_manager = ModelManager()

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    try:
        print("Loading tokenizer...")
        model_manager.load_tokenizer()
        
        print("Loading models based on available hardware...")
        model_manager.load_original_model()
        
        if torch.cuda.is_available():
            print("CUDA available, loading quantized models...")
            model_manager.load_int8_model()
            model_manager.load_int4_model()
        else:
            print("CUDA not available, using CPU with dynamic quantization...")
            model_manager.quantize_dynamic()
        
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        # Don't raise the exception to allow the service to start
        # Users will get appropriate error messages when trying to use unavailable models

@app.get("/")
async def root():
    return {
        "message": "Airavata Model Service",
        "available_models": list(model_manager.models.keys()),
        "device": str(model_manager.device)
    }

@app.post("/generate/{model_type}", response_model=GenerationResponse)
async def generate(model_type: str, request: GenerationRequest):
    """Generate text using specified model type"""
    if model_type not in model_manager.models:
        raise HTTPException(
            status_code=404,
            detail=f"Model type '{model_type}' not found. Available: {list(model_manager.models.keys())}"
        )
    
    try:
        generated_texts, inference_time, tokens_per_second = await asyncio.get_event_loop().run_in_executor(
            model_manager.executor,
            model_manager.generate_text,
            model_type,
            request
        )
        
        return GenerationResponse(
            generated_text=generated_texts,
            inference_time=inference_time,
            tokens_per_second=tokens_per_second,
            model_type=model_type
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/benchmark/{model_type}", response_model=BenchmarkResponse)
async def benchmark(model_type: str, iterations: int = 10):
    """Benchmark specified model"""
    if model_type not in model_manager.models:
        raise HTTPException(
            status_code=404,
            detail=f"Model type '{model_type}' not found. Available: {list(model_manager.models.keys())}"
        )
    
    try:
        results = await asyncio.get_event_loop().run_in_executor(
            model_manager.executor,
            model_manager.benchmark_model,
            model_type,
            iterations
        )
        
        return BenchmarkResponse(**results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/benchmark/all")
async def benchmark_all(iterations: int = 10):
    """Benchmark all loaded models"""
    results = {}
    
    for model_type in model_manager.models.keys():
        try:
            results[model_type] = await asyncio.get_event_loop().run_in_executor(
                model_manager.executor,
                model_manager.benchmark_model,
                model_type,
                iterations
            )
        except Exception as e:
            results[model_type] = {"error": str(e)}
    
    return results

@app.get("/system/info")
async def system_info():
    """Get system information"""
    info = {
        "timestamp": datetime.now().isoformat(),
        "device": str(model_manager.device),
        "loaded_models": list(model_manager.models.keys()),
        "cpu": {
            "count": psutil.cpu_count(),
            "usage_percent": psutil.cpu_percent(interval=1),
            "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else "N/A"
        },
        "memory": {
            "total_gb": psutil.virtual_memory().total / (1024**3),
            "available_gb": psutil.virtual_memory().available / (1024**3),
            "used_percent": psutil.virtual_memory().percent
        }
    }
    
    if torch.cuda.is_available():
        info["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "memory_used_mb": torch.cuda.memory_allocated() / (1024 * 1024),
            "memory_total_mb": torch.cuda.get_device_properties(0).total_memory / (1024 * 1024),
            "utilization_percent": "N/A",  # Default value
            "temperature_c": "N/A"  # Default value
        }
        
        # Try to get additional GPU info if GPUtil is available
        if GPUTIL_AVAILABLE:
            try:
                gpu = GPUtil.getGPUs()[0]
                info["gpu"]["utilization_percent"] = gpu.load * 100
                info["gpu"]["temperature_c"] = gpu.temperature
                info["gpu"]["memory_used_mb"] = gpu.memoryUsed
                info["gpu"]["memory_total_mb"] = gpu.memoryTotal
            except (IndexError, Exception):
                pass  # Keep default values
    
    return info

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)