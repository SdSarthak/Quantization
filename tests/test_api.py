def test_root_reports_device_and_variants(client):
    body = client.get("/").json()
    assert body["device"] == "cpu"
    assert body["available_models"] == ["original", "dynamic_quant"]
    assert "version" in body


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["device"] == "cpu"


def test_models_lists_support_per_device(client):
    body = client.get("/models").json()
    variants = {v["name"]: v for v in body["variants"]}
    assert variants["original"]["supported"] is True
    assert variants["int4"]["supported"] is False
    assert variants["int4"]["requires_device"] == "cuda"
    assert variants["int4"]["relative_weight_memory"] == 0.25
    # int8 against this host's FP32 baseline, not against FP16.
    assert variants["dynamic_quant"]["relative_weight_memory"] == 0.25


def test_generate_returns_only_the_continuation(client):
    response = client.post(
        "/generate/original",
        json={"prompt": "hello", "max_length": 4, "temperature": 0.0},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model_type"] == "original"
    assert body["generated_text"] == ["10 11 12 13"]
    assert body["prompt_tokens"] == 3
    assert body["generated_tokens"] == 4


def test_generate_loads_the_variant_on_demand(client):
    assert client.get("/health").json()["loaded_models"] == []
    client.post("/generate/original", json={"prompt": "hi", "max_length": 2})
    assert client.get("/health").json()["loaded_models"] == ["original"]


def test_generate_with_an_unknown_variant_is_404(client):
    response = client.post("/generate/int2", json={"prompt": "hi"})
    assert response.status_code == 404
    assert "int2" in response.json()["detail"]


def test_generate_with_a_gpu_variant_on_cpu_is_409(client):
    response = client.post("/generate/int4", json={"prompt": "hi"})
    assert response.status_code == 409
    assert "cuda" in response.json()["detail"]


def test_generate_rejects_out_of_range_parameters(client):
    assert (
        client.post(
            "/generate/original", json={"prompt": "hi", "temperature": 9}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/generate/original", json={"prompt": "hi", "max_length": 999999}
        ).status_code
        == 422
    )
    assert client.post("/generate/original", json={"prompt": ""}).status_code == 422
    assert (
        client.post(
            "/generate/original", json={"prompt": "hi", "top_p": 0}
        ).status_code
        == 422
    )


def test_benchmark_single_variant(client):
    response = client.get("/benchmark/original?iterations=1&max_new_tokens=4")
    assert response.status_code == 200
    body = response.json()
    assert body["model_type"] == "original"
    assert body["samples"] == 3  # three default prompts
    assert body["avg_latency"] > 0
    assert body["memory_usage"]["ram_total_gb"] > 0
    assert body["hardware_info"]["device"] == "cpu"


def test_benchmark_all_is_not_shadowed_by_the_parameterised_route(client):
    """Regression: /benchmark/{model_type} used to swallow the literal "all"."""
    response = client.get("/benchmark/all?iterations=1&max_new_tokens=2")
    assert response.status_code == 200
    body = response.json()
    assert set(body["results"]) == {"original", "dynamic_quant"}
    assert body["errors"] == {}
    assert "original" in body["comparison"]
    assert body["comparison"]["original"]["latency_speedup"] > 0


def test_benchmark_all_releases_the_variants_it_loaded(client):
    """Otherwise the sweep needs room for every variant at once."""
    client.get("/benchmark/all?iterations=1&max_new_tokens=2")
    assert client.get("/health").json()["loaded_models"] == []

    client.get("/benchmark/all?iterations=1&max_new_tokens=2&keep_loaded=true")
    assert client.get("/health").json()["loaded_models"] == [
        "dynamic_quant",
        "original",
    ]


def test_benchmark_reports_the_variant_weight_footprint(client):
    body = client.get("/benchmark/original?iterations=1&max_new_tokens=2").json()
    assert body["model_memory_mb"] > 0
    assert body["peak_gpu_memory_mb"] is None


def test_benchmark_all_compares_measured_memory(client):
    body = client.get("/benchmark/all?iterations=1&max_new_tokens=2").json()
    assert body["comparison"]["original"]["weight_memory_ratio"] == 1.0


def test_generate_rejects_a_seed_torch_cannot_represent(client):
    assert (
        client.post(
            "/generate/original", json={"prompt": "hi", "seed": 2**64}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/generate/original", json={"prompt": "hi", "seed": 2**64 - 1}
        ).status_code
        == 200
    )


def test_generate_accepts_unicode_prompts(client):
    response = client.post(
        "/generate/original",
        json={"prompt": "भारत के बारे में बताइए", "max_length": 2, "temperature": 0.0},
    )
    assert response.status_code == 200
    assert response.json()["generated_tokens"] == 2


def test_benchmark_unknown_variant_is_404(client):
    assert client.get("/benchmark/int2").status_code == 404


def test_benchmark_rejects_bad_query_parameters(client):
    assert client.get("/benchmark/original?iterations=0").status_code == 422
    assert client.get("/benchmark/original?iterations=1000").status_code == 422


def test_save_and_unload_roundtrip(client, settings):
    saved = client.post("/models/original/save")
    assert saved.status_code == 200
    assert (settings.quantized_model_path / "original").is_dir()

    assert client.delete("/models/original").status_code == 200
    assert client.delete("/models/original").status_code == 404


def test_system_info_shape(client):
    body = client.get("/system/info").json()
    assert body["device"] == "cpu"
    assert body["memory"]["total_gb"] > 0
    assert body["cpu"]["count"] >= 1
    assert "timestamp" in body


def test_openapi_schema_is_generated(client):
    schema = client.get("/openapi.json").json()
    assert "/benchmark/all" in schema["paths"]
    assert "/generate/{model_type}" in schema["paths"]
