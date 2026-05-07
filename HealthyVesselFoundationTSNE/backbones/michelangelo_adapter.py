"""Michelangelo adapter.

Michelangelo-style repos are less standardized than OpenShape/Uni3D. This
adapter supports three loading modes:

1. `factory_callable`: a user-provided callable returning an initialized model.
2. `config_path`: OmegaConf config with a `model` block and an
   `instantiate_from_config` helper in the local repo.
3. `model_module` + `model_class`: directly import and instantiate a class.

The forward path then tries common encoder names such as `encode_shape_embed`,
`encode`, and `encode_pc`.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import numpy as np

from .common import (
    add_repo_path,
    call_first_working_model_method,
    extract_tensor,
    import_torch,
    load_checkpoint_into_model,
    resolve_callable,
    tensor_to_numpy,
    xyznormal_numpy,
)


_MODEL_CACHE: Dict[Tuple[str, str, str], Any] = {}


def _instantiate_from_config(config: Mapping[str, Any]) -> Any:
    config_path = config.get("config_path")
    if not config_path:
        raise ValueError("config_path is required for config-based Michelangelo loading")

    try:
        from omegaconf import OmegaConf
    except Exception as exc:
        raise ImportError("OmegaConf is required for config_path-based Michelangelo loading") from exc

    cfg = OmegaConf.load(str(config_path))
    model_cfg = cfg.get("model", cfg)

    helper_specs = [
        "michelangelo.utils.misc:instantiate_from_config",
        "michelangelo.utils.config:instantiate_from_config",
        "utils.misc:instantiate_from_config",
        "utils.config:instantiate_from_config",
    ]
    last_error = None
    for spec in helper_specs:
        try:
            helper = resolve_callable(spec)
            return helper(model_cfg)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not instantiate Michelangelo config. Last error: {last_error}")


def _instantiate_from_class(config: Mapping[str, Any]) -> Any:
    module_name = config.get("model_module")
    class_name = config.get("model_class")
    if not module_name or not class_name:
        raise ValueError("model_module and model_class are required for class-based Michelangelo loading")
    import importlib

    module = importlib.import_module(str(module_name))
    cls = getattr(module, str(class_name))
    kwargs = dict(config.get("model_kwargs", {}) or {})
    return cls(**kwargs)


def _load_michelangelo_model(config: Mapping[str, Any], device: str):
    add_repo_path(config.get("repo_path"))
    torch, torch_device = import_torch(device)

    identity = (
        str(config.get("factory_callable") or config.get("config_path") or config.get("model_module") or "none"),
        str(config.get("checkpoint") or ""),
        str(torch_device),
    )
    if identity in _MODEL_CACHE:
        return _MODEL_CACHE[identity], torch, torch_device

    if config.get("factory_callable"):
        factory = resolve_callable(str(config["factory_callable"]))
        try:
            model = factory(config, device)
        except TypeError:
            model = factory(config)
    elif config.get("config_path"):
        model = _instantiate_from_config(config)
    else:
        model = _instantiate_from_class(config)

    load_checkpoint_into_model(model, config.get("checkpoint"), strict=bool(config.get("strict_checkpoint", False)))
    model.to(torch_device)
    model.eval()
    _MODEL_CACHE[identity] = model
    return model, torch, torch_device


def embed_point_clouds(points: np.ndarray, config: Mapping[str, Any], device: str = "cuda") -> np.ndarray:
    model, torch, torch_device = _load_michelangelo_model(config, device)
    pc_np = xyznormal_numpy(points, config)
    surface = torch.from_numpy(pc_np).to(torch_device)
    methods = list(config.get("encode_methods", ["encode_shape_embed", "encode", "encode_pc", "forward"]))
    with torch.no_grad():
        output = call_first_working_model_method(model, methods, surface, config)
    tensor = extract_tensor(output, preferred_keys=config.get("output_keys"))
    return tensor_to_numpy(tensor, pool=str(config.get("pool", "auto")))
