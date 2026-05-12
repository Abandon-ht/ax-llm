import argparse
import importlib.machinery
import json
import os
import random
import struct
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
from axengine import InferenceSession

DEFAULT_QWEN_TTS_ROOT = Path("Qwen3-TTS")
DEFAULT_HF_MODEL_PATH = Path("../Qwen/Qwen3-TTS-12Hz-0.6B-Base")
DEFAULT_TALKER_AXMODEL_DIR = Path(
    "../Qwen/Qwen3-TTS-12Hz-0.6B-Base-AX650/talker"
)
DEFAULT_CODE_PREDICTOR_AXMODEL_DIR = Path(
    "../Qwen/Qwen3-TTS-12Hz-0.6B-Base-AX650/code-predictor"
)
DEFAULT_OUTPUT_DIR =  "qwen3_tts_axmodel_outputs"


class CfgLoader:
    def __init__(self, *_args, **_kwargs):
        self.cfg = None

    def get_cfg(self):
        return self.cfg


class ModelAttrsLoader:
    def __init__(self, *_args, **_kwargs):
        pass


def build_layer(*_args, **_kwargs):
    raise NotImplementedError("qwen3_tts_axengine_replace_talker_code_predictor_demo is an inference-only demo")


def build_post_layer(*_args, **_kwargs):
    raise NotImplementedError("qwen3_tts_axengine_replace_talker_code_predictor_demo is an inference-only demo")


def _prepend_path(path: Path):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _install_import_stubs():
    if "cv2" not in sys.modules:
        cv2 = types.ModuleType("cv2")
        cv2.__spec__ = importlib.machinery.ModuleSpec("cv2", loader=None)
        for name, value in {
            "INTER_NEAREST": 0,
            "INTER_LINEAR": 1,
            "INTER_AREA": 3,
            "BORDER_CONSTANT": 0,
            "BORDER_REFLECT_101": 4,
        }.items():
            setattr(cv2, name, value)
        sys.modules["cv2"] = cv2

    if "torchvision" not in sys.modules:
        torchvision = types.ModuleType("torchvision")
        torchvision.__spec__ = importlib.machinery.ModuleSpec("torchvision", loader=None, is_package=True)
        torchvision.__path__ = []
        torchvision_ops = types.ModuleType("torchvision.ops")
        torchvision_ops.__spec__ = importlib.machinery.ModuleSpec("torchvision.ops", loader=None)

        def _stub_roi_align(*_args, **_kwargs):
            raise RuntimeError("torchvision.ops.roi_align stub is not available in this Qwen3-TTS demo")

        torchvision_ops.roi_align = _stub_roi_align
        torchvision.ops = torchvision_ops
        sys.modules["torchvision"] = torchvision
        sys.modules["torchvision.ops"] = torchvision_ops

    if "IPython" not in sys.modules:
        ipython = types.ModuleType("IPython")
        ipython.__spec__ = importlib.machinery.ModuleSpec("IPython", loader=None)
        ipython.embed = lambda *_args, **_kwargs: None
        sys.modules["IPython"] = ipython


def _patch_onnx_tensorproto():
    import onnx

    for name, value in {"FLOAT4E2M1": 23, "FLOAT8E8M0": 24}.items():
        if not hasattr(onnx.TensorProto, name):
            setattr(onnx.TensorProto, name, value)


def _patch_numba_cache():
    try:
        import numba
    except Exception:
        return

    if getattr(numba.jit, "_qwen3_tts_no_cache_patch", False):
        return

    original_jit = numba.jit

    def jit_no_cache(*args, **kwargs):
        kwargs.pop("cache", None)
        return original_jit(*args, **kwargs)

    jit_no_cache._qwen3_tts_no_cache_patch = True
    numba.jit = jit_no_cache


def _patch_transformers_check_model_inputs():
    try:
        import inspect
        from transformers.utils import generic as transformers_generic
    except Exception:
        return

    original = getattr(transformers_generic, "check_model_inputs", None)
    if original is None or getattr(original, "_qwen3_tts_compat_patch", False):
        return

    try:
        params = list(inspect.signature(original).parameters.values())
    except Exception:
        params = []
    first_required = bool(
        params
        and params[0].default is inspect.Signature.empty
        and params[0].kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    if not first_required:
        return

    def check_model_inputs_compat(func=None, *args, **kwargs):
        if callable(func) and not args and not kwargs:
            return original(func)

        def decorator(real_func):
            return original(real_func)

        return decorator

    check_model_inputs_compat._qwen3_tts_compat_patch = True
    transformers_generic.check_model_inputs = check_model_inputs_compat


def _prepare_import_environment(args):
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    os.environ.setdefault("JOBLIB_TEMP_FOLDER", "/tmp/joblib")
    _prepend_path(args.qwen_tts_root)
    _patch_onnx_tensorproto()
    _patch_numba_cache()
    _patch_transformers_check_model_inputs()


def _fp32_to_bf16_raw(arr: np.ndarray) -> bytes:
    """Convert float32 numpy array to bfloat16 raw bytes (C++ compatible)."""
    arr = np.asarray(arr, dtype=np.float32)
    u32 = arr.view(np.uint32)
    bf16 = (u32 >> 16).astype(np.uint16)
    return bf16.tobytes()


def _save_prefill_embeds_bin(path: Path, tensor) -> None:
    """Save prefill embeddings as raw bfloat16 [S, hidden_size] without header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np_arr = tensor.detach().cpu().squeeze(0).float().numpy()  # [S, H] float32
    path.write_bytes(_fp32_to_bf16_raw(np_arr))


def _save_meta_json(path: Path, S: int, hidden_size: int, audio_token_id: int, trailing_start: int, streaming: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "S": S,
        "hidden_size": hidden_size,
        "audio_token_id": audio_token_id,
        "trailing_start": trailing_start,
        "streaming": streaming,
    }
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def _save_tts_pad_vec_bin(path: Path, vec) -> None:
    """Save tts_pad_vec as int32 hidden_size + fp32 data (C++ compatible)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np_vec = vec.detach().cpu().squeeze().float().numpy()
    hidden_size = int(np_vec.shape[0])
    with open(path, "wb") as f:
        f.write(struct.pack("i", hidden_size))
        f.write(np_vec.astype(np.float32).tobytes())


def _numpy_dtype(dtype: str):
    key = dtype.lower()
    if key in {"bf16", "bfloat16"}:
        try:
            from ml_dtypes import bfloat16

            return bfloat16
        except Exception as exc:
            raise RuntimeError("ml_dtypes.bfloat16 is required for bf16 axengine inputs") from exc
    if key in {"fp16", "float16"}:
        return np.float16
    if key in {"fp32", "float32"}:
        return np.float32
    raise ValueError(f"unsupported numpy dtype: {dtype}")


# def _load_axengine_inference_session():
#     from axengine import InferenceSession

#     return InferenceSession


def _tensor_shape(value_info) -> Tuple[int, ...]:
    shape = []
    for dim in value_info.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value"):
            raise RuntimeError(f"dynamic axmodel shape is not supported for {value_info.name}: {dim}")
        shape.append(int(dim.dim_value))
    return tuple(shape)


def _load_axmodel_io(model_path: Path):
    import onnx

    model = onnx.load(str(model_path))
    initializer_names = {tensor.name for tensor in model.graph.initializer}
    input_shapes = {
        value.name: _tensor_shape(value) for value in model.graph.input if value.name not in initializer_names
    }
    output_names = [value.name for value in model.graph.output]
    return input_shapes, output_names


def _shape_group_output_names(
    shape_group: Optional[int],
    base_names: Tuple[str, ...] = ("K_cache_out", "V_cache_out", "output"),
):
    if shape_group is None or int(shape_group) == 0:
        return list(base_names)
    return [f"{name}_{int(shape_group)}" for name in base_names]


class AxEngineSession:
    def __init__(self, model_path: Path, device: Optional[str] = None):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"missing axmodel: {self.model_path}")
        self.input_shapes, self.output_names = _load_axmodel_io(self.model_path)
        session_cls = InferenceSession
        print(f"[AxEngineSession] model_path={self.model_path}, device={device}")
        if device is not None:
            try:
                dev_id = int(device)
                self.session = session_cls(
                    str(self.model_path),
                    provider_options=[{"device_id": dev_id}]
                )
            except Exception as exc:
                print(f"[WARNING] failed to create session with device={device}: {exc}. Fallback to default device.")
                self.session = session_cls(str(self.model_path))
        else:
            self.session = session_cls(str(self.model_path))

    def input_shape(self, name: str) -> Tuple[int, ...]:
        if name not in self.input_shapes:
            raise KeyError(f"{self.model_path} missing input {name!r}; inputs={list(self.input_shapes)}")
        return self.input_shapes[name]

    def run(
        self,
        input_feed: Dict[str, object],
        shape_group: Optional[int] = None,
        expected_output_names: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        feed = {name: np.ascontiguousarray(value) for name, value in input_feed.items()}
        try:
            if shape_group is None:
                raw_outputs = self.session.run(None, feed)
            else:
                raw_outputs = self.session.run(None, feed, shape_group=int(shape_group))
        except TypeError as exc:
            if shape_group is None or "shape_group" not in str(exc):
                raise
            raw_outputs = self.session.run(None, feed)
        return self._normalize_outputs(raw_outputs, shape_group, expected_output_names)

    def _normalize_outputs(
        self,
        raw_outputs,
        shape_group: Optional[int],
        expected_output_names: Optional[List[str]],
    ) -> Dict[str, np.ndarray]:
        if isinstance(raw_outputs, dict):
            return raw_outputs

        if not isinstance(raw_outputs, (list, tuple)):
            raise TypeError(f"unsupported axengine output type: {type(raw_outputs)} from {self.model_path}")

        if expected_output_names is not None and len(raw_outputs) == len(expected_output_names):
            names = expected_output_names
        elif shape_group is not None and len(raw_outputs) == 3:
            names = _shape_group_output_names(shape_group)
        elif len(raw_outputs) == len(self.output_names):
            names = self.output_names
        elif expected_output_names is not None:
            names = expected_output_names[: len(raw_outputs)]
        else:
            names = self.output_names[: len(raw_outputs)]

        if len(names) != len(raw_outputs):
            raise RuntimeError(
                f"cannot map {len(raw_outputs)} axengine outputs for {self.model_path}; "
                f"expected={expected_output_names}, all={self.output_names}"
            )
        return dict(zip(names, raw_outputs))


class NumpyOnnxLinearSession:
    def __init__(self, model_path: Path):
        import onnx
        from onnx import numpy_helper

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"missing ONNX lm_head: {self.model_path}")

        model = onnx.load(str(self.model_path), load_external_data=True)
        initializer_names = {tensor.name for tensor in model.graph.initializer}
        self.input_shapes = {
            value.name: _tensor_shape(value) for value in model.graph.input if value.name not in initializer_names
        }
        self.output_names = [value.name for value in model.graph.output]
        if len(self.input_shapes) != 1:
            raise RuntimeError(f"expected one ONNX lm_head input, got {list(self.input_shapes)} from {self.model_path}")
        if len(self.output_names) != 1:
            raise RuntimeError(f"expected one ONNX lm_head output, got {self.output_names} from {self.model_path}")

        initializers = {tensor.name: numpy_helper.to_array(tensor).astype(np.float32, copy=False) for tensor in model.graph.initializer}
        if len(model.graph.node) != 1 or model.graph.node[0].op_type != "MatMul":
            raise RuntimeError(f"only single MatMul ONNX lm_head is supported: {self.model_path}")
        node = model.graph.node[0]
        if len(node.input) != 2:
            raise RuntimeError(f"unexpected MatMul inputs for {self.model_path}: {list(node.input)}")
        weight_names = [name for name in node.input if name in initializers]
        input_names = [name for name in node.input if name not in initializers]
        if len(weight_names) != 1 or len(input_names) != 1:
            raise RuntimeError(f"cannot identify MatMul input/weight for {self.model_path}: {list(node.input)}")
        self.input_name = input_names[0]
        self.output_name = self.output_names[0]
        self.weight = initializers[weight_names[0]]
        if self.weight.ndim != 2:
            raise RuntimeError(f"unexpected ONNX lm_head weight shape={self.weight.shape} from {self.model_path}")

    def input_shape(self, name: str) -> Tuple[int, ...]:
        if name not in self.input_shapes:
            raise KeyError(f"{self.model_path} missing input {name!r}; inputs={list(self.input_shapes)}")
        return self.input_shapes[name]

    def run(self, input_feed: Dict[str, object]) -> Dict[str, np.ndarray]:
        if self.input_name not in input_feed:
            raise KeyError(f"{self.model_path} expected input {self.input_name!r}; got={list(input_feed)}")
        data = np.asarray(input_feed[self.input_name], dtype=np.float32)
        output = np.matmul(data, self.weight).astype(np.float32, copy=False)
        return {self.output_name: output}


def _build_decode_mask_cache(kv_cache_len: int, mask_dtype):
    row_ids = np.arange(kv_cache_len + 1, dtype=np.int32)[:, None]
    col_ids = np.arange(kv_cache_len + 1, dtype=np.int32)[None, :]
    mask_2d = np.where(col_ids < row_ids, 0.0, -65536.0).astype(np.float32)
    mask_2d[:, -1] = 0.0
    return mask_2d.astype(mask_dtype).reshape((kv_cache_len + 1, 1, 1, kv_cache_len + 1))


def _position_ids_to_static_indices(position_ids, valid_len: int, prefill_len: int):
    if position_ids is None:
        pos = np.arange(valid_len, dtype=np.uint32).reshape(1, valid_len)
        pos = np.repeat(pos, 3, axis=0)
    else:
        pos = position_ids.detach().cpu().numpy()
        if pos.ndim == 3 and pos.shape[0] == 4:
            pos = pos[1:]
        if pos.ndim == 3 and pos.shape[1] == 1:
            pos = pos[:, 0, :]
        elif pos.ndim == 2 and pos.shape[0] == 1:
            pos = np.repeat(pos, 3, axis=0)
        elif pos.ndim == 2 and pos.shape[0] == 3:
            pass
        else:
            raise AssertionError(f"unexpected position_ids shape: {pos.shape}")
        pos = pos[:, -valid_len:]

    indices = np.ones((3, prefill_len), dtype=np.uint32)
    indices[:, :valid_len] = pos.astype(np.uint32)
    return indices


def _kv_dim_from_config(config):
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = int(config.hidden_size) // int(config.num_attention_heads)
    return int(config.num_key_value_heads) * int(head_dim)


@dataclass
class StaticTalkerState:
    k_caches: list
    v_caches: list
    current_len: int
    prompt_len: int
    max_cache_len: int

    def get_seq_length(self, *_args, **_kwargs):
        return int(self.current_len)

    def get_max_cache_shape(self):
        return int(self.max_cache_len)

    @property
    def generated_len(self) -> int:
        return max(0, int(self.current_len) - int(self.prompt_len))


def _last_static_position_id(position_ids, cache_position, fallback: int) -> int:
    if position_ids is not None:
        pos = position_ids.detach().cpu().numpy()
        if pos.ndim == 3 and pos.shape[0] == 4:
            pos = pos[1:]
        if pos.ndim == 3:
            return int(pos[0, 0, -1])
        if pos.ndim == 2:
            return int(pos[0, -1])
        if pos.ndim == 1:
            return int(pos[-1])
        raise AssertionError(f"unexpected decode position_ids shape: {pos.shape}")
    if cache_position is not None:
        return int(cache_position.detach().cpu().reshape(-1)[-1].item())
    return int(fallback)


class StaticTalkerLayerRunner:
    def __init__(
        self,
        talker_compiled_dir: Path,
        prefill_len: int,
        hidden_state_type: str,
        hidden_size: int,
        kv_dim: int,
        num_layers: int,
        vocab_size: int,
        model_type: str,
        axengine_device: Optional[str] = None,
        log_progress: bool = False,
    ):
        self.hidden_state_type = hidden_state_type
        self.m_dtype = _numpy_dtype(hidden_state_type)
        self.bf16_type = _numpy_dtype("bf16")
        self.hidden_size = int(hidden_size)
        self.kv_dim = int(kv_dim)
        self.num_layers = int(num_layers)
        self.vocab_size = int(vocab_size)
        self.prefill_len = int(prefill_len)
        self.log_progress = bool(log_progress)

        self.layer_sessions = []
        for layer_idx in range(self.num_layers):
            ax_path = talker_compiled_dir / f"{model_type}_p{self.prefill_len}_l{layer_idx}_together.axmodel"
            self.layer_sessions.append(AxEngineSession(ax_path, device=axengine_device))

        post_path = talker_compiled_dir / f"{model_type}_post.axmodel"
        self.post_session = AxEngineSession(post_path, device=axengine_device)
        post_input_shape = self.post_session.input_shape("input")
        if len(post_input_shape) != 3 or post_input_shape[0] != 1 or post_input_shape[2] != self.hidden_size:
            raise AssertionError(f"unexpected talker post input shape={post_input_shape}, hidden_size={self.hidden_size}")

        k_cache_shape = self.layer_sessions[0].input_shape("K_cache")
        if len(k_cache_shape) != 3 or k_cache_shape[0] != 1 or k_cache_shape[2] != self.kv_dim:
            raise AssertionError(f"unexpected K_cache shape={k_cache_shape}, kv_dim={self.kv_dim}")
        self.kv_cache_len = int(k_cache_shape[1])
        self.decode_mask_cache = _build_decode_mask_cache(self.kv_cache_len, self.bf16_type)
        self.decode_output_names = _shape_group_output_names(0)

        prefill_group_ids = []
        for input_name in self.layer_sessions[0].input_shapes:
            if not input_name.startswith("input_"):
                continue
            suffix = input_name.rsplit("_", 1)[-1]
            if suffix.isdigit():
                prefill_group_ids.append(int(suffix))
        self.prefill_group_ids = sorted(prefill_group_ids)
        if not self.prefill_group_ids or self.prefill_group_ids != list(range(1, max(self.prefill_group_ids) + 1)):
            raise AssertionError(f"unexpected talker prefill shape groups: {self.prefill_group_ids}")
        self.prefill_chunk_len = int(self.layer_sessions[0].input_shape("input_1")[1])
        if self.prefill_chunk_len != self.prefill_len:
            raise AssertionError(
                f"talker prefill_len arg={self.prefill_len} does not match axmodel chunk={self.prefill_chunk_len}"
            )
        self.max_prefill_len = self.prefill_chunk_len * len(self.prefill_group_ids)

    def static_prefill_len(self, valid_len: int) -> int:
        if not (0 < valid_len <= self.max_prefill_len):
            raise AssertionError(
                f"valid_len={valid_len} exceeds talker chunked prefill capacity={self.max_prefill_len} "
                f"({len(self.prefill_group_ids)} chunks x {self.prefill_chunk_len})"
            )
        chunk_count = (int(valid_len) + self.prefill_chunk_len - 1) // self.prefill_chunk_len
        return chunk_count * self.prefill_chunk_len

    def prefill(self, prefill_embed, prefill_indices, valid_len: int) -> Tuple[StaticTalkerState, np.ndarray]:
        padded_len = self.static_prefill_len(valid_len)
        chunk_count = padded_len // self.prefill_chunk_len
        if prefill_indices.shape[-1] < padded_len:
            raise AssertionError(f"prefill_indices length={prefill_indices.shape[-1]} < padded_len={padded_len}")

        data = np.zeros((1, padded_len, self.hidden_size), dtype=self.m_dtype)
        data[:, :valid_len, :] = prefill_embed.detach().cpu().float().numpy().astype(self.m_dtype)
        prefill_mask = np.zeros((1, padded_len, padded_len), dtype=np.float32) - 65536.0
        for row in range(valid_len):
            prefill_mask[:, row, : row + 1] = 0.0
        prefill_mask = prefill_mask.astype(self.bf16_type)

        k_caches = [np.zeros((1, self.kv_cache_len, self.kv_dim), dtype=self.m_dtype) for _ in range(self.num_layers)]
        v_caches = [np.zeros((1, self.kv_cache_len, self.kv_dim), dtype=self.m_dtype) for _ in range(self.num_layers)]
        debug_info: Dict[str, np.ndarray] = {}

        for layer_idx, session in enumerate(self.layer_sessions):
            if self.log_progress:
                print(
                    f"[axengine][talker][prefill] layer {layer_idx + 1}/{self.num_layers} "
                    f"chunks={chunk_count}",
                    flush=True,
                )
            layer_outputs = []
            for chunk_idx in range(chunk_count):
                group_id = chunk_idx + 1
                start = chunk_idx * self.prefill_chunk_len
                end = start + self.prefill_chunk_len
                suffix = f"_{group_id}"
                if chunk_idx == 0:
                    k_input = np.zeros(session.input_shape(f"K_cache{suffix}"), dtype=self.m_dtype)
                    v_input = np.zeros(session.input_shape(f"V_cache{suffix}"), dtype=self.m_dtype)
                else:
                    k_input = k_caches[layer_idx][:, :start, :]
                    v_input = v_caches[layer_idx][:, :start, :]

                outputs = session.run(
                    {
                        f"indices": prefill_indices[:, start:end],
                        f"input": data[:, start:end, :],
                        f"mask": prefill_mask[:, start:end, :end],
                        f"K_cache": k_input,
                        f"V_cache": v_input,
                    },
                    shape_group=group_id,
                    expected_output_names=_shape_group_output_names(group_id),
                )
                k_prefill = outputs[f"K_cache_out{suffix}"].reshape((1, self.prefill_chunk_len, self.kv_dim))
                v_prefill = outputs[f"V_cache_out{suffix}"].reshape((1, self.prefill_chunk_len, self.kv_dim))
                hidden_chunk = outputs[f"output{suffix}"].reshape((1, self.prefill_chunk_len, self.hidden_size))
                cache_end = min(end, self.kv_cache_len)
                if cache_end > start:
                    cache_width = cache_end - start
                    k_caches[layer_idx][:, start:cache_end, :] = k_prefill[:, :cache_width, :]
                    v_caches[layer_idx][:, start:cache_end, :] = v_prefill[:, :cache_width, :]
                layer_outputs.append(hidden_chunk)

            data = np.concatenate(layer_outputs, axis=1)
            data = data.astype(self.m_dtype)

        state = StaticTalkerState(
            k_caches=k_caches,
            v_caches=v_caches,
            current_len=int(valid_len),
            prompt_len=int(valid_len),
            max_cache_len=int(self.kv_cache_len),
        )
        return state, data[:, :valid_len, :].astype(self.m_dtype)

    def decode_one(self, decode_embed, state: StaticTalkerState, position_index: Optional[int] = None):
        if state.current_len >= self.kv_cache_len:
            raise AssertionError(
                f"decode current_len={state.current_len} reaches kv_cache_len={self.kv_cache_len}; "
                f"prompt_len={state.prompt_len}, generated_len={state.generated_len}. "
                "The static talker axmodel ran out of KV cache before EOS was generated. "
                "Reduce max_new_tokens/recompile with a larger kv_cache_len; if this happens for short text, "
                "check the talker logits and code_predictor outputs."
            )
        data_decode = decode_embed.detach().cpu().float().numpy().astype(self.m_dtype)
        if data_decode.shape != (1, 1, self.hidden_size):
            raise AssertionError(f"unexpected decode embed shape={data_decode.shape}")

        decode_mask = self.decode_mask_cache[state.current_len]
        decode_indices = np.array([[state.current_len if position_index is None else int(position_index)]], dtype=np.uint32)
        for layer_idx, session in enumerate(self.layer_sessions):
            if self.log_progress:
                print(
                    f"[axengine][talker][decode] pos {state.current_len} layer {layer_idx + 1}/{self.num_layers}",
                    flush=True,
                )
            outputs = session.run(
                {
                    "K_cache": state.k_caches[layer_idx],
                    "V_cache": state.v_caches[layer_idx],
                    "indices": decode_indices,
                    "input": data_decode,
                    "mask": decode_mask,
                },
                shape_group=0,
                expected_output_names=self.decode_output_names,
            )
            data_decode = outputs["output"].reshape((1, 1, self.hidden_size))
            state.k_caches[layer_idx][:, state.current_len, :] = outputs["K_cache_out"].reshape((1, self.kv_dim))
            state.v_caches[layer_idx][:, state.current_len, :] = outputs["V_cache_out"].reshape((1, self.kv_dim))
            data_decode = data_decode.astype(self.m_dtype)

        state.current_len += 1
        return data_decode.astype(self.m_dtype)

    def run_post_logits(self, raw_hidden: np.ndarray) -> np.ndarray:
        hidden_token = raw_hidden[:, -1:, :].astype(self.m_dtype, copy=False)
        outputs = self.post_session.run({"input": hidden_token})
        out_key = "output" if "output" in outputs else "logits"
        logits = outputs[out_key].astype(np.float32, copy=False)
        if logits.size % self.vocab_size != 0:
            raise AssertionError(f"unexpected talker post output size={logits.size}, vocab_size={self.vocab_size}")
        logits = logits.reshape((1, logits.size // self.vocab_size, self.vocab_size))
        return np.ascontiguousarray(logits[:, -1:, :])


def _build_axengine_talker_module(
    torch_module,
    modeling_outputs,
    original_model,
    runner: StaticTalkerLayerRunner,
    original_codec_head=None,
    compare_frames: int = 0,
    dump_cpp_dir: Optional[str] = None,
    dump_trailing_start: int = 7,
    dump_audio_token_id: Optional[int] = None,
    dump_streaming: bool = False,
):
    class _AxEngineQwen3TTSTalkerModel(torch_module.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = original_model.config
            self.codec_embedding = original_model.get_input_embeddings()
            self.text_embedding = original_model.get_text_embeddings()
            self.norm = original_model.norm
            self.gradient_checkpointing = False
            self.runner = runner
            self.compare_frames = max(0, int(compare_frames))
            self._debug_forward_idx = 0
            self._static_state = None
            self._last_raw_hidden = None
            self._hf_past_key_values = None
            self._dump_cpp_dir = dump_cpp_dir
            self._dump_trailing_start = dump_trailing_start
            self._dump_audio_token_id = dump_audio_token_id
            self._dump_streaming = dump_streaming
            self._dump_done = False
            object.__setattr__(self, "_hf_reference_model", original_model if self.compare_frames > 0 else None)
            object.__setattr__(self, "_hf_reference_codec_head", original_codec_head if self.compare_frames > 0 else None)
            if self._hf_reference_model is not None:
                self._hf_reference_model.eval()
            if self._hf_reference_codec_head is not None:
                self._hf_reference_codec_head.eval()

        @staticmethod
        def _topk_list(scores: np.ndarray, k: int = 8):
            flat = np.asarray(scores, dtype=np.float32).reshape(-1)
            k = min(max(1, int(k)), flat.shape[0])
            idx = np.argsort(flat)[-k:][::-1]
            return [(int(i), float(flat[i])) for i in idx]

        @staticmethod
        def _cosine_np(a: np.ndarray, b: np.ndarray) -> float:
            aa = np.asarray(a, dtype=np.float32).reshape(-1)
            bb = np.asarray(b, dtype=np.float32).reshape(-1)
            denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
            if denom < 1e-12:
                return float("nan")
            return float(np.dot(aa, bb) / denom)

        @staticmethod
        def _slice_position_ids(pos, valid_len: int):
            if pos is None:
                return None
            if pos.ndim == 3:
                return pos[:, :, -valid_len:]
            if pos.ndim == 2:
                return pos[:, -valid_len:]
            if pos.ndim == 1:
                return pos[-valid_len:]
            return pos

        def _compare_with_hf_reference(
            self,
            frame_idx: int,
            stage: str,
            raw_hidden: np.ndarray,
            hidden_states,
            hf_inputs_embeds,
            attention_mask,
            position_ids,
            use_cache,
            cache_position,
        ):
            hf_model = self._hf_reference_model
            hf_head = self._hf_reference_codec_head
            if hf_model is None or hf_head is None or frame_idx > self.compare_frames:
                return
            try:
                with torch_module.no_grad():
                    hf_outputs = hf_model(
                        input_ids=None,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=None if stage == "prefill" else self._hf_past_key_values,
                        inputs_embeds=hf_inputs_embeds,
                        use_cache=use_cache,
                        output_attentions=False,
                        output_hidden_states=False,
                        cache_position=cache_position,
                    )
                    self._hf_past_key_values = hf_outputs.past_key_values
                    hf_hidden = hf_outputs.last_hidden_state[:, -1:, :]
                    hf_logits = hf_head(hf_hidden).detach().cpu().float().numpy().reshape(-1)
                ax_hidden = hidden_states[:, -1:, :].detach().cpu().float().numpy()
                ax_logits = self.runner.run_post_logits(raw_hidden).reshape(-1)
                hidden_cos = self._cosine_np(hf_hidden.detach().cpu().float().numpy(), ax_hidden)
                logits_cos = self._cosine_np(hf_logits, ax_logits)
                hf_argmax = int(np.argmax(hf_logits))
                ax_argmax = int(np.argmax(ax_logits))
                print(
                    f"[compare][talker][frame={frame_idx}][{stage}] "
                    f"hidden_cos={hidden_cos} logits_cos={logits_cos} "
                    f"hf_argmax={hf_argmax} ax_argmax={ax_argmax} "
                    f"hf_top={self._topk_list(hf_logits)} ax_top={self._topk_list(ax_logits)}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[compare][talker][frame={frame_idx}][{stage}] failed: {type(exc).__name__}: {exc}",
                    flush=True,
                )

        @property
        def device(self):
            return self.codec_embedding.weight.device

        @property
        def dtype(self):
            return self.codec_embedding.weight.dtype

        def get_input_embeddings(self):
            return self.codec_embedding

        def get_text_embeddings(self):
            return self.text_embedding

        def set_input_embeddings(self, value):
            self.codec_embedding = value

        def _to_hidden_tensor(self, raw_hidden, device, dtype):
            hidden = torch_module.from_numpy(raw_hidden.astype(np.float32)).to(device=device, dtype=dtype)
            return self.norm(hidden)

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            cache_position=None,
            **_kwargs,
        ):
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError("either input_ids or inputs_embeds must be provided")
                inputs_embeds = self.codec_embedding(input_ids)
            if inputs_embeds.shape[0] != 1:
                raise AssertionError("axengine talker replacement currently supports batch_size=1 only")

            seq_len = int(inputs_embeds.shape[1])
            is_prefill = past_key_values is None or self._static_state is None or seq_len > 1
            hf_inputs_embeds = inputs_embeds[:, -1:, :]
            hf_attention_mask = attention_mask
            hf_position_ids = position_ids
            hf_cache_position = cache_position
            stage = "decode"
            if is_prefill:
                stage = "prefill"
                self._hf_past_key_values = None
                if attention_mask is None:
                    valid_len = seq_len
                    hf_attention_mask = None
                else:
                    valid_len = int(attention_mask[0].sum().item())
                    valid_len = min(valid_len, seq_len)
                    hf_attention_mask = attention_mask[:, -valid_len:]
                active_embeds = inputs_embeds[:, -valid_len:, :]
                if self._dump_cpp_dir is not None and not self._dump_done:
                    dump_dir = Path(self._dump_cpp_dir)
                    dump_dir.mkdir(parents=True, exist_ok=True)
                    _save_prefill_embeds_bin(dump_dir / "prefill_embeds.bin", active_embeds)
                    audio_token_id = self._dump_audio_token_id
                    if audio_token_id is None:
                        audio_token_id = int(getattr(self.config, "tts_bos_token_id", 151672))
                    _save_meta_json(
                        dump_dir / "meta.json",
                        S=int(valid_len),
                        hidden_size=int(active_embeds.shape[-1]),
                        audio_token_id=audio_token_id,
                        trailing_start=self._dump_trailing_start,
                        streaming=self._dump_streaming,
                    )
                    self._dump_done = True
                    print(f"[dump] C++ prefill inputs saved to {dump_dir}")
                hf_inputs_embeds = active_embeds
                hf_position_ids = self._slice_position_ids(position_ids, valid_len)
                hf_cache_position = None
                prefill_indices = _position_ids_to_static_indices(
                    position_ids,
                    valid_len,
                    self.runner.static_prefill_len(valid_len),
                )
                self._static_state, raw_hidden = self.runner.prefill(active_embeds, prefill_indices, valid_len)
                past = self._static_state
            else:
                state = past_key_values if isinstance(past_key_values, StaticTalkerState) else self._static_state
                position_index = _last_static_position_id(position_ids, cache_position, state.current_len)
                raw_hidden = self.runner.decode_one(inputs_embeds[:, -1:, :], state, position_index=position_index)
                self._static_state = state
                past = state

            self._last_raw_hidden = raw_hidden
            hidden_states = self._to_hidden_tensor(raw_hidden, inputs_embeds.device, inputs_embeds.dtype)
            self._debug_forward_idx += 1
            self._compare_with_hf_reference(
                frame_idx=self._debug_forward_idx,
                stage=stage,
                raw_hidden=raw_hidden,
                hidden_states=hidden_states,
                hf_inputs_embeds=hf_inputs_embeds,
                attention_mask=hf_attention_mask,
                position_ids=hf_position_ids,
                use_cache=use_cache,
                cache_position=hf_cache_position,
            )
            return modeling_outputs.BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past if use_cache is not False else None,
                hidden_states=(hidden_states,) if output_hidden_states else None,
                attentions=None,
            )

    return _AxEngineQwen3TTSTalkerModel()


def _build_axengine_talker_post_head(torch_module, axengine_talker_model):
    class _AxEngineQwen3TTSTalkerPostHead(torch_module.nn.Module):
        def __init__(self):
            super().__init__()
            self.axengine_talker_model = axengine_talker_model

        def forward(self, hidden_states):
            raw_hidden = self.axengine_talker_model._last_raw_hidden
            if raw_hidden is None:
                raise RuntimeError("talker post head was called before axengine talker model forward")
            logits = self.axengine_talker_model.runner.run_post_logits(raw_hidden)
            return torch_module.from_numpy(logits).to(device=hidden_states.device, dtype=torch_module.float32)

    return _AxEngineQwen3TTSTalkerPostHead()


def replace_talker_model(qwen_wrapper, args, torch_module):
    _install_import_stubs()
    _patch_onnx_tensorproto()

    from transformers import modeling_outputs

    talker = qwen_wrapper.model.talker
    original_model = talker.model
    original_codec_head = talker.codec_head
    cfg = original_model.config
    runner = StaticTalkerLayerRunner(
        talker_compiled_dir=args.compiled_model_path,
        prefill_len=args.prefill_len,
        hidden_state_type=args.hidden_state_type,
        hidden_size=int(cfg.hidden_size),
        kv_dim=_kv_dim_from_config(cfg),
        num_layers=int(cfg.num_hidden_layers),
        vocab_size=int(cfg.vocab_size),
        model_type=args.model_type,
        axengine_device=args.axengine_device,
    )
    compare_frames = int(getattr(args, "compare_talker_frames", 0))
    axengine_talker_model = _build_axengine_talker_module(
        torch_module,
        modeling_outputs,
        original_model,
        runner,
        original_codec_head=original_codec_head,
        compare_frames=compare_frames,
        dump_cpp_dir=getattr(args, "dump_cpp_input_dir", None),
        dump_trailing_start=getattr(args, "dump_trailing_start", 7),
        dump_audio_token_id=getattr(args, "dump_audio_token_id", None),
        dump_streaming=not getattr(args, "non_streaming_mode", False),
    ).eval()
    talker.model = axengine_talker_model
    talker.codec_head = _build_axengine_talker_post_head(torch_module, axengine_talker_model).eval()
    talker.rope_deltas = None
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
    print(
        f"[replace] Qwen3TTSTalkerModel -> axengine "
        f"layers={cfg.num_hidden_layers}, prefill_len={args.prefill_len}, "
        f"kv_cache_len={runner.kv_cache_len}, post_logits=axengine, compare_frames={compare_frames}",
        flush=True,
    )


def _select_next_code_from_logits(
    logits: np.ndarray,
    do_sample: bool,
    top_k: Optional[int],
    top_p: Optional[float],
    temperature: Optional[float],
) -> int:
    scores = logits.astype(np.float64).copy()
    if not do_sample:
        return int(np.argmax(scores))

    temp = max(float(temperature if temperature is not None else 1.0), 1e-5)
    scores = scores / temp

    if top_k is not None and top_k > 0 and top_k < scores.shape[0]:
        kth = np.partition(scores, -top_k)[-top_k]
        scores[scores < kth] = -1e30

    scores = scores - scores.max()
    probs = np.exp(scores)
    probs = probs / np.clip(probs.sum(), 1e-12, None)

    top_p_value = float(top_p if top_p is not None else 1.0)
    if 0 < top_p_value < 1.0:
        idx = np.argsort(probs)[::-1]
        csum = np.cumsum(probs[idx])
        drop = csum > top_p_value
        if drop.size > 0:
            drop[0] = False
        probs[idx[drop]] = 0
        probs = probs / np.clip(probs.sum(), 1e-12, None)

    return int(np.random.choice(np.arange(scores.shape[0]), p=probs))


class StaticCodePredictorRunner:
    def __init__(
        self,
        compiled_model_path: Path,
        lm_head_onnx_dir: Optional[Path],
        prefill_len: int,
        hidden_state_type: str,
        hidden_size: int,
        kv_dim: int,
        num_layers: int,
        vocab_size: int,
        num_code_groups: int,
        model_type: str,
        axengine_device: Optional[str] = None,
        log_progress: bool = False,
        prefill_io_names: str = "base",
    ):
        self.hidden_state_type = hidden_state_type
        self.m_dtype = _numpy_dtype(hidden_state_type)
        self.bf16_type = _numpy_dtype("bf16")
        self.hidden_size = int(hidden_size)
        self.kv_dim = int(kv_dim)
        self.num_layers = int(num_layers)
        self.vocab_size = int(vocab_size)
        self.num_sub_codes = int(num_code_groups) - 1
        self.prefill_len = int(prefill_len)
        self.log_progress = bool(log_progress)
        if prefill_io_names not in {"base", "suffixed"}:
            raise ValueError(f"unsupported code predictor prefill_io_names={prefill_io_names!r}")
        self.prefill_io_names = prefill_io_names

        if self.num_sub_codes <= 0:
            raise AssertionError(f"invalid num_sub_codes={self.num_sub_codes}")

        self.layer_sessions = []
        for layer_idx in range(self.num_layers):
            ax_path = compiled_model_path / f"{model_type}_p{self.prefill_len}_l{layer_idx}_together.axmodel"
            self.layer_sessions.append(AxEngineSession(ax_path, device=axengine_device))

        post_path = compiled_model_path / f"{model_type}_post.axmodel"
        self.post_session = AxEngineSession(post_path, device=axengine_device)

        self.lm_head_sessions = []
        self.lm_head_backend = "onnx_numpy" if lm_head_onnx_dir is not None else "axengine"
        for step in range(self.num_sub_codes):
            if lm_head_onnx_dir is not None:
                lm_path = Path(lm_head_onnx_dir) / f"code_predictor_lm_head_{step}.onnx"
                self.lm_head_sessions.append(NumpyOnnxLinearSession(lm_path))
            else:
                lm_path = compiled_model_path / f"code_predictor_lm_head_{step}.axmodel"
                self.lm_head_sessions.append(AxEngineSession(lm_path, device=axengine_device))

        k_cache_shape = self.layer_sessions[0].input_shape("K_cache")
        if len(k_cache_shape) != 3 or k_cache_shape[0] != 1 or k_cache_shape[2] != self.kv_dim:
            raise AssertionError(f"unexpected K_cache shape={k_cache_shape}, kv_dim={self.kv_dim}")
        self.kv_cache_len = int(k_cache_shape[1])
        self.decode_mask_cache = _build_decode_mask_cache(self.kv_cache_len, self.bf16_type)
        self.prefill_shape_group = 1
        self.prefill_output_names = _shape_group_output_names(self.prefill_shape_group)
        self.decode_output_names = _shape_group_output_names(0)
        self.zero_k_prefill = np.zeros(self.layer_sessions[0].input_shape("K_cache_1"), dtype=self.m_dtype)
        self.zero_v_prefill = np.zeros(self.layer_sessions[0].input_shape("V_cache_1"), dtype=self.m_dtype)

        self.lm_head_input_buffers = []
        for lm_head_session in self.lm_head_sessions:
            input_shape = lm_head_session.input_shape("input")
            if len(input_shape) != 3 or input_shape[0] != 1 or input_shape[2] != self.hidden_size:
                raise AssertionError(f"unexpected lm_head input shape={input_shape}, hidden_size={self.hidden_size}")
            self.lm_head_input_buffers.append(np.zeros((1, int(input_shape[1]), self.hidden_size), dtype=np.float32))

    def _code_predictor_prefill_feed(
        self,
        session: AxEngineSession,
        indices: np.ndarray,
        data: np.ndarray,
        prefill_mask: np.ndarray,
        io_names: str,
    ) -> Dict[str, np.ndarray]:
        k_zeros = np.zeros(session.input_shape("K_cache_1"), dtype=self.m_dtype)
        v_zeros = np.zeros(session.input_shape("V_cache_1"), dtype=self.m_dtype)
        if io_names == "suffixed":
            return {
                "indices_1": indices,
                "input_1": data,
                "mask_1": prefill_mask,
                "K_cache_1": k_zeros,
                "V_cache_1": v_zeros,
            }
        if io_names == "base":
            return {
                "indices": indices,
                "input": data,
                "mask": prefill_mask,
                "K_cache": k_zeros,
                "V_cache": v_zeros,
            }
        raise ValueError(f"unsupported code predictor prefill io_names={io_names!r}")

    def _run_post_norm(self, hidden_token: np.ndarray) -> np.ndarray:
        outputs = self.post_session.run({"input": hidden_token})
        out_key = "output_norm" if "output_norm" in outputs else "output"
        out = outputs[out_key].astype(np.float32)
        if out.size % self.hidden_size != 0:
            raise AssertionError(f"unexpected post output size={out.size}, hidden_size={self.hidden_size}")
        out = out.reshape((1, out.size // self.hidden_size, self.hidden_size))
        return out[:, -1:, :]

    def _run_lm_head_logits(self, step: int, hidden_norm: np.ndarray) -> np.ndarray:
        lm_input = self.lm_head_input_buffers[step]
        lm_input.fill(0.0)
        lm_input[:, -1:, :] = hidden_norm.astype(np.float32, copy=False)

        outputs = self.lm_head_sessions[step].run({"input": lm_input})
        out_key = "output" if "output" in outputs else "logits"
        logits = outputs[out_key].astype(np.float32, copy=False)
        if logits.size % self.vocab_size != 0:
            raise AssertionError(f"unexpected lm_head output size={logits.size}, vocab_size={self.vocab_size}")
        logits = logits.reshape((1, logits.size // self.vocab_size, self.vocab_size))
        return logits[:, -1, :].reshape(-1)

    def _decode_one(
        self,
        decode_embed: np.ndarray,
        k_caches: List[np.ndarray],
        v_caches: List[np.ndarray],
        current_len: int,
    ) -> np.ndarray:
        if current_len >= self.kv_cache_len:
            raise AssertionError(f"decode current_len={current_len} reaches kv_cache_len={self.kv_cache_len}")
        data_decode = decode_embed.astype(self.m_dtype, copy=False)
        decode_mask = self.decode_mask_cache[current_len]
        decode_indices = np.array([[current_len]], dtype=np.uint32)

        for layer_idx, session in enumerate(self.layer_sessions):
            if self.log_progress:
                print(
                    f"[axengine][code_predictor][decode] pos {current_len} layer {layer_idx + 1}/{self.num_layers}",
                    flush=True,
                )
            outputs = session.run(
                {
                    "K_cache": k_caches[layer_idx],
                    "V_cache": v_caches[layer_idx],
                    "indices": decode_indices,
                    "input": data_decode,
                    "mask": decode_mask,
                },
                shape_group=0,
                expected_output_names=self.decode_output_names,
            )
            data_decode = outputs["output"].reshape((1, 1, self.hidden_size))
            k_caches[layer_idx][:, current_len, :] = outputs["K_cache_out"].reshape((1, self.kv_dim))
            v_caches[layer_idx][:, current_len, :] = outputs["V_cache_out"].reshape((1, self.kv_dim))
            data_decode = data_decode.astype(self.m_dtype)
        return data_decode.astype(self.m_dtype)

    def generate_from_inputs_embeds(
        self,
        inputs_embeds,
        embedding_tables,
        projection_module,
        torch_module,
        max_new_tokens: Optional[int],
        do_sample: bool,
        top_k: Optional[int],
        top_p: Optional[float],
        temperature: Optional[float],
        prefill_io_names: Optional[str] = None,
        return_debug: bool = False,
    ) -> List[int]:
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1:
            raise AssertionError(f"axengine code_predictor supports [1, seq, hidden], got {tuple(inputs_embeds.shape)}")
        valid_len = int(inputs_embeds.shape[1])
        if not (0 < valid_len <= self.prefill_len):
            raise AssertionError(f"valid_len={valid_len} exceeds compiled prefill_len={self.prefill_len}")
        if int(inputs_embeds.shape[-1]) != self.hidden_size:
            raise AssertionError(f"input hidden={inputs_embeds.shape[-1]} != compiled hidden={self.hidden_size}")

        start_lm_step = max(0, valid_len - 2)
        requested = self.num_sub_codes - start_lm_step if max_new_tokens is None else int(max_new_tokens)
        num_to_generate = min(max(0, requested), self.num_sub_codes - start_lm_step)
        if num_to_generate <= 0:
            return []
        io_names = self.prefill_io_names if prefill_io_names is None else prefill_io_names
        if io_names not in {"base", "suffixed"}:
            raise ValueError(f"unsupported code predictor prefill_io_names={io_names!r}")

        data = np.zeros((1, self.prefill_len, self.hidden_size), dtype=self.m_dtype)
        data[:, :valid_len, :] = inputs_embeds.detach().cpu().float().numpy().astype(self.m_dtype)

        indices = np.arange(self.prefill_len, dtype=np.uint32).reshape((1, self.prefill_len))
        indices[:, valid_len:] = 0
        prefill_mask = np.zeros((1, self.prefill_len, self.prefill_len), dtype=np.float32) - 65536.0
        for row in range(valid_len):
            prefill_mask[:, row, : row + 1] = 0.0
        prefill_mask = prefill_mask.astype(self.bf16_type)

        k_caches = [np.zeros((1, self.kv_cache_len, self.kv_dim), dtype=self.m_dtype) for _ in range(self.num_layers)]
        v_caches = [np.zeros((1, self.kv_cache_len, self.kv_dim), dtype=self.m_dtype) for _ in range(self.num_layers)]
        debug_info: Dict[str, np.ndarray] = {}

        for layer_idx, session in enumerate(self.layer_sessions):
            if self.log_progress:
                print(f"[axengine][code_predictor][prefill] layer {layer_idx + 1}/{self.num_layers}", flush=True)
            outputs = session.run(
                self._code_predictor_prefill_feed(
                    session=session,
                    indices=indices,
                    data=data,
                    prefill_mask=prefill_mask,
                    io_names=io_names,
                ),
                shape_group=self.prefill_shape_group,
                expected_output_names=self.prefill_output_names,
            )
            data = outputs["output_1"].reshape((1, self.prefill_len, self.hidden_size))
            if return_debug:
                debug_info[f"prefill_layer_{layer_idx}_last"] = data[:, valid_len - 1 : valid_len, :].astype(
                    np.float32, copy=True
                )
            k_prefill = outputs["K_cache_out_1"].reshape((1, self.prefill_len, self.kv_dim))
            v_prefill = outputs["V_cache_out_1"].reshape((1, self.prefill_len, self.kv_dim))
            copy_len = min(valid_len, self.prefill_len, self.kv_cache_len)
            k_caches[layer_idx][:, :copy_len, :] = k_prefill[:, :copy_len, :]
            v_caches[layer_idx][:, :copy_len, :] = v_prefill[:, :copy_len, :]
            data = data.astype(self.m_dtype)

        generated_ids: List[int] = []
        current_len = valid_len
        last_hidden_raw = data[:, valid_len - 1 : valid_len, :].astype(self.m_dtype)
        device = inputs_embeds.device

        for offset in range(num_to_generate):
            lm_step = start_lm_step + offset
            if offset > 0:
                if current_len >= self.kv_cache_len:
                    print(
                        f"[axengine][code_predictor] stop decode at step={lm_step}: "
                        f"current_len={current_len} reaches kv_cache_len={self.kv_cache_len}",
                        flush=True,
                    )
                    break
                prev_id = generated_ids[-1]
                prev_tensor = torch_module.tensor([[prev_id]], device=device, dtype=torch_module.long)
                with torch_module.no_grad():
                    decode_embed = embedding_tables[lm_step - 1](prev_tensor)
                    if projection_module is not None:
                        decode_embed = projection_module(decode_embed)
                last_hidden_raw = self._decode_one(
                    decode_embed=decode_embed.detach().cpu().float().numpy(),
                    k_caches=k_caches,
                    v_caches=v_caches,
                    current_len=current_len,
                )
                current_len += 1

            hidden_norm = self._run_post_norm(last_hidden_raw)
            logits = self._run_lm_head_logits(lm_step, hidden_norm)
            if return_debug and offset == 0:
                debug_info["first_hidden_norm"] = hidden_norm.astype(np.float32, copy=True)
                debug_info["first_logits"] = logits.astype(np.float32, copy=True)
            next_id = _select_next_code_from_logits(
                logits=logits,
                do_sample=bool(do_sample),
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
            generated_ids.append(int(next_id))

        if return_debug:
            return generated_ids, debug_info
        return generated_ids


def _build_axengine_code_predictor_module(
    torch_module,
    original_model,
    runner: StaticCodePredictorRunner,
    compare_frames: int = 0,
):
    class _AxEngineQwen3TTSTalkerCodePredictorModelForConditionalGeneration(torch_module.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = original_model.config
            self.generation_config = getattr(original_model, "generation_config", None)
            self.vocab_size = int(getattr(original_model, "vocab_size", self.config.vocab_size))
            self.codec_embedding = original_model.get_input_embeddings()
            self.small_to_mtp_projection = original_model.small_to_mtp_projection
            self.runner = runner
            self.compare_frames = max(0, int(compare_frames))
            self._debug_frame_idx = 0
            self.hf_reference = original_model if self.compare_frames > 0 else None
            if self.hf_reference is not None:
                self.hf_reference.eval()

        @staticmethod
        def _sequence_to_list(sequences) -> List[int]:
            if sequences.ndim == 1:
                seq = sequences
            else:
                seq = sequences[0]
            return [int(x) for x in seq.detach().cpu().reshape(-1).tolist()]

        @staticmethod
        def _topk_list(scores: np.ndarray, k: int = 8):
            flat = np.asarray(scores, dtype=np.float32).reshape(-1)
            k = min(max(1, int(k)), flat.shape[0])
            idx = np.argsort(flat)[-k:][::-1]
            return [(int(i), float(flat[i])) for i in idx]

        @staticmethod
        def _cosine_np(a: np.ndarray, b: np.ndarray) -> float:
            aa = np.asarray(a, dtype=np.float32).reshape(-1)
            bb = np.asarray(b, dtype=np.float32).reshape(-1)
            denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
            if denom < 1e-12:
                return float("nan")
            return float(np.dot(aa, bb) / denom)

        def _compare_with_hf_reference(self, frame_idx: int, inputs_embeds, projected_inputs, max_new_tokens):
            if self.hf_reference is None or frame_idx > self.compare_frames:
                return
            compare_tokens = int(max_new_tokens) if max_new_tokens is not None else int(self.config.num_code_groups - 1)
            try:
                with torch_module.no_grad():
                    hf_forward = self.hf_reference(
                        inputs_embeds=inputs_embeds,
                        use_cache=True,
                        output_hidden_states=True,
                    )
                    hf_result = self.hf_reference.generate(
                        inputs_embeds=inputs_embeds,
                        max_new_tokens=compare_tokens,
                        do_sample=False,
                        return_dict_in_generate=True,
                    )
                hf_ids = self._sequence_to_list(hf_result.sequences)
                hf_logits = hf_forward.logits[:, -1, :].detach().cpu().float().numpy().reshape(-1)
                hf_hidden = None
                if getattr(hf_forward, "hidden_states", None):
                    hf_hidden = hf_forward.hidden_states[-1][:, -1:, :].detach().cpu().float().numpy()
                ax_ids, ax_debug = self.runner.generate_from_inputs_embeds(
                    inputs_embeds=projected_inputs,
                    embedding_tables=self.codec_embedding,
                    projection_module=self.small_to_mtp_projection,
                    torch_module=torch_module,
                    max_new_tokens=compare_tokens,
                    do_sample=False,
                    top_k=None,
                    top_p=1.0,
                    temperature=1.0,
                    return_debug=True,
                )
                ax_logits = ax_debug.get("first_logits")
                ax_hidden = ax_debug.get("first_hidden_norm")
                hidden_cos = None
                if hf_hidden is not None and ax_hidden is not None:
                    hidden_cos = self._cosine_np(hf_hidden, ax_hidden)
                layer_cos = []
                hf_hidden_states = getattr(hf_forward, "hidden_states", None)
                if hf_hidden_states:
                    for layer_idx in range(int(self.config.num_hidden_layers)):
                        ax_layer = ax_debug.get(f"prefill_layer_{layer_idx}_last")
                        if ax_layer is None or layer_idx + 1 >= len(hf_hidden_states):
                            continue
                        hf_layer = hf_hidden_states[layer_idx + 1][:, -1:, :].detach().cpu().float().numpy()
                        layer_cos.append((layer_idx, self._cosine_np(hf_layer, ax_layer)))
                print(
                    f"[compare][code_predictor][frame={frame_idx}][step0] "
                    f"hidden_cos={hidden_cos} layer_cos={layer_cos} "
                    f"hf_top={self._topk_list(hf_logits)} "
                    f"ax_top={self._topk_list(ax_logits) if ax_logits is not None else None}",
                    flush=True,
                )
                first_mismatch = next(
                    (i for i, (hf_id, ax_id) in enumerate(zip(hf_ids, ax_ids)) if int(hf_id) != int(ax_id)),
                    None,
                )
                same_len = len(hf_ids) == len(ax_ids)
                match = first_mismatch is None and same_len
                print(
                    f"[compare][code_predictor][frame={frame_idx}] greedy_match={match} "
                    f"first_mismatch={first_mismatch} hf={hf_ids} ax={ax_ids}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[compare][code_predictor][frame={frame_idx}] failed: {type(exc).__name__}: {exc}",
                    flush=True,
                )

        @property
        def device(self):
            try:
                return next(self.parameters()).device
            except StopIteration:
                return torch_module.device("cpu")

        @property
        def dtype(self):
            try:
                return next(self.parameters()).dtype
            except StopIteration:
                return torch_module.float32

        def get_input_embeddings(self):
            return self.codec_embedding

        def set_input_embeddings(self, value):
            self.codec_embedding = value

        def get_output_embeddings(self):
            return None

        def forward(self, *_args, **_kwargs):
            raise NotImplementedError("axengine code_predictor replacement implements generate() for inference only")

        def forward_finetune(self, *_args, **_kwargs):
            raise NotImplementedError("axengine code_predictor replacement does not implement forward_finetune()")

        @torch_module.no_grad()
        def generate(
            self,
            inputs_embeds=None,
            max_new_tokens=None,
            do_sample=False,
            top_k=50,
            top_p=1.0,
            temperature=1.0,
            return_dict_in_generate=False,
            **_kwargs,
        ):
            if inputs_embeds is None:
                raise ValueError("axengine code_predictor.generate requires inputs_embeds")
            self._debug_frame_idx += 1
            frame_idx = self._debug_frame_idx
            # NOTE: small_to_mtp_projection weights merged into CP layer0 axmodel, skip here
            # projected_inputs = self.small_to_mtp_projection(inputs_embeds)
            projected_inputs = inputs_embeds
            self._compare_with_hf_reference(frame_idx, inputs_embeds, projected_inputs, max_new_tokens)
            ids = self.runner.generate_from_inputs_embeds(
                inputs_embeds=projected_inputs,
                embedding_tables=self.codec_embedding,
                projection_module=None,
                torch_module=torch_module,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                prefill_io_names=self.runner.prefill_io_names,
            )
            if frame_idx <= self.compare_frames:
                mode = "sample" if bool(do_sample) else "greedy"
                print(
                    f"[compare][code_predictor][frame={frame_idx}] ax_actual_{mode}={ids}",
                    flush=True,
                )
            sequences = torch_module.tensor([ids], device=inputs_embeds.device, dtype=torch_module.long)
            if return_dict_in_generate:
                return SimpleNamespace(sequences=sequences)
            return sequences

    return _AxEngineQwen3TTSTalkerCodePredictorModelForConditionalGeneration()


def replace_code_predictor_model(qwen_wrapper, args, torch_module):
    _patch_onnx_tensorproto()

    talker = qwen_wrapper.model.talker
    original_model = talker.code_predictor
    cfg = original_model.config
    runner = StaticCodePredictorRunner(
        compiled_model_path=args.compiled_model_path,
        lm_head_onnx_dir=getattr(args, "code_predictor_lm_head_onnx_dir", None),
        prefill_len=args.prefill_len,
        hidden_state_type=args.hidden_state_type,
        hidden_size=int(cfg.hidden_size),
        kv_dim=_kv_dim_from_config(cfg),
        num_layers=int(cfg.num_hidden_layers),
        vocab_size=int(cfg.vocab_size),
        num_code_groups=int(cfg.num_code_groups),
        model_type=args.model_type,
        axengine_device=args.axengine_device,
        prefill_io_names=getattr(args, "code_predictor_prefill_io_names", "base"),
    )
    compare_frames = int(getattr(args, "compare_code_predictor_frames", 0))
    talker.code_predictor = _build_axengine_code_predictor_module(
        torch_module,
        original_model,
        runner,
        compare_frames=compare_frames,
    ).eval()
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
    print(
        f"[replace] Qwen3TTSTalkerCodePredictorModelForConditionalGeneration -> axengine "
        f"layers={cfg.num_hidden_layers}, prefill_len={args.prefill_len}, "
        f"kv_cache_len={runner.kv_cache_len}, prefill_io_names={runner.prefill_io_names}, "
        f"lm_head_backend={runner.lm_head_backend}, "
        f"compare_frames={compare_frames}",
        flush=True,
    )


def _str_to_torch_dtype(torch_module, dtype: str):
    table = {
        "bf16": torch_module.bfloat16,
        "bfloat16": torch_module.bfloat16,
        "fp16": torch_module.float16,
        "float16": torch_module.float16,
        "fp32": torch_module.float32,
        "float32": torch_module.float32,
    }
    key = dtype.lower()
    if key not in table:
        raise ValueError(f"unsupported dtype: {dtype}")
    return table[key]


def _set_seed(seed: int, torch_module):
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def _generate_and_save(qwen_wrapper, args, output_path: Path):
    import soundfile as sf

    model_type = getattr(qwen_wrapper.model, "tts_model_type", "")
    api = args.api
    if api == "auto":
        api = "voice_clone" if model_type == "base" else "custom_voice"

    generate_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        subtalker_dosample=args.subtalker_dosample,
        subtalker_top_k=args.subtalker_top_k,
        subtalker_top_p=args.subtalker_top_p,
        subtalker_temperature=args.subtalker_temperature,
    )

    if api == "voice_clone":
        ref_audio = args.ref_audio
        if not ref_audio.exists():
            candidate = args.qwen_tts_root / ref_audio
            if candidate.exists():
                ref_audio = candidate
        wavs, sr = qwen_wrapper.generate_voice_clone(
            text=args.text,
            language=args.language,
            ref_audio=str(ref_audio),
            ref_text=args.ref_text,
            x_vector_only_mode=args.x_vector_only_mode,
            non_streaming_mode=args.non_streaming_mode,
            **generate_kwargs,
        )
    elif api == "custom_voice":
        wavs, sr = qwen_wrapper.generate_custom_voice(
            text=args.text,
            language=args.language,
            speaker=args.speaker,
            instruct=args.instruct,
            **generate_kwargs,
        )
    else:
        raise ValueError(f"unsupported api: {api}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), wavs[0], sr)
    print(f"[audio] saved {output_path} sr={sr} samples={len(wavs[0])}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Qwen3-TTS voice-clone demo with talker and code_predictor on axengine."
    )
    parser.add_argument("--qwen_tts_root", type=Path, default=DEFAULT_QWEN_TTS_ROOT)
    parser.add_argument("--hf_model_path", type=Path, default=DEFAULT_HF_MODEL_PATH)
    parser.add_argument("--talker_compiled_model_path", type=Path, default=DEFAULT_TALKER_AXMODEL_DIR)
    parser.add_argument("--code_predictor_compiled_model_path", type=Path, default=DEFAULT_CODE_PREDICTOR_AXMODEL_DIR)
    parser.add_argument(
        "--code_predictor_lm_head_onnx_dir",
        type=Path,
        default=None,
        help="Use code_predictor_lm_head_*.onnx from this dir instead of lm_head axmodels.",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_name", default="output_axengine_talker_code_predictor.wav")
    parser.add_argument("--save_hf_reference", action="store_true")
    parser.add_argument("--device_map", default="cpu")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--axengine_device", default=None)
    parser.add_argument("--talker_axengine_device", default=None,
                        help="AXCL device for talker models (e.g. '0', '1'). Falls back to --axengine_device.")
    parser.add_argument("--code_predictor_axengine_device", default=None,
                        help="AXCL device for code_predictor models (e.g. '0', '1'). Falls back to --axengine_device.")
    parser.add_argument("--talker_prefill_len", type=int, default=128)
    parser.add_argument("--code_predictor_prefill_len", type=int, default=64)
    parser.add_argument("--hidden_state_type", default="bf16")
    parser.add_argument("--talker_model_type", default="qwen3_tts_talker")
    parser.add_argument("--code_predictor_model_type", default="qwen3_tts_talker_code_predictor")
    parser.add_argument(
        "--code_predictor_prefill_io_names",
        choices=["base", "suffixed"],
        default="base",
        help="Input-feed key style for code predictor prefill shape_group=1.",
    )
    parser.add_argument("--api", choices=["auto", "voice_clone", "custom_voice"], default="voice_clone")
    parser.add_argument("--text", default="高管也通过电话、短信、微信等方式对报道[j][ǐ]予好评。")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--ref_audio", type=Path, default=DEFAULT_QWEN_TTS_ROOT / "assets/zero_shot_prompt.wav")
    parser.add_argument("--ref_text", default="希望你以后能够做的比我还好呦。")
    parser.add_argument("--x_vector_only_mode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--non_streaming_mode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--speaker", default="Vivian")
    parser.add_argument("--instruct", default="用特别愤怒的语气说")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--do_sample", "--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--subtalker_dosample", "--subtalker-dosample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--subtalker_top_k", type=int, default=50)
    parser.add_argument("--subtalker_top_p", type=float, default=1.0)
    parser.add_argument("--subtalker_temperature", type=float, default=0.9)
    parser.add_argument(
        "--compare_code_predictor_frames",
        type=int,
        default=0,
        help="Compare HF greedy vs axengine greedy 15 sub-code outputs for the first N code predictor calls.",
    )
    parser.add_argument(
        "--compare_talker_frames",
        type=int,
        default=0,
        help="Compare HF talker.model hidden/logits vs axengine for the first N talker forwards.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dump_cpp_input_dir", type=Path, default=None,
                        help="Directory to dump C++ compatible inputs (prefill_embeds.bin, meta.json, tts_pad_vec.bin)")
    parser.add_argument("--dump_trailing_start", type=int, default=7,
                        help="trailing_start value written to meta.json (default: 7)")
    parser.add_argument("--dump_audio_token_id", type=int, default=None,
                        help="audio_token_id written to meta.json (default: config.tts_bos_token_id)")
    return parser.parse_args()


def _replace_args(args, compiled_model_path: Path, prefill_len: int, model_type: str, axengine_device=None):
    return SimpleNamespace(
        compiled_model_path=compiled_model_path,
        code_predictor_lm_head_onnx_dir=getattr(args, "code_predictor_lm_head_onnx_dir", None),
        prefill_len=prefill_len,
        hidden_state_type=args.hidden_state_type,
        model_type=model_type,
        axengine_device=axengine_device if axengine_device is not None else args.axengine_device,
        compare_code_predictor_frames=getattr(args, "compare_code_predictor_frames", 0),
        compare_talker_frames=getattr(args, "compare_talker_frames", 0),
        code_predictor_prefill_io_names=getattr(args, "code_predictor_prefill_io_names", "base"),
        dump_cpp_input_dir=getattr(args, "dump_cpp_input_dir", None),
        dump_trailing_start=getattr(args, "dump_trailing_start", 7),
        dump_audio_token_id=getattr(args, "dump_audio_token_id", None),
        non_streaming_mode=getattr(args, "non_streaming_mode", False),
    )


def main():
    args = parse_args()
    _prepare_import_environment(args)

    import torch
    from qwen_tts import Qwen3TTSModel

    if str(args.device_map).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"device_map={args.device_map!r} requested, but CUDA is not available in this process. "
            "Pass --device_map cpu on NPU-only hosts."
        )

    random.seed(args.seed)
    _set_seed(args.seed, torch)
    load_kwargs = {
        "device_map": args.device_map,
        "dtype": _str_to_torch_dtype(torch, args.dtype),
    }
    if args.local_files_only:
        load_kwargs["local_files_only"] = True

    print(f"[load] {args.hf_model_path}", flush=True)
    qwen = Qwen3TTSModel.from_pretrained(str(args.hf_model_path), **load_kwargs)
    qwen.model.eval()

    if args.dump_cpp_input_dir is not None:
        dump_dir = Path(args.dump_cpp_input_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        try:
            talker = qwen.model.talker
            tts_pad_token_id = getattr(qwen.model.config, "tts_pad_token_id", 151671)
            with torch.no_grad():
                tts_pad_embed = talker.model.text_projection(
                    talker.model.get_text_embeddings()(
                        torch.tensor([[tts_pad_token_id]], device=talker.model.device, dtype=torch.long)
                    )
                )
            _save_tts_pad_vec_bin(dump_dir / "tts_pad_vec.bin", tts_pad_embed)
            print(f"[dump] tts_pad_vec.bin saved to {dump_dir}")
        except Exception as exc:
            print(f"[dump] warning: failed to save tts_pad_vec.bin: {exc}")

    if args.save_hf_reference:
        _set_seed(args.seed, torch)
        qwen.model.talker.rope_deltas = None
        _generate_and_save(qwen, args, args.output_dir / "output_hf_reference.wav")

    replace_code_predictor_model(
        qwen,
        _replace_args(
            args,
            compiled_model_path=args.code_predictor_compiled_model_path,
            prefill_len=args.code_predictor_prefill_len,
            model_type=args.code_predictor_model_type,
            axengine_device=args.code_predictor_axengine_device,
        ),
        torch,
    )
    replace_talker_model(
        qwen,
        _replace_args(
            args,
            compiled_model_path=args.talker_compiled_model_path,
            prefill_len=args.talker_prefill_len,
            model_type=args.talker_model_type,
            axengine_device=args.talker_axengine_device,
        ),
        torch,
    )

    _set_seed(args.seed, torch)
    qwen.model.talker.rope_deltas = None
    _generate_and_save(qwen, args, args.output_dir / args.output_name)


if __name__ == "__main__":
    main()
