from pathlib import Path
from typing import Any, Dict, Union

import colored
import numpy as np
import onnxruntime as ort

from .enum import Backend
from .metadata import parse_metadata_from_onnx
from .tools import get_onnx_input_infos, get_onnx_output_infos


class ONNXEngine:
    def __init__(
        self,
        model_path: Union[str, Path],
        gpu_id: int = 0,
        backend: Union[str, int, Backend] = Backend.cpu,
        session_option: Dict[str, Any] = {},
        provider_option: Dict[str, Any] = {},
    ):
        """
        Initialize an ONNX model inference engine.

        Args:
            model_path (Union[str, Path]):
                Filename or serialized ONNX or ORT format model in a byte string.
            gpu_id (int, optional):
                GPU ID. Defaults to 0.
            backend (Union[str, int, Backend], optional):
                Backend. Defaults to Backend.cuda.
            session_option (Dict[str, Any], optional):
                Session options. Defaults to {}.
            provider_option (Dict[str, Any], optional):
                Provider options. Defaults to {}.
        """
        # setting device info
        backend = Backend.obj_to_enum(backend)
        self.device_id = 0 if backend.name == "cpu" else gpu_id

        # setting provider options
        providers = self._get_providers(backend, provider_option)

        # setting session options
        sess_options = self._get_session_info(session_option)

        # setting onnxruntime session
        model_path = str(model_path) if isinstance(model_path, Path) else model_path
        self.sess = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)

        # setting onnxruntime session info
        self.model_path = model_path
        self.metadata = parse_metadata_from_onnx(model_path)
        self.providers = self.sess.get_providers()
        self.provider_options = self.sess.get_provider_options()

        self.input_infos = get_onnx_input_infos(model_path)
        self.output_infos = get_onnx_output_infos(model_path)

    def __call__(self, **xs) -> Dict[str, np.ndarray]:
        output_names = list(self.output_infos.keys())
        outs = self.sess.run(output_names, {k: v for k, v in xs.items()})
        outs = {k: v for k, v in zip(output_names, outs)}
        return outs

    def _get_session_info(self, session_option: Dict[str, Any] = {}) -> ort.SessionOptions:
        """
        Ref: https://onnxruntime.ai/docs/api/python/api_summary.html#sessionoptions
        """
        sess_opt = ort.SessionOptions()
        session_option_default = {
            "graph_optimization_level": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            "log_severity_level": 2,
        }
        session_option_default.update(session_option)
        for k, v in session_option_default.items():
            setattr(sess_opt, k, v)
        return sess_opt

    def _get_providers(self, backend: Union[str, int, Backend], provider_option: Dict[str, Any] = {}) -> Backend:
        """
        Ref: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#configuration-options
        """
        if backend == Backend.cuda:
            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": self.device_id,
                        "cudnn_conv_use_max_workspace": "1",
                        **provider_option,
                    },
                )
            ]
        elif backend == Backend.coreml:
            providers = [
                (
                    "CoreMLExecutionProvider",
                    {
                        "ModelFormat": "MLProgram",
                        "MLComputeUnits": "ALL",
                        "RequireStaticInputShapes": "1",
                        **provider_option,
                    },
                )
            ]
        elif backend == Backend.cpu:
            providers = [("CPUExecutionProvider", {})]
            # "CPUExecutionProvider" is different from everything else.
            # provider_option = None
        else:
            raise ValueError(f"backend={backend} is not supported.")
        return providers

    def __repr__(self) -> str:
        import re

        def strip_ansi_codes(text):
            """Remove ANSI escape codes from a string."""
            ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
            return ansi_escape.sub("", text)

        def format_nested_dict(dict_data, indent=0):
            """Recursively format nested dictionaries with indentation."""
            info = []
            prefix = "  " * indent
            for key, value in dict_data.items():
                if isinstance(value, dict):
                    info.append(f"{prefix}{key}:")
                    info.append(format_nested_dict(value, indent + 1))
                elif isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                    try:
                        nested_dict = eval(value)
                        if isinstance(nested_dict, dict):
                            info.append(f"{prefix}{key}:")
                            info.append(format_nested_dict(nested_dict, indent + 1))
                        else:
                            info.append(f"{prefix}{key}: {value}")
                    except Exception:
                        info.append(f"{prefix}{key}: {value}")
                else:
                    info.append(f"{prefix}{key}: {value}")
            return "\n".join(info)

        title = "DOCSAID X ONNXRUNTIME"
        divider_length = 50
        divider = f"+{'-' * divider_length}+"
        styled_title = colored.stylize(title, [colored.fg("blue"), colored.attr("bold")])

        def center_text(text, width):
            """Center text within a fixed width, handling ANSI escape codes."""
            plain_text = strip_ansi_codes(text)
            text_length = len(plain_text)
            left_padding = (width - text_length) // 2
            right_padding = width - text_length - left_padding
            return f"|{' ' * left_padding}{text}{' ' * right_padding}|"

        path = f"Model Path: {self.model_path}"
        input_info = format_nested_dict(self.input_infos)
        output_info = format_nested_dict(self.output_infos)
        metadata = format_nested_dict({"metadata": self.metadata})
        providers = f"Provider: {', '.join(self.providers)}"
        provider_options = format_nested_dict(self.provider_options)

        sections = [
            divider,
            center_text(styled_title, divider_length),
            divider,
            path,
            input_info,
            output_info,
            metadata,
            providers,
            provider_options,
            divider,
        ]

        return "\n\n".join(sections)
