from pathlib import Path
from typing import Any, Dict, Union

import colored
import numpy as np
import onnxruntime as ort

from .metadata import get_onnx_metadata
from .tools import get_onnx_input_infos, get_onnx_output_infos


class ONNXEngineIOBinding:
    def __init__(
        self,
        model_path: Union[str, Path],
        input_initializer: Dict[str, np.ndarray],
        gpu_id: int = 0,
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
            session_option (Dict[str, Any], optional):
                Session options. Defaults to {}.
            provider_option (Dict[str, Any], optional):
                Provider options. Defaults to {}.
        """
        self.device_id = gpu_id
        providers = ["CUDAExecutionProvider"]
        provider_options = [
            {
                "device_id": self.device_id,
                "cudnn_conv_use_max_workspace": "1",
                "enable_cuda_graph": "1",
                **provider_option,
            }
        ]

        # setting session options
        sess_options = self._get_session_info(session_option)

        # setting onnxruntime session
        model_path = str(model_path) if isinstance(model_path, Path) else model_path
        self.sess = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers,
            provider_options=provider_options,
        )
        self.device = "cuda" if "CUDAExecutionProvider" in self.sess.get_providers() else "cpu"

        # setting onnxruntime session info
        self.model_path = model_path
        self.metadata = get_onnx_metadata(model_path)
        self.providers = self.sess.get_providers()
        self.provider_options = self.sess.get_provider_options()

        input_infos, output_infos = self._init_io_infos(model_path, input_initializer)

        io_binding, x_ortvalues, y_ortvalues = self._setup_io_binding(input_infos, output_infos)
        self.io_binding = io_binding
        self.x_ortvalues = x_ortvalues
        self.y_ortvalues = y_ortvalues
        self.input_infos = input_infos
        self.output_infos = output_infos
        # # Pass gpu_graph_id to RunOptions through RunConfigs
        # ro = ort.RunOptions()
        # # gpu_graph_id is optional if the session uses only one cuda graph
        # ro.add_run_config_entry("gpu_graph_id", "1")
        # self.run_option = ro

    def __call__(self, **xs) -> Dict[str, np.ndarray]:
        self._update_x_ortvalues(xs)
        # self.sess.run_with_iobinding(self.io_binding, self.run_option)
        self.sess.run_with_iobinding(self.io_binding)
        return {k: v.numpy() for k, v in self.y_ortvalues.items()}

    def _get_session_info(
        self,
        session_option: Dict[str, Any] = {},
    ) -> ort.SessionOptions:
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

    def _init_io_infos(self, model_path, input_initializer: dict):
        sess = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        outs = sess.run(None, input_initializer)
        input_shapes = {k: v.shape for k, v in input_initializer.items()}
        output_shapes = {x.name: o.shape for x, o in zip(sess.get_outputs(), outs)}
        input_infos = get_onnx_input_infos(model_path)
        output_infos = get_onnx_output_infos(model_path)
        for k, v in input_infos.items():
            v["shape"] = input_shapes[k]
        for k, v in output_infos.items():
            v["shape"] = output_shapes[k]
        del sess
        return input_infos, output_infos

    def _setup_io_binding(self, input_infos, output_infos):
        x_ortvalues = {}
        y_ortvalues = {}
        for k, v in input_infos.items():
            m = np.zeros(**v)
            x_ortvalues[k] = ort.OrtValue.ortvalue_from_numpy(m, device_type=self.device, device_id=self.device_id)
        for k, v in output_infos.items():
            m = np.zeros(**v)
            y_ortvalues[k] = ort.OrtValue.ortvalue_from_numpy(m, device_type=self.device, device_id=self.device_id)

        io_binding = self.sess.io_binding()
        for k, v in x_ortvalues.items():
            io_binding.bind_ortvalue_input(k, v)
        for k, v in y_ortvalues.items():
            io_binding.bind_ortvalue_output(k, v)

        return io_binding, x_ortvalues, y_ortvalues

    def _update_x_ortvalues(self, xs: dict):
        for k, v in self.x_ortvalues.items():
            v.update_inplace(xs[k])

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
