from enum import Enum
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
        providers = ['CUDAExecutionProvider']
        provider_options = [
            {
                'device_id': self.device_id,
                'cudnn_conv_use_max_workspace': '1',
                'enable_cuda_graph': '1',
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
            'graph_optimization_level': ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            'log_severity_level': 2,
        }
        session_option_default.update(session_option)
        for k, v in session_option_default.items():
            setattr(sess_opt, k, v)
        return sess_opt

    def _init_io_infos(self, model_path, input_initializer: dict):
        sess = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider'],
        )
        outs = sess.run(None, input_initializer)
        input_shapes = {k: v.shape for k, v in input_initializer.items()}
        output_shapes = {x.name: o.shape for x, o in zip(sess.get_outputs(), outs)}
        input_infos = get_onnx_input_infos(model_path)
        output_infos = get_onnx_output_infos(model_path)
        for k, v in input_infos.items():
            v['shape'] = input_shapes[k]
        for k, v in output_infos.items():
            v['shape'] = output_shapes[k]
        del sess
        return input_infos, output_infos

    def _setup_io_binding(self, input_infos, output_infos):
        x_ortvalues = {}
        y_ortvalues = {}
        for k, v in input_infos.items():
            m = np.zeros(**v)
            x_ortvalues[k] = ort.OrtValue.ortvalue_from_numpy(m, device_type='cuda', device_id=self.device_id)
        for k, v in output_infos.items():
            m = np.zeros(**v)
            y_ortvalues[k] = ort.OrtValue.ortvalue_from_numpy(m, device_type='cuda', device_id=self.device_id)

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
        def format_nested_dict(dict_data, indent=0):
            info = ""
            for k, v in dict_data.items():
                prefix = "  " * indent
                if isinstance(v, dict):
                    info += f"{prefix}{k}:\n" + format_nested_dict(v, indent + 1)
                elif isinstance(v, str) and v.startswith('{') and v.endswith('}'):
                    try:
                        nested_dict = eval(v)
                        if isinstance(nested_dict, dict):
                            info += f"{prefix}{k}:\n" + format_nested_dict(nested_dict, indent + 1)
                        else:
                            info += f"{prefix}{k}: {v}\n"
                    except:
                        info += f"{prefix}{k}: {v}\n"
                else:
                    info += f"{prefix}{k}: {v}\n"
            return info

        title = 'DOCSAID X ONNXRUNTIME'
        styled_title = colored.stylize(
            title, [colored.fg('blue'), colored.attr('bold')])
        divider_length = 50
        title_length = len(title)
        left_padding = (divider_length - title_length) // 2
        right_padding = divider_length - title_length - left_padding

        path = f'Model Path: {self.model_path}'
        input_info = format_nested_dict(self.input_infos)
        output_info = format_nested_dict(self.output_infos)
        metadata = format_nested_dict(self.metadata)
        providers = f'Provider: {", ".join(self.providers)}'
        provider_options = format_nested_dict(self.provider_options)

        divider = colored.stylize(
            f"+{'-' * divider_length}+", [colored.fg('blue'), colored.attr('bold')])
        infos = f'\n\n{divider}\n|{" " * left_padding}{styled_title}{" " * right_padding}|\n{divider}\n\n{path}\n\n{input_info}\n{output_info}\n\n{metadata}\n\n{providers}\n\n{provider_options}\n{divider}'
        return infos
