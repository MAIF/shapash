"""Framework bridges for explicit tabular DeepExplainer and GradientExplainer."""

from contextlib import contextmanager
from typing import Any, Literal

import numpy as np
import pandas as pd
import shap

from shapash._optional import import_optional_module
from shapash.model.tabular import KerasModelAdapter, PyTorchModelAdapter, TensorFlowModelAdapter


class NeuralShapUnsupportedError(ValueError):
    """The adapter has no supported differentiable SHAP bridge."""


def is_neural_compatibility_error(error: Exception) -> bool:
    """Recognize unsupported operations without hiding input or resource errors."""
    if isinstance(error, (NeuralShapUnsupportedError, NotImplementedError)):
        return True
    message = str(error).lower()
    if isinstance(error, LookupError):
        return "gradient registry" in message or "no gradient defined" in message
    if isinstance(error, AssertionError):
        return "explanations do not sum up" in message
    if isinstance(error, RuntimeError):
        return any(
            text in message
            for text in (
                "does not require grad",
                "not have been used in the graph",
                "not implemented for",
                "backward hook",
            )
        )
    return False


class NeuralShap:
    """Keep tensors and framework objects out of the generic backend path."""

    def __init__(
        self, adapter: Any, background: pd.DataFrame, method: Literal["deep", "gradient"], args: dict[str, Any]
    ) -> None:
        self.adapter = adapter
        self.background_columns = list(background.columns)
        self.is_torch = isinstance(adapter, PyTorchModelAdapter)
        if not isinstance(adapter, (KerasModelAdapter, TensorFlowModelAdapter, PyTorchModelAdapter)):
            raise ValueError("Neural SHAP requires KerasModelAdapter, TensorFlowModelAdapter or PyTorchModelAdapter.")
        # Validate prediction shapes and probability semantics before constructing gradients.
        prediction = (
            adapter.predict_proba(background) if adapter.task == "classification" else adapter.predict(background)
        )
        self.base = np.asarray(prediction).mean(axis=0)
        if self.is_torch:
            from shapash.backend._torch_tabular import TorchExplanationModel  # noqa: PLC0415

            self.model = TorchExplanationModel(adapter)
        else:
            if isinstance(adapter, KerasModelAdapter):
                keras = import_optional_module("keras")
                if keras.backend.backend() != "tensorflow":
                    raise NeuralShapUnsupportedError(
                        "Neural SHAP for Keras requires the TensorFlow backend; use agnostic SHAP otherwise."
                    )
            tf = import_optional_module("tensorflow")
            if not isinstance(adapter.model, tf.keras.Model):
                raise NeuralShapUnsupportedError(
                    "Neural SHAP for TensorFlow requires a tf.keras.Model; "
                    "tf.Module and SavedModel callables support agnostic SHAP only."
                )
            inputs = tf.keras.Input(shape=(adapter.n_features_in_,), dtype="float32")
            outputs = adapter.model(inputs, training=False)
            if adapter.output_key is not None:
                outputs = outputs[adapter.output_key]
            if len(outputs.shape) == 1:
                outputs = tf.keras.ops.expand_dims(outputs, axis=-1)
            if adapter.task == "classification" and adapter.output == "logits":
                activation = "sigmoid" if outputs.shape[-1] == 1 else "softmax"
                outputs = tf.keras.layers.Activation(activation)(outputs)
            self.model = tf.keras.Model([inputs], outputs)
        cls = shap.DeepExplainer if method == "deep" else shap.GradientExplainer
        with self.inference_context():
            self.explainer = cls(self.model, self.tensor(background), **args)

    def tensor(self, x: pd.DataFrame) -> Any:
        """Use the same input conversion as the prediction adapter."""
        values = x.to_numpy(dtype=np.float32, copy=True)
        if not self.is_torch:
            return values
        torch = import_optional_module("torch")
        reference = next(self.adapter.model.parameters(), None)
        if reference is None:
            reference = next(self.adapter.model.buffers(), None)
        device = reference.device if reference is not None else "cpu"
        dtype = reference.dtype if reference is not None and reference.is_floating_point() else torch.float32
        return torch.as_tensor(values, device=device, dtype=dtype)

    @contextmanager
    def inference_context(self):
        """Allow gradients but preserve the user's torch training flags and gradients."""
        if not self.is_torch:
            # AutoGraph wraps missing-gradient LookupErrors in its own exception.
            autograph = import_optional_module("tensorflow.python.autograph.impl.api")
            execute = import_optional_module("tensorflow.python.eager.execute")
            record_gradient = execute.record_gradient
            try:
                yield
            except autograph.StagingError as error:
                if "gradient registry" not in str(error).lower():
                    raise
                raise NeuralShapUnsupportedError(str(error)) from error
            finally:
                # Deep SHAP restores this global callback only on successful calls.
                execute.record_gradient = record_gradient
            return
        torch = import_optional_module("torch")
        states = [(module, module.training) for module in self.model.modules()]
        gradients = [(parameter, parameter.grad) for parameter in self.model.parameters()]
        # Deep SHAP does not remove its hooks/temporary parameters when gradients fail.
        hook_names = ("_forward_hooks", "_backward_hooks", "_forward_pre_hooks", "_backward_pre_hooks")
        hooks = [(module, name, set(getattr(module, name))) for module, _ in states for name in hook_names]
        attributes = [
            (module, name, hasattr(module, name), getattr(module, name, None))
            for module, _ in states
            for name in ("x", "y")
        ]
        backward_flags = [(module, module._is_full_backward_hook) for module, _ in states]
        try:
            self.model.eval()
            with torch.enable_grad():
                yield
        finally:
            for module, name, existing in hooks:
                registry = getattr(module, name)
                for key in set(registry) - existing:
                    del registry[key]
            for module, flag in backward_flags:
                module._is_full_backward_hook = flag
            for module, name, existed, value in attributes:
                if existed:
                    setattr(module, name, value)
                elif hasattr(module, name):
                    delattr(module, name)
            for module, training in states:
                module.training = training
            for parameter, gradient in gradients:
                parameter.grad = gradient

    def explain(self, x: pd.DataFrame, args: dict[str, Any]) -> dict[str, Any]:
        """Normalize SHAP's single-output and per-class layouts for Shapash."""
        if list(x.columns) != list(self.background_columns):
            raise ValueError("Neural SHAP input columns must match the encoded background columns in the same order.")
        if self.adapter.task == "classification":
            self.adapter.predict_proba(x)
        else:
            self.adapter.predict(x)
        with self.inference_context():
            values = self.explainer.shap_values(self.tensor(x), **args)
        if isinstance(values, tuple):
            raise ValueError("ranked_outputs/variance outputs are unsupported; explain all model outputs.")
        if isinstance(values, list):
            values = np.stack(values, axis=-1)
        values = np.asarray(values)
        if values.ndim == 3 and values.shape[-1] == 1:
            values = values[..., 0]
        if values.shape[:2] != x.shape or values.ndim not in (2, 3) or not np.isfinite(values).all():
            raise ValueError("Neural SHAP returned invalid tabular contributions.")
        base = np.asarray(self.base)
        if self.adapter.task == "classification":
            n_classes = len(self.adapter.classes_)
            if values.ndim == 2 and n_classes == 2:
                # The single output is P(second class); the first class is its complement.
                values = np.stack([-values, values], axis=-1)
            elif values.ndim != 3 or values.shape[-1] != n_classes:
                raise ValueError("Neural SHAP contributions do not match the model classes.")
            base = np.broadcast_to(base, (len(x), n_classes)).copy()
        else:
            if values.ndim != 2:
                raise ValueError("Neural SHAP regression requires a single output.")
            base = np.full(len(x), float(base))
        return {"contributions": values, "base_values": base}
