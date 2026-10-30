"""Optional deep-learning inference adapters for tabular explainability."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from sklearn.utils.metaestimators import available_if

from shapash._optional import import_optional_module


class _TabularModelAdapter:
    """Share the single-input, single-task contract across inference frameworks."""

    def __init__(
        self,
        model: Any,
        *,
        n_features: int,
        task: Literal["regression", "classification"],
        classes: Sequence[int | float | str] | None = None,
        output: Literal["probabilities", "logits"] = "probabilities",
        threshold: float = 0.5,
        output_key: str | None = None,
    ) -> None:
        """Configure inference without training or changing the model's parameters.

        ``classes`` lists labels in network-output order. Binary networks may
        return one output (the second class) or two. Multiclass networks must
        return one output per class. Regression supports one target only.
        ``output_key`` selects a tensor from dictionary-valued model outputs.

        Parameters
        ----------
        model : Any
            Trained framework model or TensorFlow inference callable.
        n_features : int
            Number of numeric features after preprocessing.
        task : {"regression", "classification"}
            Explicit task; never inferred from a network's output shape.
        classes : sequence of int, float or str, optional
            Classification labels in network-output order.
        output : {"probabilities", "logits"}
            Classification output interpretation. Regression uses the default.
        threshold : float
            Second-class probability threshold for binary classification.
        output_key : str, optional
            Dictionary key selecting a single tensor output.
        """
        if isinstance(n_features, bool) or not isinstance(n_features, int) or n_features < 1:
            raise ValueError("n_features must be a positive integer.")
        if task not in ("regression", "classification"):
            raise ValueError("task must be 'regression' or 'classification'.")
        if output not in ("probabilities", "logits"):
            raise ValueError("output must be 'probabilities' or 'logits'.")
        if not np.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("threshold must be strictly between 0 and 1.")
        if output_key is not None and not isinstance(output_key, str):
            raise ValueError("output_key must be a string or None.")
        self.model = model
        self.n_features_in_ = n_features
        self.task = task
        self.output = output
        self.threshold = threshold
        self.output_key = output_key
        if task == "classification":
            labels = (
                [label.item() if isinstance(label, np.generic) else label for label in classes]
                if classes is not None
                else []
            )
            if len(labels) < 2:
                raise ValueError("Classification requires at least two classes in network-output order.")
            if not all(isinstance(label, int | float | str) for label in labels):
                raise ValueError("Class labels must be integers, floats, or strings.")
            if any(isinstance(label, str) for label in labels) and not all(isinstance(label, str) for label in labels):
                raise ValueError("Class labels must not mix strings and numeric values.")
            if any(isinstance(label, float) and not np.isfinite(label) for label in labels):
                raise ValueError("Class labels must be finite.")
            if len(set(labels)) != len(labels):
                raise ValueError("Class labels must be unique.")
            self.classes_ = np.asarray(labels)
        elif classes is not None or output != "probabilities":
            raise ValueError("Regression does not accept classes or a classification output mode.")
        self._validate_model()

    def _validate_model(self) -> None:
        raise NotImplementedError

    def _forward(self, values: np.ndarray) -> Any:
        raise NotImplementedError

    def _to_numpy(self, result: Any) -> np.ndarray:
        return np.asarray(result)

    def _values(self, x: Any) -> np.ndarray:
        try:
            values = np.asarray(x, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError("Adapter inputs must be a numeric two-dimensional table.") from error
        if values.ndim != 2 or values.shape[1] != self.n_features_in_ or len(values) == 0:
            raise ValueError(f"Adapter inputs must have shape (n_samples, {self.n_features_in_}) with n_samples > 0.")
        if not np.isfinite(values).all():
            raise ValueError("Adapter inputs must contain finite values; preprocess missing values first.")
        result = self._forward(values)
        if self.output_key is not None:
            if not isinstance(result, dict) or self.output_key not in result:
                raise ValueError(f"Model output must be a dictionary containing {self.output_key!r}.")
            result = result[self.output_key]
        elif isinstance(result, (dict, list, tuple)):
            raise ValueError("Multiple model outputs are unsupported; select a dictionary tensor with output_key.")
        try:
            result = self._to_numpy(result)
            if np.iscomplexobj(result):
                raise ValueError("Complex model outputs are unsupported.")
            result = np.asarray(result, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("Model output must be a numeric tensor.") from error
        if result.ndim == 1:
            result = result[:, None]
        if result.ndim != 2 or result.shape[0] != len(values) or result.shape[1] == 0:
            raise ValueError("Model output must have shape (n_samples, n_outputs).")
        if not np.isfinite(result).all():
            raise ValueError("Model output must contain finite values.")
        return result

    @available_if(lambda self: self.task == "classification")
    def predict_proba(self, x: Any) -> np.ndarray:
        """Return probabilities ordered like classes_, validating the output contract."""
        values = self._values(x)
        n_classes = len(self.classes_)
        if values.shape[1] != n_classes and not (n_classes == 2 and values.shape[1] == 1):
            raise ValueError(f"Classification output must have {n_classes} columns (or one for binary classification).")
        if self.output == "logits":
            if values.shape[1] == 1:
                values = np.exp(-np.logaddexp(0, -values))
            else:
                values = np.exp(values - values.max(axis=1, keepdims=True))
                values /= values.sum(axis=1, keepdims=True)
        if (values < 0).any() or (values > 1).any():
            raise ValueError("Probability outputs must lie in [0, 1]; use output='logits' for unnormalized scores.")
        if values.shape[1] == 1:
            values = np.column_stack([1 - values[:, 0], values[:, 0]])
        if not np.allclose(values.sum(axis=1), 1, atol=1e-6, rtol=1e-6):
            raise ValueError("Probability outputs must sum to one per row; multilabel classification is unsupported.")
        return values

    def predict(self, x: Any) -> np.ndarray:
        """Return one regression value or one class label per input row."""
        if self.task == "classification":
            probabilities = self.predict_proba(x)
            indices = (
                (probabilities[:, 1] >= self.threshold).astype(int)
                if len(self.classes_) == 2
                else probabilities.argmax(axis=1)
            )
            return self.classes_[indices]
        values = self._values(x)
        if values.shape[1] != 1:
            raise ValueError("Regression supports one output only; multi-target regression is unsupported.")
        return values[:, 0]


class KerasModelAdapter(_TabularModelAdapter):
    """Adapt a Keras 3 model using its configured backend (TensorFlow, JAX or torch).

    Accepts the arguments of the shared tabular adapter: ``model``, ``n_features``,
    ``task``, ``classes``, ``output``, ``threshold`` and ``output_key``.
    Inference calls the model with ``training=False`` and returns NumPy arrays.
    Install Keras and its chosen backend separately.
    """

    def _validate_model(self) -> None:
        keras = import_optional_module("keras", "Install Keras 3 and a supported backend.")
        if not hasattr(keras, "ops"):
            raise ValueError("KerasModelAdapter requires Keras 3; use TensorFlowModelAdapter for legacy tf.keras.")
        if not isinstance(self.model, keras.Model):
            raise ValueError("KerasModelAdapter requires a keras.Model.")

    def _forward(self, values: np.ndarray) -> Any:
        return self.model(values, training=False)

    def _to_numpy(self, result: Any) -> np.ndarray:
        keras = import_optional_module("keras")
        return np.asarray(keras.ops.convert_to_numpy(result))


class TensorFlowModelAdapter(_TabularModelAdapter):
    """Adapt a TensorFlow callable, tf.Module, tf.keras model or SavedModel signature.

    Accepts the shared tabular adapter arguments. Inputs are float32 tensors.
    For SavedModel, pass an explicit positional-input callable such as
    ``loaded.signatures["serving_default"]`` and select its output with
    ``output_key``. Keyword-only signatures need an application-defined callable.
    TensorFlow modules must implement deterministic inference; tf.keras models
    are called with ``training=False``.
    """

    def _validate_model(self) -> None:
        import_optional_module("tensorflow", "Install TensorFlow to use TensorFlowModelAdapter.")
        if not callable(self.model):
            raise ValueError("TensorFlowModelAdapter requires a callable TensorFlow model or signature.")

    def _forward(self, values: np.ndarray) -> Any:
        tf = import_optional_module("tensorflow")
        tensor = tf.convert_to_tensor(values, dtype=tf.float32)
        if isinstance(self.model, tf.keras.Model):
            return self.model(tensor, training=False)
        return self.model(tensor)

    def _to_numpy(self, result: Any) -> np.ndarray:
        if not hasattr(result, "numpy"):
            raise ValueError("TensorFlow model output must be a tensor.")
        return np.asarray(result.numpy())


class PyTorchModelAdapter(_TabularModelAdapter):
    """Adapt a torch.nn.Module, using its parameter device and floating-point dtype.

    Accepts the shared tabular adapter arguments. Inference disables gradients
    and temporarily switches all submodules to evaluation mode, restoring their
    original training flags afterwards. Parameterless models use CPU float32,
    or their first buffer's device and floating-point dtype when available.
    """

    def _validate_model(self) -> None:
        torch = import_optional_module("torch", "Install PyTorch to use PyTorchModelAdapter.")
        if not isinstance(self.model, torch.nn.Module):
            raise ValueError("PyTorchModelAdapter requires a torch.nn.Module.")

    def _forward(self, values: np.ndarray) -> Any:
        torch = import_optional_module("torch")
        reference = next(self.model.parameters(), None)
        if reference is None:
            reference = next(self.model.buffers(), None)
        device = reference.device if reference is not None else "cpu"
        dtype = reference.dtype if reference is not None and reference.is_floating_point() else torch.float32
        tensor = torch.as_tensor(values, device=device, dtype=dtype)
        states = [(module, module.training) for module in self.model.modules()]
        try:
            self.model.eval()
            with torch.inference_mode():
                return self.model(tensor)
        finally:
            for module, training in states:
                module.training = training

    def _to_numpy(self, result: Any) -> np.ndarray:
        if not hasattr(result, "detach"):
            raise ValueError("PyTorch model output must be a tensor.")
        torch = import_optional_module("torch")
        result = result.detach().cpu()
        if result.dtype == torch.bfloat16:
            result = result.float()
        return result.numpy()
