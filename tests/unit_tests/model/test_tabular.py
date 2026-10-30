"""Framework-independent validation of the tabular adapter contract."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from shapash.model import KerasModelAdapter, PyTorchModelAdapter, TensorFlowModelAdapter
from shapash.model.tabular import _TabularModelAdapter


class ArrayAdapter(_TabularModelAdapter):
    def _validate_model(self):
        pass

    def _forward(self, values):
        return self.model(values)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"n_features": 0}, "positive integer"),
        ({"n_features": True}, "positive integer"),
        ({"task": "multilabel"}, "task must"),
        ({"output": "unknown"}, "output must"),
        ({"threshold": 0}, "threshold must"),
        ({"threshold": np.nan}, "threshold must"),
        ({"output_key": 1}, "output_key must"),
        ({"task": "classification"}, "at least two classes"),
        ({"task": "classification", "classes": [0, 0]}, "unique"),
        ({"task": "classification", "classes": [0, np.inf]}, "finite"),
        ({"task": "classification", "classes": [object(), object()]}, "Class labels"),
        ({"task": "classification", "classes": [1, "two"]}, "must not mix"),
        ({"classes": [0, 1]}, "Regression does not"),
        ({"output": "logits"}, "Regression does not"),
    ],
)
def test_invalid_configuration(options, message):
    with pytest.raises(ValueError, match=message):
        ArrayAdapter(lambda x: x, **({"n_features": 2, "task": "regression"} | options))


@pytest.mark.parametrize("x", [[[1]], [1, 2], [], [["a", "b"]], [[np.nan, 0]], [[0, np.inf]]])
def test_invalid_inputs(x):
    adapter = ArrayAdapter(lambda values: values[:, :1], n_features=2, task="regression")
    with pytest.raises(ValueError, match="Adapter inputs"):
        adapter.predict(x)


@pytest.mark.parametrize(
    "result, message",
    [
        (np.zeros((1, 1)), "shape"),
        (np.zeros((2, 1, 1)), "shape"),
        (np.zeros((2, 2)), "one output"),
        (np.full((2, 1), np.nan), "finite"),
        ({"score": np.zeros((2, 1))}, "Multiple model outputs"),
        ([np.zeros((2, 1))], "Multiple model outputs"),
        (np.array([["a"], ["b"]]), "numeric tensor"),
        (np.array([[1j], [2j]]), "numeric tensor"),
    ],
)
def test_invalid_regression_outputs(result, message):
    adapter = ArrayAdapter(lambda values: result, n_features=2, task="regression")
    with pytest.raises(ValueError, match=message):
        adapter.predict([[1, 2], [3, 4]])


@pytest.mark.parametrize("values", [[[1.2]], [[-0.1]], [[0.2, 0.3]], [[0.1, 0.2, 0.7]]])
def test_invalid_classification_outputs(values):
    adapter = ArrayAdapter(lambda x: np.array(values), n_features=2, task="classification", classes=[0, 1])
    with pytest.raises(ValueError, match="Probability|Classification"):
        adapter.predict_proba([[1, 2]])


def test_stable_logits_and_custom_threshold():
    adapter = ArrayAdapter(
        lambda values: np.array([[-1000], [1000], [0]]),
        n_features=2,
        task="classification",
        classes=[3, 8],
        output="logits",
        threshold=0.6,
    )
    np.testing.assert_array_equal(adapter.predict([[0, 0]] * 3), [3, 8, 3])
    np.testing.assert_allclose(adapter.predict_proba([[0, 0]] * 3), [[1, 0], [0, 1], [0.5, 0.5]])


def test_dictionary_output_selection():
    adapter = ArrayAdapter(
        lambda values: {"score": values.sum(axis=1)}, n_features=2, task="regression", output_key="score"
    )
    np.testing.assert_array_equal(adapter.predict([[1, 2]]), [3])
    adapter.output_key = "missing"
    with pytest.raises(ValueError, match="dictionary containing"):
        adapter.predict([[1, 2]])


def test_numpy_class_labels():
    adapter = ArrayAdapter(
        lambda values: np.full((len(values), 1), 0.8),
        n_features=2,
        task="classification",
        classes=np.array([3, 8]),
    )
    np.testing.assert_array_equal(adapter.predict([[1, 2]]), [8])


def test_stable_multiclass_logits():
    adapter = ArrayAdapter(
        lambda values: np.array([[1000, 0, -1000]]),
        n_features=2,
        task="classification",
        classes=[3, 8, 42],
        output="logits",
    )
    np.testing.assert_allclose(adapter.predict_proba([[0, 0]]), [[1, 0, 0]])
    np.testing.assert_array_equal(adapter.predict([[0, 0]]), [3])


@pytest.mark.parametrize("adapter_type", [KerasModelAdapter, TensorFlowModelAdapter, PyTorchModelAdapter])
def test_optional_dependency_error_is_explicit(adapter_type):
    with patch("shapash.model.tabular.import_optional_module", side_effect=ImportError("Missing optional dependency")):
        with pytest.raises(ImportError, match="Missing optional dependency"):
            adapter_type(object(), n_features=2, task="regression")


def test_keras_adapter_inference_contract():
    class Model:
        def __call__(self, values, *, training):
            assert training is False
            return values[:, :1]

    keras = SimpleNamespace(Model=Model, ops=SimpleNamespace(convert_to_numpy=np.asarray))
    with patch("shapash.model.tabular.import_optional_module", return_value=keras):
        adapter = KerasModelAdapter(Model(), n_features=2, task="regression")
        np.testing.assert_array_equal(adapter.predict([[1, 2]]), [1])


def test_keras_adapter_rejects_legacy_keras_and_invalid_models():
    with patch("shapash.model.tabular.import_optional_module", return_value=SimpleNamespace()):
        with pytest.raises(ValueError, match="requires Keras 3"):
            KerasModelAdapter(object(), n_features=2, task="regression")

    class Model:
        pass

    with patch(
        "shapash.model.tabular.import_optional_module",
        return_value=SimpleNamespace(Model=Model, ops=object()),
    ):
        with pytest.raises(ValueError, match="requires a keras.Model"):
            KerasModelAdapter(object(), n_features=2, task="regression")


def test_tensorflow_adapter_model_and_callable_inference():
    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values)

        def __getitem__(self, item):
            return self.values[item]

        def numpy(self):
            return self.values

    class KerasModel:
        def __call__(self, values, *, training):
            assert training is False
            return Tensor(values[:, :1])

    tf = SimpleNamespace(
        float32=np.float32,
        convert_to_tensor=lambda values, dtype: Tensor(values.astype(dtype)),
        keras=SimpleNamespace(Model=KerasModel),
    )
    with patch("shapash.model.tabular.import_optional_module", return_value=tf):
        model_adapter = TensorFlowModelAdapter(KerasModel(), n_features=2, task="regression")
        callable_adapter = TensorFlowModelAdapter(lambda values: Tensor(values[:, :1]), n_features=2, task="regression")
        np.testing.assert_array_equal(model_adapter.predict([[1, 2]]), [1])
        np.testing.assert_array_equal(callable_adapter.predict([[3, 4]]), [3])


def test_tensorflow_adapter_rejects_non_tensor_outputs():
    tf = SimpleNamespace(
        float32=np.float32,
        convert_to_tensor=lambda values, dtype: np.asarray(values, dtype=dtype),
        keras=SimpleNamespace(Model=type("KerasModel", (), {})),
    )
    with patch("shapash.model.tabular.import_optional_module", return_value=tf):
        adapter = TensorFlowModelAdapter(lambda values: values, n_features=2, task="regression")
        with pytest.raises(ValueError, match="numeric tensor"):
            adapter.predict([[1, 2]])


def test_pytorch_adapter_inference_and_output_conversion():
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)

        def forward(self, values):
            return self.linear(values)

    model = Model()
    model.train()
    adapter = PyTorchModelAdapter(model, n_features=2, task="regression")
    result = adapter.predict([[1, 2]])
    assert result.dtype == np.float64
    assert model.training
    with pytest.raises(ValueError, match="output must be a tensor"):
        adapter._to_numpy(np.ones((1, 1)))
    np.testing.assert_array_equal(
        adapter._to_numpy(torch.tensor([[1.0]], dtype=torch.bfloat16)),
        [[1.0]],
    )


def test_pytorch_adapter_parameterless_and_buffer_dtype():
    torch = pytest.importorskip("torch")

    class Parameterless(torch.nn.Module):
        def forward(self, values):
            return values[:, :1]

    class Buffered(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("reference", torch.ones(1, dtype=torch.float64))

        def forward(self, values):
            return values[:, :1]

    for model, dtype in ((Parameterless(), torch.float32), (Buffered(), torch.float64)):
        adapter = PyTorchModelAdapter(model, n_features=2, task="regression")
        assert adapter._forward(np.ones((1, 2), dtype=np.float32)).dtype == dtype
