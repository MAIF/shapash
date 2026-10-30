from contextlib import nullcontext
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from shapash.backend._neural_shap import (
    NeuralShap,
    NeuralShapUnsupportedError,
    is_neural_compatibility_error,
)


@pytest.mark.parametrize(
    "error, compatible",
    [
        (NeuralShapUnsupportedError("unsupported"), True),
        (NotImplementedError("unsupported"), True),
        (KeyError("No gradient defined for operation"), True),
        (KeyError("missing output"), False),
        (AssertionError("explanations do not sum up"), True),
        (AssertionError("invalid shape"), False),
        (RuntimeError("does not require grad"), True),
        (RuntimeError("not have been used in the graph"), True),
        (RuntimeError("not implemented for this device"), True),
        (RuntimeError("backward hook"), True),
        (RuntimeError("out of memory"), False),
        (ValueError("invalid input"), False),
    ],
)
def test_is_neural_compatibility_error(error, compatible):
    assert is_neural_compatibility_error(error) is compatible


def _neural_shap(values, *, task="regression", classes=None, base=0.25):
    explainer = object.__new__(NeuralShap)
    explainer.adapter = Mock(task=task, classes_=classes)
    explainer.background_columns = ["a", "b"]
    explainer.is_torch = False
    explainer.base = np.asarray(base)
    explainer.explainer = Mock()
    explainer.explainer.shap_values.return_value = values
    explainer.inference_context = nullcontext
    explainer.tensor = lambda x: x.to_numpy(dtype=np.float32, copy=True)
    return explainer


def test_neural_shap_rejects_non_adapter():
    with pytest.raises(ValueError, match="requires KerasModelAdapter"):
        NeuralShap(object(), pd.DataFrame({"a": [1.0]}), "deep", {})


def test_explain_normalizes_regression_values():
    x = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["a", "b"])
    neural = _neural_shap(np.array([[0.1, 0.2], [0.3, 0.4]]))

    result = neural.explain(x, {"check_additivity": False})

    np.testing.assert_array_equal(result["contributions"], [[0.1, 0.2], [0.3, 0.4]])
    np.testing.assert_array_equal(result["base_values"], [0.25, 0.25])
    neural.adapter.predict.assert_called_once_with(x)
    neural.explainer.shap_values.assert_called_once()


def test_explain_normalizes_binary_and_multiclass_values():
    x = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["a", "b"])
    binary = _neural_shap(
        np.array([[[0.1], [0.2]], [[0.3], [0.4]]]),
        task="classification",
        classes=[0, 1],
        base=[0.6, 0.4],
    )
    binary_result = binary.explain(x, {})
    np.testing.assert_array_equal(
        binary_result["contributions"],
        [[[-0.1, 0.1], [-0.2, 0.2]], [[-0.3, 0.3], [-0.4, 0.4]]],
    )
    np.testing.assert_array_equal(binary_result["base_values"], [[0.6, 0.4], [0.6, 0.4]])
    binary.adapter.predict_proba.assert_called_once_with(x)

    multiclass = _neural_shap(
        [np.ones((2, 2)), np.full((2, 2), 2), np.full((2, 2), 3)],
        task="classification",
        classes=["a", "b", "c"],
        base=[0.2, 0.3, 0.5],
    )
    result = multiclass.explain(x, {})
    assert result["contributions"].shape == (2, 2, 3)
    np.testing.assert_array_equal(result["base_values"], [[0.2, 0.3, 0.5]] * 2)


@pytest.mark.parametrize(
    "values, message",
    [
        ((np.zeros((2, 2)), np.zeros((2, 2))), "ranked_outputs"),
        (np.zeros((1, 2)), "invalid tabular contributions"),
        (np.array([[0.0, np.nan], [0.0, 0.0]]), "invalid tabular contributions"),
    ],
)
def test_explain_rejects_invalid_values(values, message):
    x = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["a", "b"])
    neural = _neural_shap(values)
    with pytest.raises(ValueError, match=message):
        neural.explain(x, {})


def test_explain_rejects_column_and_class_mismatches():
    x = pd.DataFrame([[1.0, 2.0]], columns=["b", "a"])
    neural = _neural_shap(np.zeros((1, 2)))
    with pytest.raises(ValueError, match="columns must match"):
        neural.explain(x, {})

    x = pd.DataFrame([[1.0, 2.0]], columns=["a", "b"])
    neural = _neural_shap(np.zeros((1, 2, 3)), task="classification", classes=[0, 1])
    with pytest.raises(ValueError, match="do not match the model classes"):
        neural.explain(x, {})


def test_explain_rejects_multitarget_regression():
    x = pd.DataFrame([[1.0, 2.0]], columns=["a", "b"])
    neural = _neural_shap(np.zeros((1, 2, 2)))
    with pytest.raises(ValueError, match="single output"):
        neural.explain(x, {})


def test_torch_neural_shap_constructor_uses_model_dtype_and_restores_state():
    torch = pytest.importorskip("torch")
    from shapash.model import PyTorchModelAdapter

    model = torch.nn.Linear(2, 1, dtype=torch.float64)
    model.train()
    adapter = PyTorchModelAdapter(model, n_features=2, task="regression")
    background = pd.DataFrame([[0.0, 1.0], [1.0, 2.0]], columns=["a", "b"])
    shap_explainer = object()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("shapash.backend._neural_shap.shap.DeepExplainer", lambda *args, **kwargs: shap_explainer)
        neural = NeuralShap(adapter, background, "deep", {})

    assert neural.explainer is shap_explainer
    assert model.training
    tensor = neural.tensor(background)
    assert tensor.dtype == torch.float64
    assert tensor.device == next(model.parameters()).device


def test_torch_explanation_model_applies_binary_and_multiclass_logits():
    torch = pytest.importorskip("torch")
    from shapash.backend._torch_tabular import TorchExplanationModel

    class Model(torch.nn.Module):
        def forward(self, values):
            return values

    values = torch.tensor([[0.0], [2.0]])
    binary = TorchExplanationModel(Mock(model=Model(), task="classification", output="logits", output_key=None))
    torch.testing.assert_close(binary(values), torch.sigmoid(values))

    multiclass_values = torch.tensor([[1.0, 2.0, 3.0]])
    multiclass = TorchExplanationModel(Mock(model=Model(), task="classification", output="logits", output_key=None))
    torch.testing.assert_close(multiclass(multiclass_values), torch.softmax(multiclass_values, dim=-1))


def test_torch_explanation_model_selects_dictionary_output_and_expands_regression():
    torch = pytest.importorskip("torch")
    from shapash.backend._torch_tabular import TorchExplanationModel

    class Model(torch.nn.Module):
        def forward(self, values):
            return {"score": values[:, 0]}

    wrapped = TorchExplanationModel(Mock(model=Model(), task="regression", output="probabilities", output_key="score"))
    assert wrapped(torch.tensor([[1.0, 2.0]])).shape == (1, 1)
