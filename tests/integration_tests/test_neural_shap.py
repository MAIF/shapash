"""Real neural explainers through tabular adapters and production export."""

import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from shapash import SmartExplainer
from shapash.utils.load_smartpredictor import load_smartpredictor
from tests.integration_tests.test_tabular_deep_learning import _build_adapter


@pytest.mark.parametrize("framework", ["keras", "tensorflow", "torch"])
@pytest.mark.parametrize("method", ["auto", "deep", "gradient", "agnostic"])
@pytest.mark.parametrize("with_preprocessing", [False, True])
@pytest.mark.parametrize(
    "task, output",
    [
        ("regression", "probabilities"),
        ("binary", "probabilities"),
        ("binary", "logits"),
        ("binary_two", "probabilities"),
        ("multiclass", "probabilities"),
        ("multiclass", "logits"),
    ],
)
def test_neural_shap_workflow(framework, method, with_preprocessing, task, output, tmp_path):
    x = pd.DataFrame(np.random.default_rng(42).normal(size=(12, 3)), columns=["a", "b", "c"], index=range(100, 112))
    preprocessing = None
    encoded = x
    if with_preprocessing:
        preprocessing = ColumnTransformer([("scale", StandardScaler(), ["c", "a", "b"])]).fit(x)
        encoded = pd.DataFrame(preprocessing.transform(x), columns=["col_0", "col_1", "col_2"], index=x.index)
    adapter, _, _ = _build_adapter(framework, 3, task, output)
    options = {"nsamples": 2000, "rseed": 42} if method == "gradient" else {}
    explainer = SmartExplainer(
        model=adapter,
        preprocessing=preprocessing,
        neural_explainer=method,
        neural_background=encoded.iloc[:8],
        explainer_compute_args=options,
    )
    sample = encoded.iloc[8:10]
    explainer.compile(x=sample)
    backend = explainer.explainer.backend
    expected_explainer = {
        "auto": shap.DeepExplainer,
        "deep": shap.DeepExplainer,
        "gradient": shap.GradientExplainer,
        "agnostic": shap.ExactExplainer,
    }[method]
    assert isinstance(backend.explainer, expected_explainer)
    expected = adapter.predict(sample) if task == "regression" else adapter.predict_proba(sample)
    data = backend.explain_data if backend.explain_data is not None else explainer.explainer.explain_data
    contributions = explainer.explainer.contributions
    if task == "regression":
        reconstructed = contributions.sum(axis=1).to_numpy() + data["base_values"]
    else:
        assert len(contributions) == len(adapter.classes_)
        reconstructed = np.column_stack([frame.sum(axis=1) for frame in contributions]) + data["base_values"]
    np.testing.assert_allclose(reconstructed, expected, atol=0.1 if method == "gradient" else 1e-4, rtol=0.02)
    predictor = explainer.to_smartpredictor()
    predictor.add_input(x=x.iloc[8:10])
    np.testing.assert_array_equal(predictor.predict().iloc[:, 0], adapter.predict(sample))
    assert predictor.data["contributions"].shape == (2, 3)
    assert np.isfinite(predictor.data["contributions"].to_numpy(dtype=float)).all()
    assert len(predictor.detail_contributions()) == 2
    assert len(explainer.plot.features_importance().data) > 0
    path = str(tmp_path / "predictor.pkl")
    predictor.save(path)
    restored = load_smartpredictor(path)
    restored.add_input(x=x.iloc[8:10])
    np.testing.assert_array_equal(restored.predict().iloc[:, 0], adapter.predict(sample))
    np.testing.assert_allclose(
        restored.data["contributions"].to_numpy(dtype=float),
        predictor.data["contributions"].to_numpy(dtype=float),
        atol=1e-5,
    )


@pytest.mark.parametrize("method", ["deep", "gradient"])
def test_neural_torch_preserves_training_and_parameter_gradients(method):
    torch = pytest.importorskip("torch")
    from shapash.model import PyTorchModelAdapter
    from shapash.backend import ShapBackend

    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.ReLU(), torch.nn.Dropout(0.5), torch.nn.Linear(3, 1))
    model.train()
    model[1].eval()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    states = [module.training for module in model.modules()]
    gradients = [parameter.grad.clone() for parameter in model.parameters()]
    adapter = PyTorchModelAdapter(model, n_features=2, task="regression")
    x = pd.DataFrame(np.random.default_rng(0).normal(size=(6, 2)))
    backend = ShapBackend(adapter, masker=x, neural_explainer=method)
    data = backend.run_explainer(x.iloc[:2])
    assert data["contributions"].shape == (2, 2)
    assert [module.training for module in model.modules()] == states
    for parameter, gradient in zip(model.parameters(), gradients):
        torch.testing.assert_close(parameter.grad, gradient)


def test_auto_torch_falls_back_to_gradient_on_additivity_failure():
    torch = pytest.importorskip("torch")
    from shapash.model import PyTorchModelAdapter
    from shapash.backend import ShapBackend

    class Square(torch.nn.Module):
        def forward(self, values):
            return values.square()

    model = torch.nn.Sequential(torch.nn.Linear(2, 1), Square())
    with torch.no_grad():
        model[0].weight.fill_(0.5)
        model[0].bias.zero_()
    adapter = PyTorchModelAdapter(model, n_features=2, task="regression")
    background = pd.DataFrame([[0.0, 0.0], [1.0, 1.0]], columns=["a", "b"])
    sample = pd.DataFrame([[2.0, 2.0]], columns=["a", "b"])
    backend = ShapBackend(adapter, neural_background=background)
    with pytest.warns(UserWarning, match="Falling back to GradientExplainer"):
        data = backend.run_explainer(sample)
    assert isinstance(backend.explainer, shap.GradientExplainer)
    data = backend._neural.explain(sample, {"nsamples": 5000, "rseed": 42})
    np.testing.assert_allclose(
        data["contributions"].sum(axis=1) + data["base_values"], adapter.predict(sample), atol=0.1
    )


def test_auto_torch_gradient_failure_cleans_hooks_before_agnostic():
    torch = pytest.importorskip("torch")
    from shapash.model import PyTorchModelAdapter
    from shapash.backend import ShapBackend

    class Detached(torch.nn.Module):
        def forward(self, values):
            return values.detach()

    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.ReLU(), torch.nn.Linear(3, 1), Detached())
    model.train()
    model[1].eval()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    states = [module.training for module in model.modules()]
    gradients = [parameter.grad.clone() for parameter in model.parameters()]
    hook = model[0].register_forward_hook(lambda module, args, output: None)
    original_hooks = [dict(module._forward_hooks) for module in model.modules()]
    flags = [module._is_full_backward_hook for module in model.modules()]
    adapter = PyTorchModelAdapter(model, n_features=2, task="regression")
    x = pd.DataFrame([[0.0, 1.0], [1.0, 2.0]], columns=["a", "b"])
    backend = ShapBackend(adapter, masker=x)
    with pytest.warns(UserWarning) as recorded:
        data = backend.run_explainer(x)
    assert any("Falling back to GradientExplainer" in str(item.message) for item in recorded)
    assert any("Falling back to agnostic SHAP" in str(item.message) for item in recorded)
    assert isinstance(backend.explainer, shap.ExactExplainer)
    np.testing.assert_allclose(data["contributions"].sum(axis=1) + data["base_values"], adapter.predict(x))
    assert [module.training for module in model.modules()] == states
    assert [dict(module._forward_hooks) for module in model.modules()] == original_hooks
    assert [module._is_full_backward_hook for module in model.modules()] == flags
    for module in model.modules():
        assert not module._backward_hooks
        assert not hasattr(module, "x")
        assert not hasattr(module, "y")
    for parameter, gradient in zip(model.parameters(), gradients):
        torch.testing.assert_close(parameter.grad, gradient)
    hook.remove()


def test_auto_tensorflow_unsupported_deep_operation_uses_gradient():
    tf = pytest.importorskip("tensorflow")
    from shapash.model import TensorFlowModelAdapter
    from shapash.backend import ShapBackend

    inputs = tf.keras.Input(shape=(2,))
    outputs = tf.keras.ops.sum(tf.keras.ops.sin(inputs), axis=1, keepdims=True)
    adapter = TensorFlowModelAdapter(tf.keras.Model(inputs, outputs), n_features=2, task="regression")
    background = pd.DataFrame([[0.0, 0.0], [0.1, 0.2]], columns=["a", "b"])
    sample = pd.DataFrame([[0.3, 0.4]], columns=["a", "b"])
    backend = ShapBackend(adapter, neural_background=background)
    with pytest.warns(UserWarning, match="Falling back to GradientExplainer"):
        data = backend.run_explainer(sample)
    assert isinstance(backend.explainer, shap.GradientExplainer)
    np.testing.assert_allclose(
        data["contributions"].sum(axis=1) + data["base_values"], adapter.predict(sample), atol=0.03
    )


def test_auto_retains_gradient_selection_after_reload(tmp_path):
    from unittest.mock import patch
    from shapash.backend import ShapBackend

    adapter, _, _ = _build_adapter("torch", 2, "regression", "probabilities")
    x = pd.DataFrame([[0.0, 1.0], [1.0, 2.0]], columns=["a", "b"])
    with patch(
        "shapash.backend._neural_shap.shap.DeepExplainer", side_effect=NotImplementedError("unsupported operation")
    ):
        with pytest.warns(UserWarning, match="Falling back to GradientExplainer"):
            backend = ShapBackend(adapter, masker=x)
    explainer = SmartExplainer(model=adapter, backend=backend)
    explainer.compile(x=x)
    predictor = explainer.to_smartpredictor()
    path = str(tmp_path / "gradient.pkl")
    predictor.save(path)
    restored = load_smartpredictor(path)
    with patch("shapash.backend._neural_shap.shap.DeepExplainer") as deep:
        restored.add_input(x=x)
        deep.assert_not_called()
    assert isinstance(restored.backend.explainer, shap.GradientExplainer)


def test_auto_keras_tutorial_architecture_uses_deep():
    keras = pytest.importorskip("keras")
    from shapash.model import KerasModelAdapter

    keras.utils.set_random_seed(42)
    model = keras.Sequential(
        [
            keras.Input(shape=(10,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    adapter = KerasModelAdapter(model, n_features=10, task="classification", classes=[0, 1])
    x = pd.DataFrame(np.random.default_rng(42).normal(size=(12, 10)))
    explainer = SmartExplainer(model=adapter, neural_background=x.iloc[:8])
    explainer.compile(x=x.iloc[8:10])
    assert isinstance(explainer.explainer.backend.explainer, shap.DeepExplainer)
    data = explainer.explainer.explain_data
    np.testing.assert_allclose(
        data["contributions"].sum(axis=1) + data["base_values"], adapter.predict_proba(x.iloc[8:10]), atol=1e-5
    )


def test_auto_tensorflow_module_uses_agnostic_with_warnings():
    tf = pytest.importorskip("tensorflow")
    from shapash.model import TensorFlowModelAdapter
    from shapash.backend import ShapBackend

    class Module(tf.Module):
        def __call__(self, values):
            return tf.reduce_sum(values, axis=1)

    adapter = TensorFlowModelAdapter(Module(), n_features=2, task="regression")
    x = pd.DataFrame([[0.0, 1.0], [1.0, 2.0]], columns=["a", "b"])
    with pytest.warns(UserWarning) as recorded:
        backend = ShapBackend(adapter, masker=x)
    assert len(recorded) == 2
    assert "Falling back to GradientExplainer" in str(recorded[0].message)
    assert "Falling back to agnostic SHAP" in str(recorded[1].message)
    assert isinstance(backend.explainer, shap.ExactExplainer)
    data = backend.run_explainer(x)
    np.testing.assert_allclose(data["contributions"].sum(axis=1) + data["base_values"], adapter.predict(x))


def test_auto_retains_agnostic_fallback_after_reload(tmp_path):
    from unittest.mock import patch
    from shapash.backend import ShapBackend

    adapter, _, _ = _build_adapter("torch", 2, "regression", "probabilities")
    x = pd.DataFrame([[0.0, 1.0], [1.0, 2.0]], columns=["a", "b"])
    with patch("shapash.backend.shap_backend.NeuralShap", side_effect=NotImplementedError("unsupported operation")):
        with pytest.warns(UserWarning) as recorded:
            backend = ShapBackend(adapter, masker=x)
    assert len(recorded) == 2
    explainer = SmartExplainer(model=adapter, backend=backend)
    explainer.compile(x=x)
    predictor = explainer.to_smartpredictor()
    path = str(tmp_path / "agnostic.pkl")
    predictor.save(path)
    restored = load_smartpredictor(path)
    with patch("shapash.backend.shap_backend.NeuralShap") as bridge:
        restored.add_input(x=x)
        bridge.assert_not_called()
    assert isinstance(restored.backend.explainer, shap.ExactExplainer)
    np.testing.assert_array_equal(restored.predict().iloc[:, 0], adapter.predict(x))
