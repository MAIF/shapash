"""Real-framework tabular adapters, prediction backends and production export."""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from shapash import SmartExplainer
from shapash.model import KerasModelAdapter, PyTorchModelAdapter, TensorFlowModelAdapter
from shapash.utils.load_smartpredictor import load_smartpredictor


def _build_adapter(framework, n_features, task, output):
    n_outputs = {"regression": 1, "binary": 1, "binary_two": 2, "multiclass": 3}[task]
    weights = np.arange(n_features * n_outputs, dtype=np.float32).reshape(n_features, n_outputs) / 10 + 0.1
    bias = np.linspace(-0.2, 0.2, n_outputs, dtype=np.float32)
    if framework == "torch":
        torch = pytest.importorskip("torch")
        model = torch.nn.Sequential(torch.nn.Linear(n_features, n_outputs))
        with torch.no_grad():
            model[0].weight.copy_(torch.tensor(weights.T))
            model[0].bias.copy_(torch.tensor(bias))
        if task != "regression" and output == "probabilities":
            model.append(torch.nn.Sigmoid() if task == "binary" else torch.nn.Softmax(dim=-1))
        adapter_type = PyTorchModelAdapter
    else:
        keras = pytest.importorskip("keras") if framework == "keras" else pytest.importorskip("tensorflow").keras
        activation = None
        if task != "regression" and output == "probabilities":
            activation = "sigmoid" if task == "binary" else "softmax"
        model = keras.Sequential(
            [keras.Input(shape=(n_features,)), keras.layers.Dense(n_outputs, activation=activation)]
        )
        model.set_weights([weights, bias])
        adapter_type = KerasModelAdapter if framework == "keras" else TensorFlowModelAdapter
    options = {"n_features": n_features, "task": "regression" if task == "regression" else "classification"}
    if task != "regression":
        options.update(classes=[3, 8, 42] if task == "multiclass" else [3, 8], output=output)
    return adapter_type(model, **options), weights, bias


@pytest.mark.parametrize("framework", ["keras", "tensorflow", "torch"])
@pytest.mark.parametrize("backend", ["shap", "lime"])
@pytest.mark.parametrize("with_preprocessing", [False, True], ids=["raw", "column_transformer"])
@pytest.mark.parametrize(
    "task, output",
    [
        ("regression", "probabilities"),
        ("binary", "probabilities"),
        ("binary", "logits"),
        ("binary_two", "probabilities"),
        ("binary_two", "logits"),
        ("multiclass", "probabilities"),
        ("multiclass", "logits"),
    ],
)
def test_tabular_adapter_workflow(framework, backend, with_preprocessing, task, output, tmp_path):
    if backend == "lime":
        pytest.importorskip("lime")
    x = pd.DataFrame(np.random.default_rng(42).normal(size=(20, 3)), columns=["a", "b", "c"], index=range(100, 120))
    preprocessing = None
    if with_preprocessing:
        preprocessing = ColumnTransformer([("scale", StandardScaler(), ["c", "a", "b"])]).fit(x)
        encoded = pd.DataFrame(preprocessing.transform(x), columns=["col_0", "col_1", "col_2"], index=x.index)
    else:
        encoded = x
    adapter, weights, bias = _build_adapter(framework, 3, task, output)
    sample = encoded.iloc[:2]
    expected = sample.to_numpy(dtype=np.float32) @ weights + bias
    if task == "regression":
        assert not hasattr(adapter, "predict_proba")
        assert not hasattr(adapter, "classes_")
        np.testing.assert_allclose(adapter.predict(sample), expected[:, 0], atol=1e-6)
    else:
        if task == "binary":
            positive = 1 / (1 + np.exp(-expected[:, 0]))
            expected = np.column_stack([1 - positive, positive])
        else:
            expected = np.exp(expected - expected.max(axis=1, keepdims=True))
            expected /= expected.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(adapter.predict_proba(sample), expected, atol=1e-6)
        indices = expected.argmax(axis=1) if task == "multiclass" else (expected[:, 1] >= 0.5).astype(int)
        np.testing.assert_array_equal(adapter.predict(sample), adapter.classes_[indices])

    options = {"neural_explainer": "agnostic"} if backend == "shap" else {}
    explainer = SmartExplainer(model=adapter, backend=backend, preprocessing=preprocessing, **options)
    explainer.compile(x=encoded)
    compiled = explainer.explainer
    frames = compiled.contributions if isinstance(compiled.contributions, list) else [compiled.contributions]
    assert len(frames) == (1 if task == "regression" else len(adapter.classes_))
    for frame in frames:
        assert frame.shape == x.shape
        assert frame.index.equals(x.index)
        assert np.isfinite(frame.to_numpy()).all()
    if backend == "shap":
        data = compiled.explain_data
        reconstructed = data["contributions"].sum(axis=1) + data["base_values"]
        expected = adapter.predict(encoded) if task == "regression" else adapter.predict_proba(encoded)
        np.testing.assert_allclose(reconstructed, expected, atol=1e-6)
    assert len(explainer.to_pandas()) == len(x)
    assert len(explainer.plot.features_importance().data) > 0
    predictor = explainer.to_smartpredictor()
    predictor.add_input(x=x.iloc[:2])
    np.testing.assert_allclose(predictor.data["x_preprocessed"], sample)
    np.testing.assert_array_equal(predictor.predict().iloc[:, 0], adapter.predict(sample))
    assert predictor.data["contributions"].shape == (2, 3)
    assert np.isfinite(predictor.data["contributions"].to_numpy(dtype=float)).all()
    assert len(predictor.detail_contributions()) == 2
    if task != "regression":
        np.testing.assert_allclose(predictor.predict_proba(), adapter.predict_proba(sample))
    path = str(tmp_path / "predictor.pkl")
    predictor.save(path)
    restored = load_smartpredictor(path)
    restored.add_input(x=x.iloc[:2])
    np.testing.assert_array_equal(restored.predict().iloc[:, 0], adapter.predict(sample))
    from shapash.utils.io import _detect_model_framework

    assert (
        _detect_model_framework(adapter)["name"]
        == {"keras": "keras", "tensorflow": "tensorflow", "torch": "torch"}[framework]
    )


def test_tensorflow_module_and_saved_model_signature(tmp_path):
    tf = pytest.importorskip("tensorflow")

    class Module(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec([None, 2], tf.float32)])
        def __call__(self, values):
            return {"score": tf.reduce_sum(values, axis=1, keepdims=True)}

    model = Module()
    x = pd.DataFrame([[1, 2], [3, 4]])
    adapter = TensorFlowModelAdapter(model, n_features=2, task="regression", output_key="score")
    np.testing.assert_array_equal(adapter.predict(x), [3, 7])
    module_adapter = adapter
    path = str(tmp_path / "saved_model")
    tf.saved_model.save(model, path)
    loaded = tf.saved_model.load(path)
    signature = loaded.signatures["serving_default"]
    # Exported signatures use keyword-only input names.
    adapter = TensorFlowModelAdapter(
        lambda values: signature(values=values), n_features=2, task="regression", output_key="score"
    )
    np.testing.assert_array_equal(adapter.predict(x), [3, 7])
    for candidate in (module_adapter, adapter):
        for backend in ("shap", "lime"):
            pytest.importorskip(backend)
            explainer = SmartExplainer(model=candidate, backend=backend)
            explainer.compile(x=x)
            predictor = explainer.to_smartpredictor()
            predictor.add_input(x=x)
            np.testing.assert_array_equal(predictor.predict().iloc[:, 0], [3, 7])


def test_torch_inference_preserves_training_flags_and_dtype():
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Dropout(0.9)).double()
    model.train()
    model[0].eval()
    before = [module.training for module in model.modules()]
    adapter = PyTorchModelAdapter(model, n_features=2, task="regression")
    first = adapter.predict([[1, 2]])
    np.testing.assert_array_equal(first, adapter.predict([[1, 2]]))
    assert [module.training for module in model.modules()] == before
    assert model[0].weight.dtype == torch.float64
    assert model[0].weight.grad is None


def test_torch_training_flags_restored_on_inference_error():
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(3, 1), torch.nn.Dropout())
    model.train()
    model[1].eval()
    before = [module.training for module in model.modules()]
    adapter = PyTorchModelAdapter(model, n_features=2, task="regression")
    with pytest.raises(RuntimeError):
        adapter.predict([[1, 2]])
    assert [module.training for module in model.modules()] == before


def test_torch_parameterless_model_uses_buffer_dtype_and_device():
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("offset", torch.tensor(0.5, dtype=torch.float64))

        def forward(self, values):
            assert values.dtype == self.offset.dtype
            assert values.device == self.offset.device
            return values.sum(dim=1, keepdim=True) + self.offset

    adapter = PyTorchModelAdapter(Model(), n_features=2, task="regression")
    np.testing.assert_allclose(adapter.predict([[1, 2]]), [3.5])


def test_torch_bfloat16_output():
    torch = pytest.importorskip("torch")
    adapter = PyTorchModelAdapter(torch.nn.Identity(), n_features=1, task="regression")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(adapter, "_forward", lambda values: torch.tensor([[1.5]], dtype=torch.bfloat16))
        np.testing.assert_array_equal(adapter.predict([[1]]), [1.5])


def test_public_adapter_imports_do_not_load_frameworks():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from shapash.model import KerasModelAdapter, TensorFlowModelAdapter, PyTorchModelAdapter; "
                "assert not {'torch', 'tensorflow', 'keras'}.intersection(sys.modules)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_keras_torch_backend_in_separate_process():
    pytest.importorskip("torch")
    pytest.importorskip("keras")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import keras
import numpy as np
import pandas as pd
from shapash import SmartExplainer
from shapash.model import KerasModelAdapter
assert keras.backend.backend() == "torch"
model = keras.Sequential([keras.Input(shape=(2,)), keras.layers.Dense(1, activation="sigmoid")])
model.set_weights([np.array([[0.2], [0.3]], dtype="float32"), np.array([0.1], dtype="float32")])
adapter = KerasModelAdapter(model, n_features=2, task="classification", classes=[0, 1])
x = pd.DataFrame([[0., 0.], [1., 1.], [2., 2.], [3., 3.]])
for backend in ("shap", "lime"):
    xpl = SmartExplainer(model=adapter, backend=backend)
    xpl.compile(x=x)
    predictor = xpl.to_smartpredictor()
    predictor.add_input(x=x)
    np.testing.assert_allclose(predictor.predict_proba(), adapter.predict_proba(x))
""",
        ],
        env=dict(os.environ, KERAS_BACKEND="torch", OMP_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
