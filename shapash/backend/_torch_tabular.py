"""Torch-only differentiable output adapter, imported only for neural SHAP."""

import torch


class TorchExplanationModel(torch.nn.Module):
    """Expose the adapter's selected probabilities or regression output to SHAP."""

    def __init__(self, adapter):
        super().__init__()
        self.model = adapter.model
        self.output_key = adapter.output_key
        self.classification = adapter.task == "classification"
        self.logits = adapter.output == "logits"
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, values):
        """Return a two-dimensional, differentiable tensor in prediction space."""
        result = self.model(values)
        if self.output_key is not None:
            result = result[self.output_key]
        if result.ndim == 1:
            result = result[:, None]
        if self.classification and self.logits:
            result = self.sigmoid(result) if result.shape[1] == 1 else self.softmax(result)
        return result
