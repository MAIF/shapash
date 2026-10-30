"""Unit tests for shapash.webapp.nlp_components.data_editor: DataEditorComponent."""

import unittest

from shapash.webapp.nlp_app import NlpWebApp

from tests.unit_tests.webapp.nlp_components._shared import FakeEngine, _callback


class TestDataEditorComponentCallbacks(unittest.TestCase):
    """The editor's own callbacks: prefill from a row/applied counterfactual, and Predict."""

    def _app(self):
        engine = FakeEngine(can_edit=True, can_cf=False)
        return NlpWebApp(engine.to_explanation(), engine=engine)

    def test_prefill_prevents_update_without_a_row_or_an_applied_text(self):
        from unittest import mock

        from dash.exceptions import PreventUpdate
        from shapash.webapp.nlp_components import data_editor as data_editor_module

        app = self._app()
        prefill = _callback(app, "data-editor-input.value")
        with mock.patch.object(data_editor_module, "callback_context") as cc:
            cc.triggered = [{"prop_id": "dataset-table.selectedRows"}]
            with self.assertRaises(PreventUpdate):
                prefill(None, None)

    def test_prefill_from_a_selected_row(self):
        from unittest import mock

        from shapash.webapp.nlp_components import data_editor as data_editor_module

        app = self._app()
        prefill = _callback(app, "data-editor-input.value")
        with mock.patch.object(data_editor_module, "callback_context") as cc:
            cc.triggered = [{"prop_id": "dataset-table.selectedRows"}]
            self.assertEqual(prefill([{"text": "a selected row"}], None), "a selected row")

    def test_prefill_from_an_applied_counterfactual(self):
        from unittest import mock

        from shapash.webapp.nlp_components import data_editor as data_editor_module

        app = self._app()
        prefill = _callback(app, "data-editor-input.value")
        with mock.patch.object(data_editor_module, "callback_context") as cc:
            cc.triggered = [{"prop_id": "whatif-apply-store.data"}]
            self.assertEqual(prefill(None, "an applied counterfactual"), "an applied counterfactual")

    def test_predict_prevents_update_without_a_click_or_text(self):
        from dash.exceptions import PreventUpdate

        app = self._app()
        predict = _callback(app, "data-editor-prob.figure")
        with self.assertRaises(PreventUpdate):
            predict(None, "some text")
        with self.assertRaises(PreventUpdate):
            predict(1, "")
        with self.assertRaises(PreventUpdate):
            predict(1, "   ")
