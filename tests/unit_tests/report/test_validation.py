from pathlib import Path

import pandas as pd
import pytest

from shapash.report.validation import (
    load_report_config,
    render_block_error,
    stats_to_table,
    validate_report_schema,
)


class TestLoadReportConfig:
    def test_missing_file_raises(self, tmp_path):
        missing = tmp_path / "missing.yml"
        with pytest.raises(FileNotFoundError):
            load_report_config(missing)

    def test_invalid_yaml_raises_value_error(self, tmp_path):
        cfg_path = tmp_path / "invalid.yml"
        cfg_path.write_text("sections: [unbalanced brackets\n")
        with pytest.raises(ValueError, match="Invalid YAML syntax"):
            load_report_config(cfg_path)

    def test_valid_yaml_returns_config(self, tmp_path):
        cfg_path = tmp_path / "valid.yml"
        cfg_path.write_text("sections:\n  - type: title\n    params: {}\n")
        cfg = load_report_config(cfg_path)
        assert cfg["sections"][0]["type"] == "title"


class TestValidateReportSchema:
    def _path(self):
        return Path("dummy.yml")

    def test_top_level_not_a_mapping(self):
        with pytest.raises(ValueError, match="top-level content must be a mapping"):
            validate_report_schema(["not", "a", "dict"], self._path())

    def test_missing_sections_key(self):
        with pytest.raises(ValueError, match="'sections' must be a non-empty list"):
            validate_report_schema({}, self._path())

    def test_empty_sections_list(self):
        with pytest.raises(ValueError, match="'sections' must be a non-empty list"):
            validate_report_schema({"sections": []}, self._path())

    def test_block_not_a_mapping(self):
        with pytest.raises(ValueError, match=r"sections\[1\] must be a mapping"):
            validate_report_schema({"sections": ["not a dict"]}, self._path())

    def test_block_missing_type(self):
        with pytest.raises(ValueError, match=r"sections\[1\].type must be a non-empty string"):
            validate_report_schema({"sections": [{"params": {}}]}, self._path())

    def test_block_params_not_a_mapping(self):
        with pytest.raises(ValueError, match=r"sections\[1\].params must be a mapping"):
            validate_report_schema({"sections": [{"type": "title", "params": []}]}, self._path())

    def test_custom_block_missing_function(self):
        with pytest.raises(ValueError, match="function is required for custom blocks"):
            validate_report_schema({"sections": [{"type": "custom", "params": {}}]}, self._path())

    def test_custom_block_with_function_is_valid(self):
        validate_report_schema(
            {"sections": [{"type": "custom", "params": {}, "function": "mymodule.myfunc"}]}, self._path()
        )

    def test_group_block_blocks_not_a_list(self):
        with pytest.raises(ValueError, match=r"sections\[1\].blocks must be a list for group blocks"):
            validate_report_schema({"sections": [{"type": "group", "params": {}, "blocks": "nope"}]}, self._path())

    def test_group_block_recurses_into_children(self):
        with pytest.raises(ValueError, match=r"sections\[1\].blocks\[1\].type must be a non-empty string"):
            validate_report_schema(
                {"sections": [{"type": "group", "params": {}, "blocks": [{"params": {}}]}]}, self._path()
            )

    def test_group_block_with_valid_children_is_valid(self):
        validate_report_schema(
            {"sections": [{"type": "group", "params": {}, "blocks": [{"type": "title", "params": {}}]}]},
            self._path(),
        )


class TestRenderBlockError:
    def test_render_block_error_returns_alert_pane(self):
        pane = render_block_error("my_block", ValueError("boom"))
        assert pane.alert_type == "danger"
        assert "my_block" in pane.object
        assert "boom" in pane.object


class TestStatsToTable:
    def test_without_train_stats(self):
        table = stats_to_table({"mean": 1.0}, names=["test"])
        assert list(table.columns) == ["test"]
        assert table.loc["mean", "test"] == 1.0

    def test_with_train_stats(self):
        table = stats_to_table({"mean": 1.0}, names=["test", "train"], train_stats={"mean": 2.0})
        assert list(table.columns) == ["train", "test"]

    def test_drops_all_nan_columns(self):
        table = stats_to_table({"mean": 1.0}, names=["test", "train"], train_stats={"missing_stat": pd.NA})
        assert "train" not in table.columns
        assert list(table.columns) == ["test"]
