"""Compute-focused Explainer class used by SmartExplainer."""

import copy
from typing import Any, cast

import numpy as np
import pandas as pd

from shapash.backend import BaseBackend, get_backend_cls_from_name
from shapash.backend.shap_backend import get_shap_interaction_values
from shapash.manipulation.select_lines import keep_right_contributions
from shapash.manipulation.summarize import create_grouped_features_values
from shapash.utils.check import (
    check_additional_data,
    check_columns_order,
    check_features_name,
    check_label_dict,
    check_model,
    check_postprocessing,
    check_y,
)
from shapash.utils.explanation_metrics import find_neighbors, get_distance, get_min_nb_features, shap_neighbors
from shapash.utils.model import predict, predict_error, predict_proba
from shapash.utils.transform import apply_postprocessing, handle_categorical_missing, inverse_transform


class Explainer:
    """
    Compute engine carrying explainability state and logic.

    This class centralizes all model-explanation computations used by
    SmartExplainer: contributions handling, postprocessing, local/global
    metrics, and tabular exports.
    """

    def __init__(
        self,
        model: Any,
        backend: str | BaseBackend = "shap",
        preprocessing: Any | None = None,
        postprocessing: dict[str, dict[str, Any]] | None = None,
        features_groups: dict[str, list[str]] | None = None,
        features_dict: dict[str, str] | None = None,
        label_dict: dict[Any, Any] | None = None,
        title_story: str | None = None,
        **backend_kwargs: Any,
    ) -> None:
        if features_dict is not None and not isinstance(features_dict, dict):
            raise ValueError("features_dict must be a dict")
        if label_dict is not None and not isinstance(label_dict, dict):
            raise ValueError("label_dict must be a dict")

        self.model = model
        self.preprocessing = preprocessing
        self.backend_name: str | None = None
        self.backend: BaseBackend | None = None
        if isinstance(backend, str):
            self.backend_name = backend
        elif isinstance(backend, BaseBackend):
            self.backend = backend
            if backend.preprocessing is None and self.preprocessing is not None:
                self.backend.preprocessing = self.preprocessing
        else:
            raise NotImplementedError(f"Unknown backend : {backend}")

        self.backend_kwargs: dict[str, Any] = backend_kwargs
        self.features_dict: dict[str, str] = dict() if features_dict is None else copy.deepcopy(features_dict)
        self.label_dict: dict[Any, Any] | None = label_dict
        self.title_story = title_story if title_story is not None else ""

        self._case, self._classes = check_model(self.model)
        self.postprocessing: dict[str, dict[str, Any]] | None = postprocessing
        self.check_label_dict()
        self.inv_label_dict: dict[Any, Any] = {}
        if self.label_dict:
            self.inv_label_dict = {v: k for k, v in self.label_dict.items()}

        self.features_groups: dict[str, list[str]] | None = features_groups
        self.local_neighbors: dict[str, np.ndarray] | None = None
        self.features_stability: dict[str, np.ndarray] | None = None
        self.features_compacity: dict[str, np.ndarray] | None = None
        self.contributions: pd.DataFrame | list[pd.DataFrame] | None = None
        self.explain_data: dict[str, Any] | None = None
        self.features_imp: pd.Series | list[pd.Series] | None = None
        self.features_imp_groups: pd.Series | list[pd.Series] | None = None
        self.state: Any = None
        self.data: dict[str, pd.DataFrame | list[pd.DataFrame]] = {}
        self.data_groups: dict[str, pd.DataFrame | list[pd.DataFrame]] = {}
        self.columns_dict: dict[int, str] = {}
        self.columns_dict_groups: dict[int, str] = {}
        self.features_desc: dict[str, int] = {}
        self.additional_features_dict: dict[str, str] = {}
        self.additional_data: pd.DataFrame | None = None
        self.columns_order: list[str] | None = None
        self.postprocessing_modifications: bool = False
        self.mask: pd.DataFrame | list[pd.DataFrame] = pd.DataFrame()
        self.masked_contributions: pd.DataFrame | list[pd.DataFrame] = pd.DataFrame()
        self.mask_params: dict[str, list[Any] | float | bool | int | None] = {
            "features_to_hide": None,
            "threshold": None,
            "positive": None,
            "max_contrib": None,
        }
        self.x_encoded: pd.DataFrame = pd.DataFrame()
        self.x_init: pd.DataFrame = pd.DataFrame()
        self.x_init_groups: pd.DataFrame = pd.DataFrame()
        self.x_contrib_plot: pd.DataFrame | None = None
        self.y_pred: pd.Series | pd.DataFrame | None = None
        self.proba_values: pd.Series | pd.DataFrame | None = None
        self.y_target: pd.Series | pd.DataFrame | None = None
        self.prediction_error: pd.Series | pd.DataFrame | np.ndarray | None = None
        self.x_interaction: pd.DataFrame | None = None
        self.interaction_values: np.ndarray | None = None
        self.plot: Any = None

    def compile(
        self,
        x: pd.DataFrame,
        contributions: Any | None = None,
        y_pred: pd.Series | pd.DataFrame | None = None,
        proba_values: pd.Series | pd.DataFrame | None = None,
        y_target: pd.Series | pd.DataFrame | None = None,
        columns_order: list[str] | None = None,
        additional_data: pd.DataFrame | None = None,
        additional_features_dict: dict[str, str] | None = None,
    ) -> None:
        """
        Prepare and structure all data needed to interpret the model predictions.

        This method is the main initialization step of the explanation workflow.
        It validates and aligns model outputs, computes contributions (or formats
        provided contributions), applies inverse preprocessing/postprocessing, and
        materializes all internal structures used by plots and exports.

        Parameters
        ----------
        x : pandas.DataFrame
            Dataset used for predictions and explanations.
        contributions : pandas.DataFrame or numpy.ndarray or list, optional
            Local feature contributions. If omitted, contributions are computed
            with the configured backend.
        y_pred : pandas.Series or pandas.DataFrame or numpy.ndarray, optional
            Predicted values. If omitted and model exposes predict, predictions
            are computed automatically.
        proba_values : pandas.Series or pandas.DataFrame or numpy.ndarray, optional
            Predicted probabilities for classification.
        y_target : pandas.Series or pandas.DataFrame or numpy.ndarray, optional
            Ground-truth target values.
        columns_order : list of str, optional
            Column order used in app/report displays.
        additional_data : pandas.DataFrame, optional
            Extra non-model columns to expose in visualization contexts.
        additional_features_dict : dict of str to str, optional
            Display-name mapping for additional_data columns.

        Returns
        -------
        None
            Results are stored on the instance (data, contributions, metadata,
            masks/ordering helpers, and related state).
        """
        if isinstance(self.backend_name, str):
            backend_cls = get_backend_cls_from_name(self.backend_name)
            self.backend = backend_cls(
                model=self.model, preprocessing=self.preprocessing, masker=x, **self.backend_kwargs
            )

        self.x_encoded = handle_categorical_missing(x)
        x_init = inverse_transform(self.x_encoded, self.preprocessing)
        self.x_init = handle_categorical_missing(x_init)
        self.y_pred = check_y(self.x_init, y_pred, y_name="y_pred")
        if (self.y_pred is None) and (hasattr(self.model, "predict")):
            self.predict()

        self.proba_values = check_y(self.x_init, proba_values, y_name="proba_values")
        if (self._case == "classification") and (self.proba_values is None) and (hasattr(self.model, "predict_proba")):
            self.predict_proba()

        self.y_target = check_y(self.x_init, y_target, y_name="y_target")
        self.prediction_error = predict_error(
            self.y_target, self.y_pred, self._case, proba_values=self.proba_values, classes=self._classes
        )

        self._get_contributions_from_backend_or_user(x, contributions)
        self.check_contributions()

        self.columns_dict = {i: col for i, col in enumerate(self.x_init.columns)}
        self.check_features_dict()
        self.inv_features_dict = {v: k for k, v in self.features_dict.items()}
        self._apply_all_postprocessing_modifications()

        self.data = self.state.assign_contributions(self.state.rank_contributions(self.contributions, self.x_init))
        self.features_desc = dict(self.x_init.nunique())
        if self.features_groups is not None:
            self._compile_features_groups(self.features_groups)
        self.additional_features_dict = (
            dict()
            if additional_features_dict is None
            else self._compile_additional_features_dict(additional_features_dict)
        )
        self.additional_data = self._compile_additional_data(additional_data)
        self.columns_order = self._compile_columns_order(columns_order)
        if hasattr(self, "plot"):
            self.plot._tuning_round_digit()

    def _get_contributions_from_backend_or_user(self, x: pd.DataFrame, contributions: Any | None) -> None:
        if self.backend is None:
            raise RuntimeError("Backend is not initialized")

        if contributions is None:
            self.explain_data = self.backend.run_explainer(x=x)
            self.contributions = self.backend.get_local_contributions(x=x, explain_data=self.explain_data)
        else:
            self.explain_data = {"contributions": contributions}
            self.contributions = self.backend.format_and_aggregate_local_contributions(x=x, contributions=contributions)
        self.state = self.backend.state

    def _apply_all_postprocessing_modifications(self) -> None:
        postprocessing = self.modify_postprocessing(self.postprocessing)
        check_postprocessing(self.x_init, postprocessing)
        self.postprocessing_modifications = self.check_postprocessing_modif_strings(postprocessing)
        self.postprocessing = postprocessing
        if self.postprocessing_modifications:
            self.x_contrib_plot = copy.deepcopy(self.x_init)
        self.x_init = self.apply_postprocessing(postprocessing)

    def _compile_features_groups(self, features_groups: dict[str, list[str]]) -> None:
        if self.backend is None:
            raise RuntimeError("Backend is not initialized")

        if self.backend.support_groups is False:
            raise AssertionError(f"Selected backend ({self.backend.name}) does not support groups of features.")
        self.contributions_groups = self.state.compute_grouped_contributions(self.contributions, features_groups)
        self.features_imp_groups = None
        self._update_features_dict_with_groups(features_groups=features_groups)
        self.x_init_groups = create_grouped_features_values(
            x_init=self.x_init,
            x_encoded=self.x_encoded,
            preprocessing=self.preprocessing,
            features_groups=self.features_groups,
            features_dict=self.features_dict,
            how="dict_of_values",
        )
        self.data_groups = self.state.assign_contributions(
            self.state.rank_contributions(self.contributions_groups, self.x_init_groups)
        )
        self.columns_dict_groups = {i: col for i, col in enumerate(self.x_init_groups.columns)}

    def _compile_additional_features_dict(self, additional_features_dict: dict[str, str]) -> dict[str, str]:
        if not isinstance(additional_features_dict, dict):
            raise ValueError("additional_features_dict must be a dict")
        return {f"_{key}": f"_{value}" for key, value in additional_features_dict.items()}

    def _compile_additional_data(self, additional_data: pd.DataFrame | None) -> pd.DataFrame | None:
        if additional_data is not None:
            check_additional_data(self.x_init, additional_data)
            for feature in additional_data.columns:
                if feature in self.features_dict.keys() and feature not in self.columns_dict.values():
                    self.additional_features_dict[f"_{feature}"] = f"_{self.features_dict[feature]}"
                    del self.features_dict[feature]
            additional_data = additional_data.add_prefix("_")
            for feature in set(list(additional_data.columns)) - set(self.additional_features_dict):
                self.additional_features_dict[feature] = feature
        return additional_data

    def _compile_columns_order(self, columns_order: list[str] | None) -> list[str] | None:
        if isinstance(columns_order, list):
            check_columns_order(columns_order)
            columns_order = [f"_{col}" if f"_{col}" in self.additional_features_dict else col for col in columns_order]

            x_cols = set(self.x_encoded.columns)
            additional_cols = set(self.additional_features_dict)
            columns_order_set = set(columns_order)

            missing_cols = x_cols - columns_order_set
            extra_cols = columns_order_set - x_cols - additional_cols

            if missing_cols:
                raise ValueError(f"The following columns are missing from columns_order: {missing_cols}")
            if extra_cols:
                raise ValueError(
                    f"The following columns in columns_order do not exist in x or additional data: {extra_cols}"
                )

        return columns_order

    def check_postprocessing_modif_strings(self, postprocessing: dict[str, dict[str, Any]] | None = None) -> bool:
        """Return whether postprocessing turns numeric displayed values into strings."""
        modif = False
        if postprocessing is not None:
            for key in postprocessing.keys():
                dict_postprocess = postprocessing[key]
                if dict_postprocess["type"] in {"prefix", "suffix"} and pd.api.types.is_numeric_dtype(self.x_init[key]):
                    modif = True
        return modif

    def modify_postprocessing(
        self, postprocessing: dict[str, dict[str, Any]] | None = None
    ) -> dict[str, dict[str, Any]] | None:
        """Normalize postprocessing keys so they reference actual dataset feature names."""
        if postprocessing:
            new_dic = dict()
            for key in postprocessing.keys():
                if key in self.features_dict:
                    new_dic[key] = postprocessing[key]
                elif isinstance(key, int) and key in self.columns_dict.keys():
                    new_dic[self.columns_dict[key]] = postprocessing[key]
                elif key in self.inv_features_dict:
                    new_dic[self.inv_features_dict[key]] = postprocessing[key]
                else:
                    raise ValueError(f"Feature name '{key}' not found in the dataset.")
            return new_dic
        return None

    def apply_postprocessing(self, postprocessing: dict[str, dict[str, Any]] | None = None) -> pd.DataFrame:
        """Apply postprocessing rules to the inverse-transformed dataset view."""
        if postprocessing:
            return apply_postprocessing(self.x_init, postprocessing)
        return self.x_init

    def check_label_dict(self) -> None:
        """Validate the optional label mapping against the model problem type and classes."""
        if self._case != "regression":
            check_label_dict(self.label_dict, self._case, self._classes)

    def check_features_dict(self) -> None:
        """Align business feature names with the current compiled dataset columns."""
        dataset_features = set(self.columns_dict.values())
        current_features = set(self.features_dict.keys())

        for feature in current_features - dataset_features:
            self.features_dict.pop(feature, None)

        for feature in dataset_features - current_features:
            self.features_dict[feature] = feature

    def _update_features_dict_with_groups(self, features_groups: dict[str, list[str]]) -> None:
        for group_name in features_groups.keys():
            self.features_desc[group_name] = 1000
            if group_name not in self.features_dict.keys():
                self.features_dict[group_name] = group_name
                self.inv_features_dict[group_name] = group_name

    def check_contributions(self) -> None:
        """Ensure computed contributions match the compiled dataset shape and order."""
        if not self.state.check_contributions(self.contributions, self.x_init):
            raise ValueError(
                """
                Prediction set and contributions should have exactly the same number of lines
                and number of columns. the order of the columns must be the same
                Please check x, contributions and preprocessing arguments.
                """
            )

    def check_label_name(self, label: Any, origin: str | None = None) -> tuple[int, Any, Any]:
        """Resolve a label identifier into numeric, model-code, and display representations."""
        if origin is None:
            if label in self._classes:
                origin = "code"
            elif self.label_dict is not None and label in self.label_dict.values():
                origin = "value"
            elif isinstance(label, int) and label in range(-1, len(self._classes)):
                origin = "num"

        try:
            if origin == "num":
                label_num = label
                label_code = self._classes[label]
                label_value = self.label_dict[label_code] if self.label_dict else label_code
            elif origin == "code":
                label_code = label
                label_num = self._classes.index(label)
                label_value = self.label_dict[label_code] if self.label_dict else label_code
            elif origin == "value":
                label_code = self.inv_label_dict[label]
                label_num = self._classes.index(label_code)
                label_value = label
            else:
                raise ValueError
        except ValueError as err:
            raise Exception({"message": "Origin must be 'num', 'code' or 'value'."}) from err
        except Exception as err:
            raise Exception({"message": f"Label ({label}) not found for origin ({origin})"}) from err

        return label_num, label_code, label_value

    def check_features_name(self, features: list[Any], use_groups: bool = False) -> list[int]:
        """Resolve feature names or aliases into contribution column positions."""
        columns_dict = self.columns_dict if use_groups is False else self.columns_dict_groups
        return check_features_name(columns_dict, self.features_dict, features)

    def add(
        self,
        y_pred: pd.Series | pd.DataFrame | None = None,
        proba_values: pd.Series | pd.DataFrame | None = None,
        y_target: pd.Series | pd.DataFrame | None = None,
        label_dict: dict[Any, Any] | None = None,
        features_dict: dict[str, str] | None = None,
        title_story: str | None = None,
        columns_order: list[str] | None = None,
        additional_data: pd.DataFrame | None = None,
        additional_features_dict: dict[str, str] | None = None,
    ) -> None:
        """
        Add or update prediction outputs and metadata without recomputing contributions.

        Use this method after compile when only business metadata or prediction
        vectors need updates (for example custom thresholds, renaming labels,
        or adding extra display columns).

        Parameters
        ----------
        y_pred : pandas.Series or pandas.DataFrame or numpy.ndarray, optional
            Predicted values aligned with x_init index.
        proba_values : pandas.Series or pandas.DataFrame or numpy.ndarray, optional
            Predicted probabilities aligned with x_init index.
        y_target : pandas.Series or pandas.DataFrame or numpy.ndarray, optional
            Ground-truth values aligned with x_init index.
        label_dict : dict, optional
            Mapping from model labels to readable labels.
        features_dict : dict, optional
            Mapping from technical feature names to business names.
        title_story : str, optional
            Title used by storytelling/reporting views.
        columns_order : list of str, optional
            Display order for columns.
        additional_data : pandas.DataFrame, optional
            Extra data aligned with x_init index.
        additional_features_dict : dict of str to str, optional
            Display-name mapping for additional_data columns.

        Returns
        -------
        None
        """
        if y_pred is not None:
            self.y_pred = check_y(self.x_init, y_pred, y_name="y_pred")
        if proba_values is not None:
            self.proba_values = check_y(self.x_init, proba_values, y_name="proba_values")
        if y_target is not None:
            self.y_target = check_y(self.x_init, y_target, y_name="y_target")
        if hasattr(self, "y_target") and self.y_target is not None:
            self.prediction_error = predict_error(
                self.y_target, self.y_pred, self._case, proba_values=self.proba_values, classes=self._classes
            )
        if label_dict is not None:
            if isinstance(label_dict, dict) is False:
                raise ValueError("label_dict must be a dict")
            self.label_dict = label_dict
            self.check_label_dict()
            self.inv_label_dict = {v: k for k, v in self.label_dict.items()}
        if features_dict is not None:
            if isinstance(features_dict, dict) is False:
                raise ValueError("features_dict must be a dict")
            self.features_dict = features_dict
            self.check_features_dict()
            self.inv_features_dict = {v: k for k, v in self.features_dict.items()}
        if title_story is not None:
            self.title_story = title_story
        if additional_features_dict is not None:
            self.additional_features_dict = self._compile_additional_features_dict(additional_features_dict)
        if additional_data is not None:
            self.additional_data = self._compile_additional_data(additional_data)
        if columns_order is not None:
            self.columns_order = self._compile_columns_order(columns_order)

    def get_interaction_values(
        self, n_samples_max: int | None = None, selection: list[Any] | None = None
    ) -> np.ndarray:
        """
        Compute SHAP interaction values on the encoded dataset.

        Parameters
        ----------
        n_samples_max : int, optional
            Maximum number of rows to use.
        selection : list, optional
            Explicit row indices to keep before applying n_samples_max.

        Returns
        -------
        numpy.ndarray
            Interaction tensor with shape (n_samples, n_features, n_features).
        """
        x = copy.deepcopy(self.x_encoded)

        if selection:
            x = x.loc[selection]

        if self.x_interaction is not None:
            if self.x_interaction.equals(x[:n_samples_max]):
                if self.interaction_values is None:
                    raise RuntimeError("interaction_values cache is unexpectedly empty")
                return self.interaction_values

        self.x_interaction = x[:n_samples_max]
        if self.backend is None:
            raise RuntimeError("Backend is not initialized")
        backend_explainer = getattr(self.backend, "explainer", None)
        self.interaction_values = get_shap_interaction_values(self.x_interaction, cast(Any, backend_explainer))
        return self.interaction_values

    def filter(
        self,
        features_to_hide: list[Any] | None = None,
        threshold: float | None = None,
        positive: bool | None = None,
        max_contrib: int | None = None,
        display_groups: bool | None = None,
    ) -> None:
        """
        Apply filtering rules to local contributions and store the resulting mask.

        This controls which contributions are shown in local summaries and
        downstream exports.

        Parameters
        ----------
        features_to_hide : list, optional
            Features (or feature indices) to hide.
        threshold : float, optional
            Absolute contribution threshold under which values are hidden.
        positive : bool, optional
            If True, keep only positive contributions. If False, keep only
            negative contributions. If None, keep both signs.
        max_contrib : int, optional
            Maximum number of contributions to keep.
        display_groups : bool, optional
            If True and groups are available, apply filtering on grouped
            contributions.

        Returns
        -------
        None
            Stores mask, masked_contributions, and mask_params attributes.
        """
        display_groups = True if (display_groups is not False and self.features_groups is not None) else False
        if display_groups:
            data = self.data_groups
        else:
            data = self.data
        mask = [self.state.init_mask(data["contrib_sorted"], True)]
        if features_to_hide:
            features_list = features_to_hide
            if not all(isinstance(feature, int) for feature in features_to_hide):
                features_list = self.check_features_name(features_to_hide, use_groups=display_groups)
            mask.append(
                self.state.hide_contributions(
                    data["var_dict"],
                    features_list=features_list,
                )
            )
        if threshold:
            mask.append(self.state.cap_contributions(data["contrib_sorted"], threshold=threshold))
        if positive is not None:
            mask.append(self.state.sign_contributions(data["contrib_sorted"], positive=positive))
        self.mask = self.state.combine_masks(mask)
        if max_contrib:
            self.mask = self.state.cutoff_contributions(self.mask, max_contrib=max_contrib)
        self.masked_contributions = self.state.compute_masked_contributions(data["contrib_sorted"], self.mask)
        self.mask_params = {
            "features_to_hide": features_to_hide,
            "threshold": threshold,
            "positive": positive,
            "max_contrib": max_contrib,
        }

    def predict_proba(self) -> None:
        """
        Compute and store prediction probabilities for x_encoded.

        This method calls model predict_proba on encoded features and writes the
        result into proba_values.

        Returns
        -------
        None
        """
        self.proba_values = predict_proba(self.model, self.x_encoded, self._classes)

    def predict(self) -> None:
        """
        Compute and store predictions for x_encoded.

        This method calls model predict on encoded features and writes the result
        into y_pred. If y_target exists, prediction_error is recomputed.

        Returns
        -------
        None
        """
        self.y_pred = predict(self.model, self.x_encoded)
        if hasattr(self, "y_target"):
            self.prediction_error = predict_error(
                self.y_target, self.y_pred, self._case, proba_values=self.proba_values, classes=self._classes
            )

    def to_pandas(
        self,
        features_to_hide: list[Any] | None = None,
        threshold: float | None = None,
        positive: bool | None = None,
        max_contrib: int | None = None,
        proba: bool = False,
        use_groups: bool | None = None,
    ) -> pd.DataFrame:
        """
        Export a local-explanations summary as a pandas DataFrame.

        The output combines prediction information with top feature
        contributions for each row. If no filtering arguments are provided,
        the last stored filter parameters can be reused when compatible.

        Parameters
        ----------
        features_to_hide : list, optional
            Features to hide from local summaries.
        threshold : float, optional
            Absolute contribution threshold.
        positive : bool, optional
            Contribution-sign filter.
        max_contrib : int, optional
            Maximum number of contributions retained.
        proba : bool, default=False
            If True, include predicted probabilities.
        use_groups : bool, optional
            If True and available, summarize grouped contributions.

        Returns
        -------
        pandas.DataFrame
            Concatenation of prediction columns and formatted contribution
            summary columns.

        Raises
        ------
        ValueError
            If predictions are missing.
        """
        use_groups = True if (use_groups is not False and self.features_groups is not None) else False
        if use_groups:
            data = self.data_groups
        else:
            data = self.data

        if self.y_pred is None:
            raise ValueError("You have to specify y_pred argument. Please use add() or compile() method")

        is_compatible_cached_mask = (
            isinstance(data["contrib_sorted"], pd.DataFrame)
            and isinstance(self.mask, pd.DataFrame)
            and len(data["contrib_sorted"].columns) == len(self.mask.columns)
        ) or (
            isinstance(data["contrib_sorted"], list)
            and isinstance(self.mask, list)
            and len(data["contrib_sorted"][0].columns) == len(self.mask[0].columns)
        )

        if (
            all(var is None for var in [features_to_hide, threshold, positive, max_contrib])
            and hasattr(self, "mask_params")
            and is_compatible_cached_mask
        ):
            print("to_pandas params: " + str(self.mask_params))
        else:
            self.filter(
                features_to_hide=features_to_hide,
                threshold=threshold,
                positive=positive,
                max_contrib=max_contrib,
                display_groups=use_groups,
            )
        if use_groups:
            columns_dict = {i: col for i, col in enumerate(self.x_init_groups.columns)}
        else:
            columns_dict = self.columns_dict
        data["summary"] = self.state.summarize(
            data["contrib_sorted"], data["var_dict"], data["x_sorted"], self.mask, columns_dict, self.features_dict
        )
        if proba:
            self.predict_proba()
            proba_values = self.proba_values
        else:
            proba_values = None

        y_pred, summary = keep_right_contributions(
            self.y_pred, data["summary"], self._case, self._classes, self.label_dict, proba_values
        )

        return pd.concat([y_pred, summary], axis=1)

    def compute_features_import(self, force: bool = False, local: bool = False) -> None:
        """
        Compute relative feature importances from contribution magnitudes.

        Parameters
        ----------
        force : bool, default=False
            Kept for API compatibility.
        local : bool, default=False
            If True, also compute additional local-level importance variants.

        Returns
        -------
        None
            Stores features_imp and optional grouped/local variants.
        """
        if self.backend is None:
            raise RuntimeError("Backend is not initialized")

        self.features_imp = self.backend.get_global_features_importance(
            contributions=self.contributions, explain_data=self.explain_data, subset=None, norm=1
        )

        if self.features_groups is not None and self.features_imp_groups is None:
            self.features_imp_groups = self.state.compute_features_import(self.contributions_groups, norm=1)

        if local:
            self.features_imp_local_lev1 = self.backend.get_global_features_importance(
                contributions=self.contributions, explain_data=self.explain_data, subset=None, norm=3
            )
            self.features_imp_local_lev2 = self.backend.get_global_features_importance(
                contributions=self.contributions, explain_data=self.explain_data, subset=None, norm=7
            )
            if self.features_groups is not None:
                self.features_imp_groups_local_lev1 = self.state.compute_features_import(
                    self.contributions_groups, norm=3
                )
                self.features_imp_groups_local_lev2 = self.state.compute_features_import(
                    self.contributions_groups, norm=7
                )

    def compute_features_stability(self, selection: list[Any]) -> None:
        """
        Compute neighborhood-based stability metrics for selected rows.

        For one selected row, stores normalized contributions with neighbors in
        local_neighbors. For multiple rows, stores amplitude and variability
        arrays in features_stability.

        Parameters
        ----------
        selection : list
            Row indices to analyze.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If problem is multi-class classification.
        """
        if (self._case == "classification") and (len(self._classes) > 2):
            raise AssertionError("Multi-class classification is not supported")

        all_neighbors = find_neighbors(selection, self.x_encoded, self.model, self._case)

        if len(selection) == 1:
            norm_shap, _, _ = shap_neighbors(all_neighbors[0], self.x_encoded, self.contributions, self._case)
            self.local_neighbors = {"norm_shap": norm_shap}
        else:
            numb_expl = len(selection)
            amplitude = np.zeros((numb_expl, self.x_init.shape[1]))
            variability = np.zeros((numb_expl, self.x_init.shape[1]))
            for i in range(numb_expl):
                (_, variability[i, :], amplitude[i, :]) = shap_neighbors(
                    all_neighbors[i], self.x_encoded, self.contributions, self._case
                )
            self.features_stability = {"variability": variability, "amplitude": amplitude}

    def compute_features_compacity(self, selection: list[Any], distance: float, nb_features: int) -> None:
        """
        Compute compacity metrics for selected rows.

        Compacity measures how well predictions can be approximated with a
        reduced number of explanatory features.

        Parameters
        ----------
        selection : list
            Row indices to analyze.
        distance : float
            Target approximation level.
        nb_features : int
            Number of features used to estimate reached approximation.

        Returns
        -------
        None
            Stores features_needed and distance_reached in features_compacity.

        Raises
        ------
        AssertionError
            If problem is multi-class classification.
        """
        if (self._case == "classification") and (len(self._classes) > 2):
            raise AssertionError("Multi-class classification is not supported")

        features_needed = get_min_nb_features(selection, self.contributions, self._case, distance)
        distance_reached = get_distance(selection, self.contributions, self._case, nb_features)
        distance_reached = np.clip(distance_reached, 0, 1)

        self.features_compacity = {"features_needed": features_needed, "distance_reached": distance_reached}

    def _local_pred(self, index: Any, label: int | None = None) -> Any | None:
        """Return the local predicted value or class probability for one sample."""
        if self._case == "classification":
            if self.proba_values is not None:
                value = self.proba_values.iloc[:, [label]].loc[index].values[0]
            else:
                value = None
        elif self._case == "regression":
            if self.y_pred is not None:
                value = self.y_pred.loc[index]
            else:
                value = self.model.predict(self.x_encoded.loc[[index]])[0]

        if isinstance(value, pd.Series):
            value = value.values[0]

        return value
