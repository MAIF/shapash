"""Generate Shapash HTML reports for Titanic and house prices examples.

Examples:
    python scripts/generate_report.py --dataset titanic --case classification
    python scripts/generate_report.py --dataset titanic --case multiclass
    python scripts/generate_report.py --dataset house_prices --case regression --report-mode custom
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from category_encoders import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from shapash import SmartExplainer
from shapash.data.data_loader import data_loading
from shapash.utils.dtypes import text_like_columns

TITANIC_CASES = {
    "classification": {
        "features": ["Pclass", "Age", "Sex", "SibSp", "Parch"],
        "target": "Survived",
        "target_names": {0: "Did not survive", 1: "Survived"},
    },
    "regression": {
        "features": ["Pclass", "Survived", "Embarked", "Sex", "Age", "SibSp", "Parch"],
        "target": "Fare",
        "target_names": None,
    },
    "multiclass": {
        "features": ["Survived", "Age", "Embarked", "Sex"],
        "target": "Pclass",
        "target_names": {1: "First class", 2: "Second class", 3: "Third class"},
    },
}


def _load_task_data(dataset: str, case: str):
    """Return feature data, target, encoder, labels, and split settings."""
    if dataset == "titanic":
        if case not in TITANIC_CASES:
            raise ValueError(f"Unsupported Titanic task: {case}")

        data, labels = data_loading("titanic")
        data["Pclass"] = data["Pclass"].map({"First class": 1, "Second class": 2, "Third class": 3})
        task = TITANIC_CASES[case]
        X = data[task["features"]]
        y = data[task["target"]]
        categorical_features = [col for col in ("Embarked", "Sex") if col in X.columns]
        encoder = OneHotEncoder(cols=categorical_features, use_cat_names=True)
        labels = {name: label for name, label in labels.items() if name in task["features"]}
        target_names = task["target_names"]
        test_size = 0.2
    elif dataset == "house_prices":
        data, labels = data_loading("house_prices")
        sale_price = data["SalePrice"]
        X = data.drop(columns="SalePrice")
        if case == "regression":
            y = sale_price
            target_names = None
        else:
            class_count = 2 if case == "classification" else 3
            y = pd.qcut(sale_price, q=class_count, labels=False).astype("int64")
            target_names = (
                {0: "Lower-priced", 1: "Higher-priced"}
                if case == "classification"
                else {0: "Low price", 1: "Mid-range price", 2: "High price"}
            )
        encoder = OrdinalEncoder(
            cols=text_like_columns(X, strict_object=False),
            handle_unknown="return_nan",
            return_df=True,
        )
        test_size = 0.25
    elif dataset == "telco_customer_churn":
        data = data_loading("telco_customer_churn").reset_index()
        data = data.drop(columns="customerID", errors="ignore")
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce").fillna(0)

        if case == "classification":
            target = "Churn"
            y = data[target].map({"No": 0, "Yes": 1}).astype("int64")
            target_names = {0: "No churn", 1: "Churn"}
        elif case == "multiclass":
            target = "InternetService"
            classes = sorted(data[target].dropna().unique())
            class_ids = {label: index for index, label in enumerate(classes)}
            y = data[target].map(class_ids).astype("int64")
            target_names = dict(enumerate(classes))
        else:
            target = "MonthlyCharges"
            y = data[target]
            target_names = None

        X = data.drop(columns=target)
        encoder = OrdinalEncoder(
            cols=text_like_columns(X, strict_object=False),
            handle_unknown="return_nan",
            return_df=True,
        )
        labels = None
        test_size = 0.25
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return X, y, encoder, labels, target_names, test_size


def _build_custom_blocks(case: str, dataset: str, explainer, X_train, y_train, y_test):
    """Create task-specific report blocks while sharing the report runtime."""
    import panel as pn  # noqa: PLC0415

    from shapash.report.blocks import ReportBlockMixin, block  # noqa: PLC0415

    if case == "multiclass":

        class MulticlassReportBlocks(ReportBlockMixin):
            @block
            def block_performance_metrics(self, title="Metrics", metrics=None):
                if self.y_test is None or self.y_pred is None:
                    raise ValueError("performance_metrics block requires y_test and y_pred.")

                table = pd.DataFrame(
                    [
                        ["Accuracy", accuracy_score(self.y_test, self.y_pred)],
                        ["Precision (macro)", precision_score(self.y_test, self.y_pred, average="macro")],
                        ["Recall (macro)", recall_score(self.y_test, self.y_pred, average="macro")],
                        ["F1 (macro)", f1_score(self.y_test, self.y_pred, average="macro")],
                    ],
                    columns=["Metric", "Value"],
                )
                metrics_pane = pn.pane.DataFrame(table, index=False, width_policy="min", sizing_mode="stretch_width")
                return title, [metrics_pane]

        return MulticlassReportBlocks(explainer=explainer, x_train=X_train, y_train=y_train, y_test=y_test)

    if dataset == "titanic" and case == "classification":

        class TitanicClassificationReportBlocks(ReportBlockMixin):
            @block
            def block_prediction_error_summary(self, title="Prediction error summary"):
                if self.y_test is None or self.y_pred is None:
                    raise ValueError("prediction_error_summary block requires y_test and y_pred.")

                y_true = pd.Series(self.y_test).reset_index(drop=True)
                y_pred = pd.Series(self.y_pred).reset_index(drop=True)
                true_positive = int(((y_true == 1) & (y_pred == 1)).sum())
                true_negative = int(((y_true == 0) & (y_pred == 0)).sum())
                false_positive = int(((y_true == 0) & (y_pred == 1)).sum())
                false_negative = int(((y_true == 1) & (y_pred == 0)).sum())
                table = pd.DataFrame(
                    [
                        ["Accuracy", f"{accuracy_score(y_true, y_pred):.3f}"],
                        ["Precision (Survived)", f"{precision_score(y_true, y_pred, zero_division=0):.3f}"],
                        ["Recall (Survived)", f"{recall_score(y_true, y_pred, zero_division=0):.3f}"],
                        ["False positives", false_positive],
                        ["False negatives", false_negative],
                        ["True positives", true_positive],
                        ["True negatives", true_negative],
                    ],
                    columns=["Metric", "Value"],
                )
                return title, ["Summary of classifier errors and survival metrics.", table]

            @block
            def block_misclassification_focus(self, title="Most confident misclassifications", top_k=10):
                report_explainer = self._require_explainer("misclassification_focus")
                if self.y_test is None or self.y_pred is None:
                    raise ValueError("misclassification_focus block requires y_test and y_pred.")
                if report_explainer.proba_values is None:
                    report_explainer.predict_proba()
                if report_explainer.proba_values.shape[1] < 2:
                    raise ValueError("misclassification_focus block requires binary class probabilities.")

                y_true = pd.Series(self.y_test, index=report_explainer.x_init.index, name="true")
                y_pred = pd.Series(self.y_pred, index=report_explainer.x_init.index, name="pred")
                probability = report_explainer.proba_values.iloc[:, 1].rename("proba_survived")
                analysis = pd.concat([y_true, y_pred, probability, report_explainer.x_init], axis=1)
                errors = analysis[analysis["true"] != analysis["pred"]].copy()
                if errors.empty:
                    return title, ["No misclassification found on the evaluated dataset."]

                errors["wrong_confidence"] = errors.apply(
                    lambda row: row["proba_survived"] if row["pred"] == 1 else 1 - row["proba_survived"],
                    axis=1,
                )
                errors = errors.sort_values("wrong_confidence", ascending=False).head(top_k)
                columns = ["true", "pred", "proba_survived", "wrong_confidence"]
                columns.extend(col for col in ["Pclass", "Sex", "Age", "Fare", "Embarked", "Title"] if col in errors)
                table = errors[columns].rename(
                    columns={"true": "True", "pred": "Pred", "proba_survived": "P(Survived)"}
                )
                table = table.reset_index(drop=False)
                for column in ["P(Survived)", "wrong_confidence"]:
                    table[column] = table[column].map(lambda value: round(float(value), 3))
                return title, ["Most confident incorrect predictions for targeted error analysis.", table]

        return TitanicClassificationReportBlocks(explainer=explainer, x_train=X_train, y_train=y_train, y_test=y_test)

    if dataset == "house_prices" and case == "regression":

        class HousePricesRegressionReportBlocks(ReportBlockMixin):
            @block
            def block_residual_error_summary(self, title="Residual error summary"):
                if self.y_test is None or self.y_pred is None:
                    raise ValueError("residual_error_summary block requires y_test and y_pred.")

                y_true = pd.Series(self.y_test).reset_index(drop=True)
                y_pred = pd.Series(self.y_pred).reset_index(drop=True)
                residuals = y_true - y_pred
                absolute_errors = residuals.abs()
                table = pd.DataFrame(
                    [
                        ["MAE", f"{mean_absolute_error(y_true, y_pred):,.2f}"],
                        ["MSE", f"{mean_squared_error(y_true, y_pred):,.2f}"],
                        ["R2", f"{r2_score(y_true, y_pred):.3f}"],
                        ["Residual mean", f"{residuals.mean():,.2f}"],
                        ["Residual std", f"{residuals.std():,.2f}"],
                        ["Median absolute error", f"{absolute_errors.median():,.2f}"],
                        ["95th pct absolute error", f"{absolute_errors.quantile(0.95):,.2f}"],
                    ],
                    columns=["Metric", "Value"],
                )
                return title, ["Summary of global regression errors and residual dispersion.", table]

            @block
            def block_largest_errors_focus(self, title="Largest absolute errors", top_k=10):
                report_explainer = self._require_explainer("largest_errors_focus")
                if self.y_test is None or self.y_pred is None:
                    raise ValueError("largest_errors_focus block requires y_test and y_pred.")

                y_true = pd.Series(self.y_test, index=report_explainer.x_init.index, name="true")
                y_pred = pd.Series(self.y_pred, index=report_explainer.x_init.index, name="pred")
                details = pd.concat([y_true, y_pred, report_explainer.x_init], axis=1)
                details["residual"] = details["true"] - details["pred"]
                details["abs_error"] = details["residual"].abs()
                focus = details.sort_values("abs_error", ascending=False).head(top_k).copy()
                if focus.empty:
                    return title, ["No rows available to compute largest errors."]

                focus = focus.rename(
                    columns={"true": "True", "pred": "Pred", "residual": "Residual", "abs_error": "AbsError"}
                )
                leading = ["True", "Pred", "Residual", "AbsError"]
                preferred = ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageArea", "Neighborhood"]
                context = [column for column in preferred if column in focus]
                trailing = [column for column in focus if column not in leading + context]
                focus = focus[leading + context + trailing].reset_index(drop=False)
                for column in leading:
                    focus[column] = focus[column].map(lambda value: round(float(value), 2))
                return title, ["Largest residuals to prioritize model and feature investigation.", focus]

        return HousePricesRegressionReportBlocks(explainer=explainer, x_train=X_train, y_train=y_train, y_test=y_test)

    return None


def generate_report(
    dataset: str,
    case: str,
    report_mode: str = "default",
    output_file: str | Path | None = None,
) -> Path:
    """Train one supported example model and export its Shapash HTML report."""
    custom_cases = {("titanic", "classification"), ("house_prices", "regression")}
    if report_mode == "custom" and (dataset, case) not in custom_cases:
        raise ValueError("Custom layouts are available for Titanic classification and house prices regression.")

    X, y, encoder, labels, target_names, test_size = _load_task_data(dataset, case)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=79,
        stratify=y if case != "regression" else None,
    )
    X_train_encoded = encoder.fit_transform(X_train)
    X_test_encoded = encoder.transform(X_test)

    if case == "regression":
        model = RandomForestRegressor(n_estimators=200, random_state=79)
    else:
        model = RandomForestClassifier(n_estimators=200, random_state=79)
    model.fit(X_train_encoded, y_train)
    y_pred = pd.DataFrame({"pred": model.predict(X_test_encoded)}, index=X_test.index)

    title_story = f"{dataset.replace('_', ' ').title()} {case.title()} Report"
    explainer = SmartExplainer(
        model=model,
        preprocessing=encoder,
        features_dict=labels,
        label_dict=target_names,
        title_story=title_story,
    )
    explainer.compile(x=X_test_encoded, y_pred=y_pred, y_target=y_test)

    project_root = Path(__file__).resolve().parents[1]
    config_dir = project_root / "tutorial" / "generate_report" / "config"
    custom_configs = {
        ("titanic", "classification"): config_dir / "custom_report_classification_titanic.yml",
        ("house_prices", "regression"): config_dir / "custom_report_regression_house_prices.yml",
    }
    suffix = "_custom" if report_mode == "custom" else ""
    if output_file is None:
        report_path = Path(__file__).resolve().parent / "reports" / f"{dataset}_{case}{suffix}.html"
    else:
        report_path = Path(output_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_args = {"output_file": str(report_path)}
    block_instance = _build_custom_blocks(case, dataset, explainer, X_train_encoded, y_train, y_test)
    if block_instance is None:
        report_args.update(x_train=X_train_encoded, y_train=y_train, y_test=y_test)
    else:
        report_args["block_instance"] = block_instance
    if report_mode == "custom":
        report_args["yaml_path"] = str(custom_configs[(dataset, case)])

    explainer.generate_report(**report_args)
    return report_path


def main(argv: list[str] | None = None) -> None:
    """Parse CLI options and generate the selected report."""
    parser = argparse.ArgumentParser(description="Generate Shapash reports for bundled example datasets.")
    parser.add_argument(
        "--dataset",
        choices=["titanic", "house_prices", "telco_customer_churn"],
        default="titanic",
    )
    parser.add_argument("--case", choices=["classification", "regression", "multiclass"], default="classification")
    parser.add_argument("--report-mode", choices=["default", "custom"], default="default")
    parser.add_argument("--output", type=Path, help="HTML output path")
    if not (argv if argv is not None else sys.argv[1:]):
        parser.print_help()
        return

    args = parser.parse_args(argv)
    try:
        output_path = generate_report(args.dataset, args.case, args.report_mode, args.output)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Report generated: {output_path}")


if __name__ == "__main__":
    main()
