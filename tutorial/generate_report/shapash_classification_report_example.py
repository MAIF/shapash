"""
Generate a Titanic survival classification report with the smart_report implementation.

The script supports both built-in and custom report layouts:
- default_report_classification_titanic.yml
- custom_report_classification_titanic.yml
"""

import argparse
import os
import sys

import pandas as pd
from category_encoders import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, "..")

from shapash import SmartExplainer
from shapash.data.data_loader import data_loading
from shapash.report.blocks import ReportBlockMixin, block


class CustomClassificationReportBlocks(ReportBlockMixin):
    """User-defined blocks for a Titanic classification explainability report."""

    @block
    def block_prediction_error_summary(self, title: str = "Prediction error summary"):
        """Summarize global classification errors with confusion counts and key metrics."""
        if self.y_test is None or self.y_pred is None:
            raise ValueError("prediction_error_summary block requires y_test and y_pred.")

        y_true = pd.Series(self.y_test).reset_index(drop=True)
        y_pred = pd.Series(self.y_pred).reset_index(drop=True)

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        metrics_df = pd.DataFrame(
            [
                ["Accuracy", f"{accuracy_score(y_true, y_pred):.3f}"],
                ["Precision (Survived)", f"{precision_score(y_true, y_pred, zero_division=0):.3f}"],
                ["Recall (Survived)", f"{recall_score(y_true, y_pred, zero_division=0):.3f}"],
                ["False positives", fp],
                ["False negatives", fn],
                ["True positives", tp],
                ["True negatives", tn],
            ],
            columns=["Metric", "Value"],
        )

        summary = (
            "This block highlights where the classifier makes mistakes. "
            "False positives are non-survivors predicted as survivors, while false negatives "
            "are survivors predicted as non-survivors."
        )

        return title, [summary, metrics_df]

    @block
    def block_misclassification_focus(
        self,
        title: str = "Most confident misclassifications",
        top_k: int = 10,
    ):
        """Show the wrong predictions with highest model confidence to guide error analysis."""
        explainer = self._require_explainer("misclassification_focus")
        if self.y_test is None or self.y_pred is None:
            raise ValueError("misclassification_focus block requires y_test and y_pred.")

        if explainer.proba_values is None:
            explainer.predict_proba()

        y_true = pd.Series(self.y_test, index=explainer.x_init.index, name="true")
        y_pred = pd.Series(self.y_pred, index=explainer.x_init.index, name="pred")

        if explainer.proba_values.shape[1] < 2:
            raise ValueError("misclassification_focus block requires binary class probabilities.")

        proba_survived = explainer.proba_values.iloc[:, 1].rename("proba_survived")

        analysis = pd.concat([y_true, y_pred, proba_survived, explainer.x_init], axis=1)
        wrong = analysis[analysis["true"] != analysis["pred"]].copy()

        if wrong.empty:
            return title, ["No misclassification found on the evaluated dataset."]

        wrong["wrong_confidence"] = wrong.apply(
            lambda row: row["proba_survived"] if row["pred"] == 1 else 1 - row["proba_survived"], axis=1
        )
        wrong = wrong.sort_values("wrong_confidence", ascending=False).head(top_k)

        selected_cols = ["true", "pred", "proba_survived", "wrong_confidence"]
        contextual_cols = [col for col in ["Pclass", "Sex", "Age", "Fare", "Embarked", "Title"] if col in wrong]
        display_df = wrong[selected_cols + contextual_cols].copy()
        display_df = display_df.rename(columns={"true": "True", "pred": "Pred", "proba_survived": "P(Survived)"})
        display_df["P(Survived)"] = display_df["P(Survived)"].map(lambda x: round(float(x), 3))
        display_df["wrong_confidence"] = display_df["wrong_confidence"].map(lambda x: round(float(x), 3))

        text = (
            "Rows below are the most confident wrong predictions. "
            "They are useful to inspect possible data drift, noise, or feature blind spots."
        )
        table = display_df.reset_index(drop=False)
        return title, [text, table]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Titanic classification report examples.")
    parser.add_argument(
        "--report-mode",
        choices=["default", "custom"],
        default="default",
        help="Choose report layout: default built-in blocks or custom blocks.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    titanic_df, titanic_dict = data_loading("titanic")

    y_df = titanic_df["Survived"]
    x_df = titanic_df[titanic_df.columns.difference(["Survived"])].copy()

    # Ensure non-numeric columns are treated as categorical before encoding.
    for col in x_df.columns:
        if not pd.api.types.is_numeric_dtype(x_df[col]):
            x_df[col] = x_df[col].astype(object)

    categorical_features = [
        col
        for col in x_df.columns
        if pd.api.types.is_object_dtype(x_df[col]) or pd.api.types.is_string_dtype(x_df[col])
    ]

    encoder = OrdinalEncoder(cols=categorical_features, handle_unknown="return_nan", return_df=True).fit(x_df)

    x_df = encoder.transform(x_df)

    xtrain, xtest, ytrain, ytest = train_test_split(
        x_df,
        y_df,
        train_size=0.75,
        random_state=1,
        stratify=y_df,
    )

    classifier = RandomForestClassifier(n_estimators=200, random_state=1).fit(xtrain, ytrain)

    # Keep y_pred as dataframe to match SmartExplainer report expectations.
    y_pred = pd.DataFrame(classifier.predict(xtest), columns=["pred"], index=xtest.index)

    cur_dir = os.path.dirname(os.path.abspath(__file__))

    xpl = SmartExplainer(
        model=classifier,
        preprocessing=encoder,  # Optional: compile step can use inverse_transform method
        features_dict=titanic_dict,
        label_dict={0: "Did not survive", 1: "Survived"},
    )

    # Compile once before report generation.
    xpl.compile(x=xtest, y_pred=y_pred, y_target=ytest)

    if args.report_mode == "custom":
        output_file = os.path.join(cur_dir, "output", "classification_report_custom.html")
        report_config_file = os.path.join(cur_dir, "config", "custom_report_classification_titanic.yml")
        block_instance = CustomClassificationReportBlocks(
            explainer=xpl,
            x_train=xtrain,
            y_train=ytrain,
            y_test=ytest,
        )
        xpl.generate_report(
            output_file=output_file,
            yaml_path=report_config_file,
            block_instance=block_instance,
        )
    else:
        output_file = os.path.join(cur_dir, "output", "classification_report.html")
        xpl.generate_report(
            output_file=output_file,
            x_train=xtrain,
            y_train=ytrain,
            y_test=ytest,
        )
