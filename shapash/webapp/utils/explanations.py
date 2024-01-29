"""
Created on Mon Oct  3 15:12:08 2022

@author: Florine
"""


class Explanations:
    """
    Contains the explanations of all "?" buttons in the app.
    ----------
    explanations: object
        SmartApp instance to point to.
    """

    def __init__(self):
        self.detail_feature = """
                        **Local interpretability:** the understanding of the decision
                        process for a single sample
                        is provided by displaying each most important feature's contributions
                        for this specific sample.
                        It is possible to select the sample by indicating its index or by clicking
                        on a point displayed
                        on the contribution_plot or clicking on the table. \n

                        The *hidden contributions* bar represents the
                        sum of the contributions of the remaining features. Each of these remaining
                        features have lower contribution than the one displayed. \n
                        To decrease the *hidden contributions* bar,
                        just increase the number of features to display, and mechanically,
                        the sum of the remaining contribution will decrease.
                        """
        self.feature_selector = """
                                **How does a feature affect the prediction?**
                                The point's colour indicates the value of the prediction. \n
                                The position on x-axis and y-axis respectively represents the
                                modality and the contribution of the variable. \n
                                If the point is located in the inferior plot area,
                                the contribution of the feature negatively impacts the prediction.\n
                                If the point in the superior plot area, the contribution of the
                                feature positively impacts the prediction. \n
                                Positive impact means that the variable favors a higher probability
                                returned by the model or
                                increases the predicted value (in case of regression problem).
                                """
        self.prediction_picking = """
                                **What are the samples with correct or wrong predictions?**
                                This graph enables to visualize and select samples to understand
                                their explainability **dynamically** with the other graphs.\n
                                If a single sample is selected, the local explicability graph is
                                updated.\n
                                If a sub-population is selected using the "box" or "lasso" tool,
                                the global explainability is updated.\n
                                It allows the comparison of the global feature contribution
                                displaying this sub-population (grey) vs the full dataset (yellow).\n
                                To reset the select, double click on the figure.
                                """
        self.feature_importance = """
                                **Global feature importance** is by default the sum of
                                *individual contributions*,
                                computed on the complete dataset.
                                You can click on each feature to update
                                the detailed contribution plot below. \n
                                In case of grouped the variables (based on the "features_groups"
                                parameter),
                                the group detail is available by clicking on the grouped variable
                                (displayed in dark orange).
                                """
        self.filter = """
                      **To create a new filter**, you must click on the
                      *Add Filter* button. You can create as many filters
                      as you want.

                      **To apply your filters**, you have to click on the
                      *Apply Filter* button.

                      If you want **to delete a filter line**, you can click
                      on the *del* button and then click on the *Apply Filter*
                      button again. **Please note** that the filters will not
                      be updated if you do not click on the *Apply Filter* button.

                      Finally, if you want **to delete all the filters**,
                      you can click on the *Reset all existing filters* button.
                      """
