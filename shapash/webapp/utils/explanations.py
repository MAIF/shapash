# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 15:12:08 2022

@author: Florine
"""

class Explanations:

    def __init__(self):
        self.detail_feature = """The **contributions displayed** are those of a
                                 single sample. It is possible to select the
                                 sample by the index or by clicking on a point
                                 of the contribution_plot or the table. The
                                 hidden contributions, are the sum of the
                                 remaining contributions, "less high than
                                 the others", to have less hidden
                                 contributions, you have to increase the
                                 features to display.
                             """
        self.feature_selector = """
                                How does a **feature influence** the prediction?
                                colour of the point represents the value of
                                the prediction. The position on the x-axis
                                represents the modality the y-axis represents
                                the contribution of the variable. If the point
                                is at the bottom, the contribution of the
                                variable negatively impacts the prediction.
                                If the point is up, the `contribution`
                                of the variable positively impacts the
                                prediction.
                                """
        self.prediction_picking = """
                                  Text **have** to be written.
                                  """
        self.feature_importance = """
                                    **feature importance** is computed from the
                                    sum of the contributions (by default) on
                                    the whole dataset you can click on each
                                    feature to update the contribution plot
                                    below. If you have grouped the variables
                                    with the "features_groups" parameter,
                                    you can have the group details by clicking
                                    on the grouped variable.
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
