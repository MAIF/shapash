"""
Smart predictor module
"""



class SmartPredictor :
    """
    The SmartPredictor class is an object which inherits from the
    SmartExplainer class.

    Thanks to an explainer, It allows the Data Scientists to perform operations
    to make results more understandable on new datasets with the same structure as
    the one used to build the explainer.

    As a lighter layer of the SmartExplainer, SmartPredictor is designed to perform
    the essential operations to make new results understandable :
    linking to preprocessing and postprocessing already used, models contributions,
    predictions, local epxlainability.

    This class allows the user to automatically summarize the results of his model
    on new datasets (prediction, preprocessing and postprocessing linking,
    explainability).

    The SmartPredictor has several methods described below.

    The SmartPredictor Attributes :

    features_dict: dict
        Dictionary mapping technical feature names to domain names.
    label_dict: dict
        Dictionary mapping integer labels to domain names (classification - target values).
    columns_dict: dict
        Dictionary mapping integer column number to technical feature names.
    model: model object
        model used to check the different values of target estimate predict proba
    preprocessing : category_encoders, ColumnTransformer, list or dict
        The processing apply to the original data.
    postprocessing : dict
        Dictionnary of postprocessing modifications to apply in x_pred dataframe.
    _case : string
        String that informs if the model used is for classification or regression problem.
    _classes : list, None
        List of labels if the model used is for classification problem, None otherwise.

    How to declare a new SmartPredictor object?

    Example
    --------
    >>> predictor = SmartPredictor(explainer=xpl)

    or predictor = xpl.to_smartpredictor()

    xpl, explainer: object
        SmartExplainer instance to point to.
    """

    def __init__(self,explainer):
        self.model = self.check_attributes(explainer,"model")
        self.features_dict = self.check_attributes(explainer,"features_dict")
        self.label_dict = self.check_attributes(explainer,"label_dict")
        self._case = self.check_attributes(explainer, "_case")
        self._classes = self.check_attributes(explainer, "_classes")
        self.columns_dict = self.check_attributes(explainer,"columns_dict")
        self.preprocessing = self.check_attributes(explainer,"preprocessing")
        self.postprocessing = self.check_attributes(explainer,"postprocessing")

        if hasattr(explainer,"mask_params"):
            self.mask_params = explainer.mask_params
        else:
            self.mask_params = self.mask_params = {
                                    'features_to_hide': None,
                                    'threshold': None,
                                    'positive': None,
                                    'max_contrib': None
            }

    def check_attributes(self,explainer,attribute):
        """
        Check that explainer has the attribute precised

        Parameters
        ----------
        explainer: object
            SmartExplainer instance to point to.
        attribute : string
            the label of the attribute to test
        """
        if hasattr(explainer, attribute):
            return explainer.__dict__[attribute]
        else:
            raise ValueError(
                """
                attribute {0} isn't an attribute of the explainer precised.
                """.format(attribute))
