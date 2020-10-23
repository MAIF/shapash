"""
Smart predictor module
"""

from shapash.utils.check import check_model,check_preprocessing
from shapash.utils.check import check_label_dict,check_mask_params



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
        Dictionary mapping integer column number (in the same order of the trained dataset) to technical feature names.
    model: model object
        model used to check the different values of target estimate predict_proba
    preprocessing: category_encoders, ColumnTransformer, list or dict (optional)
        The processing apply to the original data.
    postprocessing: dict (optional)
        Dictionnary of postprocessing modifications to apply in x_pred dataframe.
    _case: string
        String that informs if the model used is for classification or regression problem.
    _classes: list, None
        List of labels if the model used is for classification problem, None otherwise.
    mask_params: dict (optional)
        Dictionnary allowing the user to define a apply a filter to summarize the local explainability.

    How to declare a new SmartPredictor object?

    Example
    --------
    >>> predictor = SmartPredictor(features_dict,
                                    model,
                                    columns_dict,
                                    label_dict,
                                    preprocessing,
                                    postprocessing
                                    )

    or predictor = xpl.to_smartpredictor()

    xpl, explainer: object
        SmartExplainer instance to point to.
    """

    def __init__(self, features_dict, model,
                 columns_dict, label_dict=None,
                 preprocessing=None,postprocessing=None,
                 mask_params = {"features_to_hide":None,
                                "threshold":None,
                                "positive":None,
                                "max_contrib":None
                                }
                 ):

        if isinstance(features_dict, dict) == False:
            raise ValueError(
                """
                features_dict must be a dict  
                """
            )

        if label_dict is not None and isinstance(label_dict, dict) == False:
            raise ValueError(
                """
                label_dict must be a dict  
                """
            )

        self.model = model
        self._case,self._classes = self.check_model()
        self.preprocessing = preprocessing
        self.check_preprocessing()
        self.features_dict = features_dict
        self.label_dict = label_dict
        self.check_label_dict()

        if postprocessing is not None and isinstance(postprocessing, dict) == False:
            raise ValueError(
                """
                label_dict must be a dict  
                """
            )

        self.postprocessing = postprocessing

        if columns_dict is not None and isinstance(columns_dict, dict) == False:
            raise ValueError(
                """
                label_dict must be a dict  
                """
            )
        self.columns_dict = columns_dict

        self.mask_params= mask_params
        self.check_mask_params()


    def check_model(self):
        """
        Check if model has a predict_proba method is a one column dataframe of integer or float
        and if y_pred index matches x_pred index

        Returns
        -------
        string:
            'regression' or 'classification' according to the attributes of the model
        """
        _case,_classes = check_model(self.model)
        return _case,_classes

    def check_preprocessing(self):
        """
        Check that all transformation of the preprocessing are supported.
        """
        return check_preprocessing(self.preprocessing)

    def check_label_dict(self):
        """
        Check if label_dict and model _classes match
        """
        if self._case != "regression":
            return check_label_dict(self.label_dict, self._case,self._classes)

    def check_mask_params(self):
        """
        Check if mask_params given respect the expected format.
        """
        return check_mask_params(self.mask_params)

