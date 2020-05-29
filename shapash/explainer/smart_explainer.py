"""
Smart explainer module
"""
import pickle
import logging
import numpy as np
import pandas as pd
from pandas.core.common import flatten
# TODO: Remove the next 4 lines
# these lines allow you to run locally the code and import shapash content
#import os,sys,inspect
#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(currentdir)
#sys.path.insert(0,parentdir)
from shapash.webapp.smart_app import SmartApp
from shapash.utils.io import save_pickle
from shapash.utils.io import load_pickle
from shapash.utils.transform import inverse_transform
from shapash.utils.utils import get_host_name
from shapash.utils.threading import CustomThread
from shapash.utils.shap_backend import shap_contributions
from .smart_state import SmartState
from .multi_decorator import MultiDecorator
from .smart_plotter import SmartPlotter


logging.basicConfig(level=logging.INFO)

class SmartExplainer:
    """
    The SmartExplainer class is the main object of the Shapash library.
    It allows the Data Scientists to perform many operations to make the
    results more understandable
    linking encoders, models, predictions, label dict and datasets:
    SmartExplainer users have several methods which are described below

    The SmartExplainer Attributes :

    data: dict
        Data dictionary has 3 entries. Each key returns a pd.DataFrame (regression) or a list of pd.DataFrame
        (classification - The length of the lists is equivalent to the number of labels.
        All pd.DataFrame have she same shape (n_samples, n_features).
        For the regression case, data that should be regarded as a single array
        of size (n_samples, n_features, 3).

        data['contrib_sorted']: pandas.DataFrame (regression) or list of pandas.DataFrame (classification)
            Contains local contributions of the prediction set, with common line index.
            Columns are 'contrib_1', 'contrib_2', ... and contains the top contributions
            for each line from left to right. In multi-class problems, this is a list of
            contributions, one for each class.
        data['var_dict']: pandas.DataFrame (regression) or list of pandas.DataFrame (classification)
            Must contains only ints. It gives for each line the list of most import features
            regarding the local decomposition. In order to save space, columns are denoted by
            integers, the conversion being done with the columns_dict member. In multi-class
            problems, this is a list of dataframes, one for each class.
        data['x_sorted']: pandas.DataFrame (regression) or list of pandas.DataFrame (classification)
            It gives for each line the list of most important features values regarding the local
            decomposition. These values can only be understood with respect to data['var_dict']

    x_init: pandas.DataFrame
        preprocessed dataset used by the model to perform the prediction.
    x_pred: pandas.DataFrame
        x_init dataset with inverse transformation.
    y_pred: pandas.DataFrame
        User-specified prediction values.
    contributions: pandas.DataFrame (regression) or list (classification)
        local contributions aggregated if the preprocessing part requires it (e.g. one-hot encoding).
    features_dict: dict
        Dictionary mapping technical feature names to domain names.
    inv_features_dict: dict
        Inverse features_dict mapping.
    label_dict: dict
        Dictionary mapping integer labels to domain names (classification - target values).
    inv_label_dict: dict
        Inverse label_dict mapping.
    columns_dict: dict
        Dictionary mapping integer column number to technical feature names.
    inv_columns_dict: dict
        Inverse columns_dict mapping.
    plot: object
        Helper object containing all plotting functions (Bridge pattern).
    model: model object
        model used to check the different values of target estimate predict proba
    features_desc: dict
        Dictionary that references the numbers of feature values ​​in the x_pred
    features_imp: pandas.Series (regression) or list (classification)
        Features importance values

    How to declare a new SmartExplainer object?

    Example
    --------
    >>> xpl = SmartExplainer(features_dict=featd,label_dict=labeld)

    features_dict & label_dict are both optional.
    features_dict maps technical feature names to domain names.
    label_dict specify the labels of target (classification).
    """

    def __init__(self, features_dict={}, label_dict=None):
        if isinstance(features_dict,dict) == False:
            raise ValueError(
                """
                features_dict must be a dict  
                """
            )
        if label_dict is not None and isinstance(label_dict,dict) == False:
            raise ValueError(
                """
                label_dict must be a dict  
                """
            )
        self.features_dict = features_dict
        self.label_dict = label_dict
        self.plot = SmartPlotter(self)

    def compile(self, x, model, contributions=None, y_pred=None, preprocessing=None):
        """
        The compile method is the first step to understand model and prediction. It performs the sorting
        of contributions, the reverse preprocessing steps and performs all the calculations necessary for
        a quick display of plots and efficient display of summary of explanation.
        Most of the parameters are optional but all help to display results that can be understood

        This step can last a few moments with large datasets.

        Parameters
        ----------
        x : pandas.DataFrame
            Prediction set.
            IMPORTANT: this should be the raw prediction set, whose values are seen by the end user.
            x is a preprocessed dataset: Shapash can apply the model to it
        model : model object
            model used to consistency check. model object can also be used by some method to compute
            predict and predict_proba values
        contributions : pandas.DataFrame, np.ndarray or list
            single or multiple contributions (multi-class) to handle.
            if pandas.Dataframe, the index and columns should be share with the prediction set.
            if np.ndarray, index and columns will be generated according to x dataset
        y_pred : pandas.Series, optional (default: None)
            Prediction values (1 column only).
            The index must be identical to the index of x_pred.
            This is an interesting parameter for more explicit outputs. Shapash lets users define their own predict,
            as they may wish to set their own threshold (classification)
        preprocessing : category_encoders, ColumnTransformer, list, dict, optional (default: None)
            --> Differents types of preprocessing are available:

            - A single category_encoders (OrdinalEncoder/OnehotEncoder/BaseNEncoder/BinaryEncoder/TargetEncoder)
            - A single ColumnTransformer with scikit-learn encoding or category_encoders transformers
            - A list with multiple category_encoders with optional (dict, list of dict)
            - A list with a single ColumnTransformer with optional (dict, list of dict)
            - A dict
            - A list of dict

        Example
        --------
        >>> xpl.compile(x=xtest_df,model=my_model)

        """
        self.x_init = x
        self.x_pred = inverse_transform(self.x_init, preprocessing)
        self.model = model
        self._case, self._classes = self.check_model()
        self.check_label_dict()
        if self.label_dict:
            self.inv_label_dict = {v: k for k, v in self.label_dict.items()}
        if contributions is None:
            contributions = shap_contributions(model, self.x_init)
        adapt_contrib = self.adapt_contributions(contributions)
        self.state = self.choose_state(adapt_contrib)
        self.contributions = self.apply_preprocessing(self.validate_contributions(adapt_contrib), preprocessing)
        self.check_contributions()
        self.y_pred = y_pred
        self.check_y_pred()
        self.columns_dict = {i: col for i, col in enumerate(self.x_pred.columns)}
        self.inv_columns_dict = {v: k for k, v in self.columns_dict.items()}
        self.check_features_dict()
        self.inv_features_dict = {v: k for k, v in self.features_dict.items()}
        self.data = self.state.assign_contributions(
            self.state.rank_contributions(
                self.contributions,
                self.x_pred
            )
        )
        self.features_imp = None
        self.features_desc = self.check_features_desc()
        self.smartapp = SmartApp(self)

    def add(self, y_pred=None, label_dict=None, features_dict=None):
        """
        Add method allows the user to add a label_dict, features_dict
        or y_pred without compiling again (and it can last a few moments).
        y_pred can be used in the plot to color scatter.
        y_pred is needed in the to_pandas method.
        label_dict and features_dict displays allow to display clearer results.

        Parameters
        ----------
        y_pred : pandas.Series, optional (default: None)
            Prediction values (1 column only).
            The index must be identical to the index of x_pred.
        label_dict: dict, optional (default: None)
            Dictionary mapping integer labels to domain names.
        features_dict: dict, optional (default: None)
            Dictionary mapping technical feature names to domain names.
        """
        if y_pred is not None:
            self.y_pred = y_pred
            self.check_y_pred()
        if label_dict is not None:
            if isinstance(label_dict, dict) == False:
                raise ValueError(
                    """
                    label_dict must be a dict  
                    """
                )
            self.label_dict = label_dict
            self.check_label_dict()
            self.inv_label_dict = {v: k for k, v in self.label_dict.items()}
        if features_dict is not None:
            if isinstance(features_dict, dict) == False:
                raise ValueError(
                    """
                    features_dict must be a dict  
                    """
                )
            self.features_dict = features_dict
            self.check_features_dict()
            self.inv_features_dict = {v: k for k, v in self.features_dict.items()}
        self.smartapp = SmartApp(self)

    def choose_state(self, contributions):
        """
        Select implementation of the smart explainer. Typically check if it is a
        multi-class problem, in which case the implementation should be adapted
        to lists of contributions.

        Parameters
        ----------
        contributions : object
            Local contributions. Could also be a list of local contributions.

        Returns
        -------
        object
            SmartState or SmartMultiState, depending on the nature of the input.
        """
        if isinstance(contributions, list):
            return MultiDecorator(SmartState())
        else:
            return SmartState()

    def adapt_contributions(self, contributions):
        """
        If _case is "classification" and contributions a np.array or pd.DataFrame
        this function transform contributions matrix in a list of 2 contributions
        matrices: Opposite contributions and contributions matrices.

        Parameters
        ----------
        contributions : pandas.DataFrame, np.ndarray or list

        Returns
        -------
            pandas.DataFrame, np.ndarray or list
            contributions object modified
        """
        if isinstance(contributions, (np.ndarray, pd.DataFrame)) and self._case == 'classification' :
            return [contributions * -1, contributions]
        else:
            return contributions

    def validate_contributions(self, contributions):
        """
        Check len of list if _case is "classification"
        Check contributions object type if _case is "regression"
        Check type of contributions and transform into (list of) pd.Dataframe if necessary


        Parameters
        ----------
        contributions : pandas.DataFrame, np.ndarray or list

        Returns
        -------
            pandas.DataFrame or list
        """
        if self._case == "regression" and isinstance(contributions, (np.ndarray, pd.DataFrame)) == False:
            raise ValueError(
                """
                Type of contributions parameter specified is not compatible with 
                regression model.
                Please check model and contributions parameters.  
                """
            )
        elif self._case == "classification":
            if isinstance(contributions, list):
                if len(contributions) != len(self._classes):
                    raise ValueError(
                        """
                        Length of list of contributions parameter is not equal
                        to the number of classes in the target.
                        Please check model and contributions parameters.
                        """
                    )
            else:
                raise ValueError(
                    """
                    Type of contributions parameter specified is not compatible with 
                    classification model.
                    Please check model and contributions parameters.
                    """
                )

        return self.state.validate_contributions(contributions, self.x_init)

    def apply_preprocessing(self, contributions, preprocessing=None):
        """
        Reconstruct contributions for original features, taken into account a preprocessing.

        Parameters
        ----------
        contributions : object
            Local contributions, or list of local contributions.
        preprocessing : object
            Encoder taken from scikit-learn or category_encoders

        Returns
        -------
        object
            Reconstructed local contributions in the original space. Can be a list.
        """
        if preprocessing:
            return self.state.inverse_transform_contributions(
                contributions,
                preprocessing
            )
        else:
            return contributions

    def check_y_pred(self):
        """
        Check if y_pred is a one column dataframe of integer or float
        and if y_pred index matches x_pred index
        """
        if self.y_pred is not None:
            if not isinstance(self.y_pred, (pd.DataFrame, pd.Series)):
                raise ValueError("y_pred must be a one column pd.Dataframe or pd.Series.")
            if not self.y_pred.index.equals(self.x_pred.index):
                raise ValueError("x_pred and y_pred should have the same index.")
            if isinstance(self.y_pred, pd.DataFrame):
                if self.y_pred.shape[1] > 1:
                    raise ValueError("y_pred must be a one column pd.Dataframe or pd.Series.")
                if not (self.y_pred.dtypes[0] in [np.float, np.int]):
                    raise ValueError("y_pred must contain int or float only")
            if isinstance(self.y_pred, pd.Series):
                if not (self.y_pred.dtype in [np.float, np.int]):
                    raise ValueError("y_pred must contain int or float only")
                self.y_pred = self.y_pred.to_frame()

    def check_model(self):
        """
        Check if model has a predict_proba method is a one column dataframe of integer or float
        and if y_pred index matches x_pred index

        Returns
        -------
        string:
            'regression' or 'classification' according to the attributes of the model
        """
        _classes = None
        if hasattr(self.model, 'predict'):
            if hasattr(self.model, 'predict_proba') or\
                any(hasattr(self.model, attrib) for attrib in ['classes_', '_classes']):
                if hasattr(self.model, '_classes'): _classes = self.model._classes
                if hasattr(self.model, 'classes_'): _classes = self.model.classes_
                if isinstance(_classes,np.ndarray): _classes = _classes.tolist()
                if hasattr(self.model, 'predict_proba') and  _classes == []: _classes = [0, 1] #catboost binary
                if hasattr(self.model, 'predict_proba') and _classes is None:
                    raise ValueError(
                        "No attribute _classes, classification model not supported"
                    )
            if _classes not in (None,[]):
                return 'classification', _classes
            else:
                return 'regression', None
        else:
            raise ValueError(
                "No method predict in the specified model. Please, check model parameter"
            )

    def check_label_dict(self):
        """
        Check if label_dict and model _classes match
        """
        if self.label_dict is not None and self._case == 'classification':
            if set(self._classes) != set(list(self.label_dict.keys())):
                raise ValueError(
                     "label_dict and don't match: \n"+
                     f"label_dict keys: {str(list(self.label_dict.keys()))}\n"+
                     f"Classes model values {str(self._classes)}"
                )

    def check_features_dict(self):
        """
        Check the features_dict and add the necessary keys if all the
        input X columns are not present
        """
        for feature in (set(list(self.columns_dict.values())) - set(list(self.features_dict))):
            self.features_dict[feature] = feature

    def check_contributions(self):
        """
        Check if contributions and prediction set match in terms of shape and index.
        """
        if not self.state.check_contributions(self.contributions, self.x_pred):
            raise ValueError(
                """
                Prediction set and contributions should have exactly the same number of lines
                and number of columns. the order of the columns must be the same
                Please check x, contributions and preprocessing arguments.
                """
            )

    def check_label_name(self, label, origin=None):
        """
        Convert a string label in integer. If the label is already
        an integer nothing is done. In all other cases an error is raised.

        Parameters
        ----------
        label: int or string
            Integer (id) or string (business names)
        origin: None, 'num', 'code', 'value' (default: None)
            Kind of the label used in parameter

        Returns
        -------
        tuple
            label num, label code (class of the mode), label value
        """
        if origin is None:
            if label in self._classes:
                origin = 'code'
            elif self.label_dict is not None and label in self.label_dict.values():
                origin = 'value'
            elif isinstance(label, int) and label in range(-1, len(self._classes)):
                origin = 'num'

        try:
            if origin == 'num':
                label_num = label
                label_code = self._classes[label]
                label_value = self.label_dict[label_code] if self.label_dict else label_code
            elif origin == 'code':
                label_code = label
                label_num = self._classes.index(label)
                label_value = self.label_dict[label_code] if self.label_dict else label_code
            elif origin == 'value':
                label_code = self.inv_label_dict[label]
                label_num = self._classes.index(label_code)
                label_value = label
            else:
                raise ValueError

        except ValueError:
            raise Exception({"message": "Origin must be 'num', 'code' or 'value'."})

        except Exception:
            raise Exception({"message": f"Label ({label}) not found for origin ({origin})"})

        return label_num, label_code, label_value

    def check_features_name(self, features):
        """
        Convert a list of feature names (string) or features ids into features ids.
        Features names can be part of columns_dict or features_dict.

        Parameters
        ----------
        features : List
            List of ints (columns ids) or of strings (business names)

        Returns
        -------
        list of ints
            Columns ids compatible with var_dict
        """

        if all(isinstance(f, int) for f in features):
            features_ids = features

        elif all(isinstance(f, str) for f in features):
            if self.features_dict and all(f in self.features_dict.values() for f in features):
                columns_list = [self.inv_features_dict[f] for f in features]
                features_ids = [self.inv_columns_dict[c] for c in columns_list]
            elif self.inv_columns_dict and all(f in self.columns_dict.values() for f in features):
                features_ids = [self.inv_columns_dict[f] for f in features]
            else:
                raise ValueError(
                    'All features must came from the same dict of features (technical names or domain names).'
                )

        else:
            raise ValueError(
                """
                features must be a list of ints (representing ids of columns)
                or a list of string from technical features names or from domain names.
                """
            )
        return features_ids

    def check_features_desc(self):
        """
        Check x_pred dataframe, compute value counts of each feature
        used in plot part

        Returns
        -------
        dict
            Number of unique values in x_pred
        """
        return dict(self.x_pred.nunique())

    def filter(
            self,
            features_to_hide=None,
            threshold=None,
            positive=None,
            max_contrib=None
    ):

        """
        The filter method is an important method which allows to summarize the local explainability
        by using the user defined parameters which correspond to its use case.
        Filter method is used with the local_plot method of Smarplotter to see the concrete result of this summary
        with a local contribution barchart

        Please, watch the local_plot tutorial to see how these two methods are combined with a concrete example

        Parameters
        ----------
        features_to_hide : list, optional (default: None)
            List of strings, containing features to hide.
        threshold : float, optional (default: None)
            Absolute threshold below which any contribution is hidden.
        positive: bool, optional (default: None)
            If True, hide negative values. False, hide positive values
            If None, hide nothing.
        max_contrib : int, optional (default: None)
            Maximum number of contributions to show.
        """
        mask = [self.state.init_mask(self.data['contrib_sorted'], True)]
        if features_to_hide:
            mask.append(
                self.state.hide_contributions(
                    self.data['var_dict'],
                    features_list=self.check_features_name(features_to_hide)
                )
            )
        if threshold:
            mask.append(
                self.state.cap_contributions(
                    self.data['contrib_sorted'],
                    threshold=threshold
                )
            )
        if positive is not None:
            mask.append(
                self.state.sign_contributions(
                    self.data['contrib_sorted'],
                    positive=positive
                )
            )
        self.mask = self.state.combine_masks(mask)
        if max_contrib:
            self.mask = self.state.cutoff_contributions(self.mask, max_contrib=max_contrib)
        self.masked_contributions = self.state.compute_masked_contributions(
            self.data['contrib_sorted'],
            self.mask
        )
        self.mask_params = {
            'features_to_hide' : features_to_hide,
            'threshold' : threshold,
            'positive' : positive,
            'max_contrib' : max_contrib
        }

    def save(self, path, protocol=pickle.HIGHEST_PROTOCOL):
        """
        Save method allows user to save SmartExplainer object on disk
        using a pickle file.
        Save method can be useful: you don't have to recompile to display
        results later

        Parameters
        ----------
        path : str
            File path to store the pickle file
        protocol : int
            Int which indicates which protocol should be used by the pickler,
            default HIGHEST_PROTOCOL

        Example
        --------
        >>> xpl.save('path_to_pkl/xpl.pkl')
        """
        dict_to_save = {}
        for att in self.__dict__.keys():
            if isinstance(getattr(self, att), (list, dict, pd.DataFrame, pd.Series, type(None))) or att == "model":
                dict_to_save.update({att: getattr(self, att)})
        save_pickle(dict_to_save, path)

    def load(self, path):
        """
        Load method allows Shapash user to use pikled SmartExplainer.
        To use this method you must first declare your SmartExplainer object
        Watch the following example

        Parameters
        ----------
        path : str
            File path of the pickle file.

        Example
        --------
        >>> xpl = SmartExplainer()
        >>> xpl.load('path_to_pkl/xpl.pkl')
        """
        dict_to_load = load_pickle(path)
        if isinstance(dict_to_load, dict):
            for elem in dict_to_load.keys():
                setattr(self, elem, dict_to_load[elem])
            self._case, self._classes = self.check_model()
            self.state = self.choose_state(self.contributions)
        else:
            raise ValueError(
                "pickle file must contain dictionary"
            )

    def to_pandas(
            self,
            features_to_hide=None,
            threshold=None,
            positive=None,
            max_contrib=None,
            proba=False
    ):
        """
        The to_pandas method allows to export the summary of local explainability.
        This method proposes a set of parameters to summarize the explainability of each point.
        If the user does not specify any, the to_pandas method uses the parameter specified during
        the last execution of the filter method.

        In classification case, The method to_pandas summarizes the explicability which corresponds
        to the predicted values specified by the user (with compile or add method).
        the proba parameter displays the corresponding predict proba value for each point
        In classification case, There are 2 ways to use this to pandas method.
        - Provide a real prediction set to explain
        - Focus on a constant target value and look at the proba and explainability corresponding to each point.
        (in that case, specify a constant pd.Series with add or compile method)

        Examples are presented in the tutorial local_plot (please check tutorial part of this doc)

        Parameters
        ----------
        features_to_hide : list, optional (default: None)
            List of strings, containing features to hide.
        threshold : float, optional (default: None)
            Absolute threshold below which any contribution is hidden.
        positive: bool, optional (default: None)
            If True, hide negative values. Hide positive values otherwise. If None, hide nothing.
        max_contrib : int, optional (default: 5)
            Number of contributions to show in the pandas df
        proba : bool, optional (default: False)
            adding proba in output df

        Returns
        -------
        pandas.DataFrame
            - selected explanation of each row for classification case


        Examples
        --------
        >>> summary_df = xpl.to_pandas(max_contrib=2,proba=True)
        >>> summary_df
        	pred	proba	    feature_1	value_1	    contribution_1	feature_2	value_2	    contribution_2
        0	0	    0.756416	Sex	        1.0	        0.322308	    Pclass	    3.0	        0.155069
        1	3	    0.628911	Sex	        2.0	        0.585475	    Pclass	    1.0	        0.370504
        2	0	    0.543308	Sex	        2.0	        -0.486667	    Pclass	    3.0	        0.255072
        """

        # Classification: y_pred is needed
        if self.y_pred is None:
            raise ValueError(
                "You have to specify y_pred argument. Please use add() or compile() method"
            )

        # Apply filter method if necessary
        if all(var is None for var in [features_to_hide, threshold, positive, max_contrib]) \
            and hasattr(self,'mask_params'):
            print('to_pandas params: ' + str(self.mask_params))
        else:
            self.filter(features_to_hide=features_to_hide,
                        threshold=threshold,
                        positive=positive,
                        max_contrib=max_contrib)

        # Summarize information
        self.data['summary'] = self.state.summarize(
            self.data['contrib_sorted'],
            self.data['var_dict'],
            self.data['x_sorted'],
            self.mask,
            self.columns_dict,
            self.features_dict
        )

        # Matching with y_pred
        y_pred = self.y_pred
        if self._case == "classification":
            complete_sum = [list(x) for x in list(zip(*[df.values.tolist() for df in self.data['summary']]))]
            indexclas = [self._classes.index(x) for x in list(flatten(self.y_pred.values))]
            summary = pd.DataFrame([summar[ind]
                                    for ind, summar in zip(indexclas, complete_sum)],
                                   columns=self.data['summary'][0].columns,
                                   index=self.data['summary'][0].index,
                                   dtype=object)
            if self.label_dict is not None:
                y_pred = y_pred.applymap(lambda x: self.label_dict[x])
            if proba:
                if hasattr(self.model,'predict_proba'):
                    probamatrix = self.model.predict_proba(self.x_init)
                    y_proba = pd.DataFrame([proba[ind]
                                            for ind, proba in zip(indexclas, probamatrix)],
                                           columns=['proba'],
                                           index=y_pred.index)
                    y_pred = pd.concat([y_pred, y_proba], axis=1)
                else:
                    print("model has no predict_proba method")
        else:
            summary = self.data['summary']

        return pd.concat([y_pred, summary], axis=1)

    def compute_features_import(self, force=False):
        """
        Compute a relative features importance, sum of absolute values
        of the contributions for each.
        Features importance compute in base 100

        Parameters
        ----------
        force: bool (default: False)
            True to force de compute if features importance is
            already calculated

        Returns
        -------
        pd.Serie (Regression)
        or list of pd.Serie (Classification: One Serie for each target modality)
            Each Serie: feature importance, One row by feature,
            index of the serie = contributions.columns
        """
        if self.features_imp is None or force:
            self.features_imp = self.state.compute_features_import(self.contributions)

    def run_app(self, port: int = None, host: str = None) -> CustomThread:
        """
        run_app method launches the interpretability web app associated with the shapash object.
        run_app method can be used directly in a Jupyter notebook
        The link to the webapp is directly mentioned in the Jupyter output
        Use object.kill() method to kill the current instance

        Examples are presented in the web_app tutorial (please check tutorial part of this doc)

        Parameters
        ----------
        port: int (default: None)
            The port is by default on 8050. You can specify a custom port
            for your webapp.
        host: str (default: None)
            The default host is '0.0.0.0'. You can specify a custom
            ip address for your app

        Returns
        -------
        CustomThread
            Return the thread instance of your server.

        Example
        --------
        >>> app = xpl.run_app()
        >>> app.kill()
        """
        if hasattr(self, '_case'):
            if host is None:
                host = "0.0.0.0"
            if port is None:
                port = 8050
            host_name = get_host_name()
            server_instance = CustomThread(target=lambda: self.smartapp.app.run_server(debug=False, host=host, port=port))
            if host_name is None:
                host_name = host
            elif host != "0.0.0.0":
                host_name = host
            server_instance.start()
            logging.info(f"Your Shapash application run on http://{host_name}:{port}/")
            logging.info(f"Use the method .kill() to down your app.")
            return server_instance

        else:
            raise ValueError("Explainer must be compiled before running app.")

