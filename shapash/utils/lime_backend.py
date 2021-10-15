import numpy as np
import pandas as pd
import lime.lime_tabular

def lime_contributions(model,x_train, x_test, mode="classification", num_classes=None):
    """
        Compute local contribution with the Lime library
        Parameters
        ----------
        model: prediction function. For classifiers, this should be a
                function that takes a numpy array and outputs prediction
                probabilities. For regressors, this takes a numpy array and
                returns the predictions. For ScikitClassifiers, this is
                `classifier.predict_proba()`. For ScikitRegressors, this
                is `regressor.predict()`. The prediction function needs to work
                on multiple feature vectors (the vectors randomly perturbed
                from the data_row).
        x_train : pd.DataFrame
            Training dataset used as background.
        x_test : pd.DataFrame
            preprocessed dataset used by the model to perform the prediction.
        mode : "classification" or "regression"
        num_classes: int (default :None)
            Number of classes if len(classes)>2
        Returns
        -------
        np.array or list of np.array
        
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(x_train.values,
                                                       feature_names=x_train.columns,
                                                       mode=mode)
    lime_contrib =[]
    
    for i in x_test.index:
        
        if mode =="classification" and num_classes is None:
            
            exp = explainer.explain_instance(x_test.loc[i], model.predict_proba)
            lime_contrib.append(dict([[transform_name(b[0], x_test),b[1]] for b in exp.as_list()]))
            
        elif mode =="classification" and num_classes is not None:

            contribution=[]
            for j in range(num_classes):
                list_contrib=[]
                df_contrib = pd.DataFrame()
                for i in x_test.index:
                    exp = explainer.explain_instance(x_test.loc[i], model.predict_proba, top_labels=num_classes)
                    list_contrib.append(dict([[transform_name(b[0], x_test),b[1]] for b in exp.as_list(j)]))
                    df_contrib= pd.DataFrame(list_contrib)
                    df_contrib = df_contrib[list(x_test.columns)]
                contribution.append(df_contrib.values)
            return contribution

        else:
            
            exp = explainer.explain_instance(x_test.loc[i], model.predict)
            lime_contrib.append(dict([[transform_name(b[0], x_test),b[1]] for b in exp.as_list()]))
            
    contribution = pd.DataFrame(lime_contrib, index=x_test.index)
    contribution = contribution[list(x_test.columns)]
    
    return contribution

def transform_name(a,x_df):
    
    for colname in list(x_df.columns):
        if str(colname) in str(a):
            col_rename = colname
    return col_rename