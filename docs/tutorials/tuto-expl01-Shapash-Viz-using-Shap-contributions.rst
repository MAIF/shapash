Compute Contributions with Shap - Summarize It With Shapash
===========================================================

Shapash uses Shap backend to compute the Shapley contributions in order
to satisfy the most pressed users who wish to display results with
little line of code.

But we recommend that you refer to the excellent `Shap
library <https://github.com/slundberg/shap>`__.

This tutorial shows how to use precalculated contributions with Shap in
Shapash

Contents: - Build a Binary Classifier - Use Shap KernelExplainer -
Compile Shapash SmartExplainer - Display local_plot - to_pandas export

Data from Kaggle `Titanic <https://www.kaggle.com/c/titanic>`__

.. code:: ipython3

    import numpy as np
    import pandas as pd
    from category_encoders import OrdinalEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import shap

.. code:: ipython3

    from shapash.data.data_loader import data_loading

.. code:: ipython3

    titan_df, titan_dict = data_loading('titanic')
    del titan_df['Name']

.. code:: ipython3

    titan_df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Survived</th>
          <th>Pclass</th>
          <th>Sex</th>
          <th>Age</th>
          <th>SibSp</th>
          <th>Parch</th>
          <th>Fare</th>
          <th>Embarked</th>
          <th>Title</th>
        </tr>
        <tr>
          <th>PassengerId</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>1</th>
          <td>0</td>
          <td>Third class</td>
          <td>male</td>
          <td>22.0</td>
          <td>1</td>
          <td>0</td>
          <td>7.25</td>
          <td>Southampton</td>
          <td>Mr</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1</td>
          <td>First class</td>
          <td>female</td>
          <td>38.0</td>
          <td>1</td>
          <td>0</td>
          <td>71.28</td>
          <td>Cherbourg</td>
          <td>Mrs</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1</td>
          <td>Third class</td>
          <td>female</td>
          <td>26.0</td>
          <td>0</td>
          <td>0</td>
          <td>7.92</td>
          <td>Southampton</td>
          <td>Miss</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1</td>
          <td>First class</td>
          <td>female</td>
          <td>35.0</td>
          <td>1</td>
          <td>0</td>
          <td>53.10</td>
          <td>Southampton</td>
          <td>Mrs</td>
        </tr>
        <tr>
          <th>5</th>
          <td>0</td>
          <td>Third class</td>
          <td>male</td>
          <td>35.0</td>
          <td>0</td>
          <td>0</td>
          <td>8.05</td>
          <td>Southampton</td>
          <td>Mr</td>
        </tr>
      </tbody>
    </table>
    </div>



Create Classification Model
---------------------------

.. code:: ipython3

    y = titan_df['Survived']
    X = titan_df.drop('Survived', axis=1)

.. code:: ipython3

    varcat=['Pclass','Sex','Embarked','Title']

.. code:: ipython3

    categ_encoding = OrdinalEncoder(cols=varcat, \
                                    handle_unknown='ignore', \
                                    return_df=True).fit(X)
    X = categ_encoding.transform(X)

Train Test split + Random Forest fit

.. code:: ipython3

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.75, random_state=1)
    
    rf = RandomForestClassifier(n_estimators=100,min_samples_leaf=3)
    rf.fit(Xtrain, ytrain)




.. parsed-literal::

    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=3, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)



.. code:: ipython3

    ypred=pd.DataFrame(rf.predict(Xtest),columns=['pred'],index=Xtest.index)

Compute Shapley Contributions with Shap
---------------------------------------

.. code:: ipython3

    explainer = shap.KernelExplainer(rf.predict_proba, Xtest)
    shap_contrib = explainer.shap_values(Xtest)


.. parsed-literal::

    Using 223 background data samples could cause slower run times. Consider using shap.sample(data, K) or shap.kmeans(data, K) to summarize the background as K samples.



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=223), HTML(value='')))


.. parsed-literal::

    


Use Shapash With Shapley Contributions
======================================

.. code:: ipython3

    from shapash.explainer.smart_explainer import SmartExplainer

.. code:: ipython3

    xpl = SmartExplainer(features_dict=titan_dict)

Use contributions parameter of compile method to declare Shapley contributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    xpl.compile(contributions=shap_contrib, # Shap Contributions pd.DataFrame
                y_pred=ypred,
                x=Xtest,
                model=rf,
                preprocessing=categ_encoding)

.. code:: ipython3

    xpl.plot.local_plot(index=3)



.. image:: tuto-expl01-Shapash-Viz-using-Shap-contributions_files/tuto-expl01-Shapash-Viz-using-Shap-contributions_19_0.png


.. code:: ipython3

    summary_df = xpl.to_pandas(max_contrib=3,positive=True,proba=True)
    summary_df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>pred</th>
          <th>proba</th>
          <th>feature_1</th>
          <th>value_1</th>
          <th>contribution_1</th>
          <th>feature_2</th>
          <th>value_2</th>
          <th>contribution_2</th>
          <th>feature_3</th>
          <th>value_3</th>
          <th>contribution_3</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>863</th>
          <td>1</td>
          <td>0.785470</td>
          <td>Sex</td>
          <td>female</td>
          <td>0.192354</td>
          <td>Title of passenger</td>
          <td>Mrs</td>
          <td>0.158498</td>
          <td>Ticket class</td>
          <td>First class</td>
          <td>0.092583</td>
        </tr>
        <tr>
          <th>224</th>
          <td>0</td>
          <td>0.972264</td>
          <td>Title of passenger</td>
          <td>Mr</td>
          <td>0.105669</td>
          <td>Passenger fare</td>
          <td>7.9</td>
          <td>0.074578</td>
          <td>Sex</td>
          <td>male</td>
          <td>0.0693082</td>
        </tr>
        <tr>
          <th>85</th>
          <td>1</td>
          <td>0.826169</td>
          <td>Sex</td>
          <td>female</td>
          <td>0.183571</td>
          <td>Title of passenger</td>
          <td>Miss</td>
          <td>0.177265</td>
          <td>Ticket class</td>
          <td>Second class</td>
          <td>0.083088</td>
        </tr>
        <tr>
          <th>681</th>
          <td>1</td>
          <td>0.686340</td>
          <td>Title of passenger</td>
          <td>Miss</td>
          <td>0.162849</td>
          <td>Port of embarkation</td>
          <td>Queenstown</td>
          <td>0.146734</td>
          <td>Sex</td>
          <td>female</td>
          <td>0.119779</td>
        </tr>
        <tr>
          <th>536</th>
          <td>1</td>
          <td>0.965928</td>
          <td>Title of passenger</td>
          <td>Miss</td>
          <td>0.199437</td>
          <td>Ticket class</td>
          <td>Second class</td>
          <td>0.118737</td>
          <td>Sex</td>
          <td>female</td>
          <td>0.110403</td>
        </tr>
      </tbody>
    </table>
    </div>


