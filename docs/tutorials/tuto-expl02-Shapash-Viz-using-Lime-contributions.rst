Using Shapash with Lime explainer - Titanic
===========================================

You can compute your local contributions with the
`Lime <https://github.com/marcotcr/lime>`__ library and summarize it
with Shapash

This Tutorial: - Build a Binary Classifier (Random Forest) - Create
Explainer using Lime - Use Shapash to plot Local Explanation, and
summarize it

Contents: - Build a Binary Classifier - Compile Shapash SmartExplainer -
Display local_plot - to_pandas export

Data from Kaggle `Titanic <https://www.kaggle.com/c/titanic>`__

.. code:: ipython3

    import numpy as np
    import pandas as pd
    from category_encoders import OrdinalEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import lime.lime_tabular

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



Create Lime Explainer
---------------------

.. code:: ipython3

    #Training Tabular Explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(Xtrain.values, 
                                                       mode='classification',
                                                       feature_names=Xtrain.columns,
                                                       class_names=ytrain)

Apply Explainer to Test Sample And Preprocessing
------------------------------------------------

.. code:: ipython3

    # Function features_check Extract feature names from Lime Output to be used by shapash
    def features_check(s):
        for w in list(Xtest.columns):
            if f' {w} ' in f' {s} ' :
                feat = w
        return feat

.. code:: ipython3

    %%time
    # Compute local Lime Explanation for each row in Test Sample
    contrib_l=[]
    for ind in Xtest.index:
        exp = explainer.explain_instance(Xtest.loc[ind].values, rf.predict_proba, num_features=Xtest.shape[1])
        contrib_l.append(dict([[features_check(elem[0]),elem[1]] for elem in exp.as_list()]))


.. parsed-literal::

    CPU times: user 57.8 s, sys: 7.34 s, total: 1min 5s
    Wall time: 10.9 s


.. code:: ipython3

    contribution_df =pd.DataFrame(contrib_l,index=Xtest.index)
    # sorting the columns as in the original dataset
    contribution_df = contribution_df[list(Xtest.columns)]

.. code:: ipython3

    ypred=pd.DataFrame(rf.predict(Xtest),columns=['pred'],index=Xtest.index)

Use Shapash With Lime Contributions
===================================

.. code:: ipython3

    from shapash.explainer.smart_explainer import SmartExplainer

.. code:: ipython3

    xpl = SmartExplainer(features_dict=titan_dict)

Use contributions parameter of compile method to declare Lime contributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    xpl.compile(contributions=contribution_df, # Lime Contribution pd.DataFrame
                y_pred=ypred,
                x=Xtest,
                model=rf,
                preprocessing=categ_encoding)

.. code:: ipython3

    xpl.plot.local_plot(index=3)



.. image:: tuto-expl02-Shapash-Viz-using-Lime-contributions_files/tuto-expl02-Shapash-Viz-using-Lime-contributions_23_0.png


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
          <td>0.801675</td>
          <td>Sex</td>
          <td>female</td>
          <td>0.257817</td>
          <td>Title of passenger</td>
          <td>Mrs</td>
          <td>0.188714</td>
          <td>Ticket class</td>
          <td>First class</td>
          <td>0.0880992</td>
        </tr>
        <tr>
          <th>224</th>
          <td>0</td>
          <td>0.965208</td>
          <td>Sex</td>
          <td>male</td>
          <td>0.248462</td>
          <td>Title of passenger</td>
          <td>Mr</td>
          <td>0.199544</td>
          <td>Ticket class</td>
          <td>Third class</td>
          <td>0.0838383</td>
        </tr>
        <tr>
          <th>85</th>
          <td>1</td>
          <td>0.799397</td>
          <td>Sex</td>
          <td>female</td>
          <td>0.25465</td>
          <td>Title of passenger</td>
          <td>Miss</td>
          <td>0.193198</td>
          <td>Age</td>
          <td>17</td>
          <td>0.0981314</td>
        </tr>
        <tr>
          <th>681</th>
          <td>1</td>
          <td>0.786956</td>
          <td>Sex</td>
          <td>female</td>
          <td>0.252464</td>
          <td>Title of passenger</td>
          <td>Miss</td>
          <td>0.187045</td>
          <td>Relatives such as brother or wife</td>
          <td>0</td>
          <td>0.0522808</td>
        </tr>
        <tr>
          <th>536</th>
          <td>1</td>
          <td>0.936170</td>
          <td>Sex</td>
          <td>female</td>
          <td>0.250703</td>
          <td>Title of passenger</td>
          <td>Miss</td>
          <td>0.193096</td>
          <td>Age</td>
          <td>7</td>
          <td>0.104632</td>
        </tr>
      </tbody>
    </table>
    </div>


