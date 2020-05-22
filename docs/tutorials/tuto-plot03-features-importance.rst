Features importance
===================

The methode Feature importance displays bar chart represent the sum of
absolute contribution values of each feature.

this method also makes it possible to represent this sum calculated on a
subset and to compare it with the total population

This short tutorial presents the different parameters you can use.

Contents: - Classification case: Specify the target modality to display.
- selection parameter to display a subset - max_features parameter
limits the number of features

Data from Kaggle `Titanic <https://www.kaggle.com/c/titanic/data>`__

.. code:: ipython3

    import pandas as pd
    from category_encoders import OrdinalEncoder
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.model_selection import train_test_split

Building Supervized Model
-------------------------

Load Titanic data

.. code:: ipython3

    from shapash.data.data_loader import data_loading
    titanic_df, titanic_dict = data_loading('titanic')
    del titanic_df['Name']
    y_df=titanic_df['Survived'].to_frame()
    X_df=titanic_df[titanic_df.columns.difference(['Survived'])]

.. code:: ipython3

    titanic_df.head()




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



Load Titanic data

.. code:: ipython3

    from category_encoders import OrdinalEncoder
    
    categorical_features = [col for col in X_df.columns if X_df[col].dtype == 'object']
    
    encoder = OrdinalEncoder(
        cols=categorical_features,
        handle_unknown='ignore',
        return_df=True).fit(X_df)
    
    X_df=encoder.transform(X_df)

Train / Test Split + model fitting

.. code:: ipython3

    Xtrain, Xtest, ytrain, ytest = train_test_split(X_df, y_df, train_size=0.75, random_state=7)

.. code:: ipython3

    clf = ExtraTreesClassifier(n_estimators=200).fit(Xtrain,ytrain)

First step: You need to Declare and Compile SmartExplainer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    from shapash.explainer.smart_explainer import SmartExplainer

.. code:: ipython3

    response_dict = {0: 'Death', 1:' Survival'}

.. code:: ipython3

    xpl = SmartExplainer(features_dict=titanic_dict, # Optional parameters
                         label_dict=response_dict) # Optional parameters, dicts specify labels 

.. code:: ipython3

    xpl.compile(
        x=Xtest,
        model=clf,
        preprocessing=encoder, # Optional: compile step can use inverse_transform method
    )


.. parsed-literal::

    Backend: Shap TreeExplainer


Display Feature Importance
--------------------------

.. code:: ipython3

    xpl.plot.features_importance()



.. image:: tuto-plot03-features-importance_files/tuto-plot03-features-importance_17_0.png


Multiclass: Select the target modality
--------------------------------------

Features importances sum and display the absolute contribution for one
target modality. you can change this modality, selectig with label
parameter:

xpl.plot.features_importance(label=‘Death’)

with label parameter you can specify target value, label or number

Focus and compare a subset
--------------------------

selection parameter specify the subset:

.. code:: ipython3

    sel = [581, 610, 524, 636, 298, 420, 568, 817, 363, 557,
           486, 252, 390, 505, 16, 290, 611, 148, 438, 23, 810,
           875, 206, 836, 143, 843, 436, 701, 681, 67, 10]

.. code:: ipython3

    xpl.plot.features_importance(selection=sel)



.. image:: tuto-plot03-features-importance_files/tuto-plot03-features-importance_21_0.png


Tune the number of features to display
--------------------------------------

Use max_features parameter (default value: 20)

.. code:: ipython3

    xpl.plot.features_importance(max_features=3)



.. image:: tuto-plot03-features-importance_files/tuto-plot03-features-importance_23_0.png

