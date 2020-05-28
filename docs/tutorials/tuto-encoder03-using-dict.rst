Dictionnary Encoding tutorial
=============================

This tutorial shows how to use simple python dictionnaries to reverse
data preprocessing and display explicit labels

Data from Kaggle `Titanic <https://www.kaggle.com/c/titanic>`__

This Tutorial: - Encode data with dictionary - Build a Binary Classifier
(Random Forest) - Using Shapash - Show inversed data

.. code:: ipython

    import numpy as np
    import pandas as pd
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split

Load titanic Data
-----------------

.. code:: ipython

    from shapash.data.data_loader import data_loading
    
    titan_df, titan_dict = data_loading('titanic')
    del titan_df['Name']

.. code:: ipython

    titan_df.head()




.. table::

    +--------+-----------+------+---+-----+-----+-----+-----------+-----+
    |Survived|  Pclass   | Sex  |Age|SibSp|Parch|Fare | Embarked  |Title|
    +========+===========+======+===+=====+=====+=====+===========+=====+
    |       0|Third class|male  | 22|    1|    0| 7.25|Southampton|Mr   |
    +--------+-----------+------+---+-----+-----+-----+-----------+-----+
    |       1|First class|female| 38|    1|    0|71.28|Cherbourg  |Mrs  |
    +--------+-----------+------+---+-----+-----+-----+-----------+-----+
    |       1|Third class|female| 26|    0|    0| 7.92|Southampton|Miss |
    +--------+-----------+------+---+-----+-----+-----+-----------+-----+
    |       1|First class|female| 35|    1|    0|53.10|Southampton|Mrs  |
    +--------+-----------+------+---+-----+-----+-----+-----------+-----+
    |       0|Third class|male  | 35|    0|    0| 8.05|Southampton|Mr   |
    +--------+-----------+------+---+-----+-----+-----+-----------+-----+




Prepare data for the model
--------------------------

Create Target

.. code:: ipython

    y = titan_df['Survived']
    X = titan_df.drop('Survived', axis=1)

Train multiple category encoder

.. code:: ipython

    #construct new variable
    X['new_embarked'] = X.apply(lambda x : 1 if x.Embarked in ['Southampton','Cherbourg'] else 2 if x.Embarked in 'Queenstown' else 3, axis = 1)
    #Construct the reversed dict
    transfo_embarked = {'col': 'new_embarked',
                        'mapping': pd.Series(data=[1, 2, np.nan], index=['Southampton-Cherbourg', 'Queenstown','missing']),
                        'data_type': 'object'}
    
    #construct new variable
    X['new_ages'] = X.apply(lambda x : 1 if x.Age <= 25 else 2 if x.Age <= 40 else 3, axis = 1)
    #Construct the reversed dict
    transfo_age = dict()
    transfo_age = {'col': 'new_ages',
                    'mapping': pd.Series(data=[1, 2, 3, np.nan], index=['-25 years', '26-40 years', '+40 years','missing']),
                    'data_type': 'object'}

.. code:: ipython

    #put transformation into list
    encoder = [transfo_age,transfo_embarked]

.. code:: ipython

    X.head(4)




.. table::

    +-----------+------+---+-----+-----+-----+-----------+-----+------------+--------+
    |  Pclass   | Sex  |Age|SibSp|Parch|Fare | Embarked  |Title|new_embarked|new_ages|
    +===========+======+===+=====+=====+=====+===========+=====+============+========+
    |Third class|male  | 22|    1|    0| 7.25|Southampton|Mr   |           1|       1|
    +-----------+------+---+-----+-----+-----+-----------+-----+------------+--------+
    |First class|female| 38|    1|    0|71.28|Cherbourg  |Mrs  |           1|       2|
    +-----------+------+---+-----+-----+-----+-----------+-----+------------+--------+
    |Third class|female| 26|    0|    0| 7.92|Southampton|Miss |           1|       2|
    +-----------+------+---+-----+-----+-----+-----------+-----+------------+--------+
    |First class|female| 35|    1|    0|53.10|Southampton|Mrs  |           1|       2|
    +-----------+------+---+-----+-----+-----+-----------+-----+------------+--------+



Fit a model
-----------

.. code:: ipython

    X = X[['new_embarked','new_ages','Fare','Parch','Age']]

.. code:: ipython

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.75, random_state=1)
    
    clf = XGBClassifier(n_estimators=200,min_child_weight=2).fit(Xtrain,ytrain)
    clf.fit(Xtrain, ytrain)




.. parsed-literal::

    XGBClassifier(base_score=0.5, booster=None, colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                  importance_type='gain', interaction_constraints=None,
                  learning_rate=0.300000012, max_delta_step=0, max_depth=6,
                  min_child_weight=2, missing=nan, monotone_constraints=None,
                  n_estimators=200, n_jobs=0, num_parallel_tree=1,
                  objective='binary:logistic', random_state=0, reg_alpha=0,
                  reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,
                  validate_parameters=False, verbosity=None)



Using Shapash
-------------

.. code:: ipython

    from shapash.explainer.smart_explainer import SmartExplainer

.. code:: ipython

    xpl = SmartExplainer()

.. code:: ipython

    xpl.compile(
        x=Xtest,
        preprocessing=encoder,
        model=clf 
    )


.. parsed-literal::

    Backend: Shap TreeExplainer


Visualize data in pandas
------------------------

.. code:: ipython

    xpl.x_pred.head(4)




.. table::

    +---------------------+-----------+-----+-----+----+
    |    new_embarked     | new_ages  |Fare |Parch|Age |
    +=====================+===========+=====+=====+====+
    |Southampton-Cherbourg|+40 years  |25.93|    0|48.0|
    +---------------------+-----------+-----+-----+----+
    |Southampton-Cherbourg|26-40 years| 7.90|    0|29.5|
    +---------------------+-----------+-----+-----+----+
    |Southampton-Cherbourg|-25 years  |10.50|    0|17.0|
    +---------------------+-----------+-----+-----+----+
    |Queenstown           |26-40 years| 8.14|    0|29.5|
    +---------------------+-----------+-----+-----+----+



.. code:: ipython

    xpl.x_init.head(4)




.. table::

    +------------+--------+-----+-----+----+
    |new_embarked|new_ages|Fare |Parch|Age |
    +============+========+=====+=====+====+
    |           1|       3|25.93|    0|48.0|
    +------------+--------+-----+-----+----+
    |           1|       2| 7.90|    0|29.5|
    +------------+--------+-----+-----+----+
    |           1|       1|10.50|    0|17.0|
    +------------+--------+-----+-----+----+
    |           2|       2| 8.14|    0|29.5|
    +------------+--------+-----+-----+----+

