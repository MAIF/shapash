Category_encoder tutorial
=========================

This tutorial shows how to use category_encoder encoders to reverse data
preprocessing and display explicit labels.

We used Kaggle's `Titanic <https://www.kaggle.com/c/titanic/data>`__ dataset.

Content :

- Encode data with Category_encoder
- Build a binary classifier (Random Forest)
- Using Shapash
- Show inversed data

.. code:: ipython

    import numpy as np
    import pandas as pd
    from category_encoders import OrdinalEncoder
    from category_encoders import OneHotEncoder
    from category_encoders import TargetEncoder
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



Prepare data for the model with Category Encoder
------------------------------------------------

Create Target :

.. code:: ipython

    y = titan_df['Survived']
    X = titan_df.drop('Survived', axis=1)

Train category encoder :

.. code:: ipython

    #Train category encoder
    onehot = OneHotEncoder(cols=['Pclass']).fit(X)
    result_1 = onehot.transform(X)
    ordinal = OrdinalEncoder(cols=['Embarked','Title']).fit(result_1)
    result_2 = ordinal.transform(result_1)
    target = TargetEncoder(cols=['Sex']).fit(result_2,y)
    result_3 =target.transform(result_2)

.. code:: ipython

    encoder = [onehot,ordinal,target]

Fit a model
-----------

.. code:: ipython

    Xtrain, Xtest, ytrain, ytest = train_test_split(result_3, y, train_size=0.75, random_state=1)
    
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
        x=Xtest.head(10),
        preprocessing=encoder,
        model=clf
    )


.. parsed-literal::

    Backend: Shap TreeExplainer


Visualize data in pandas
------------------------

.. code:: ipython

    xpl.x_pred




.. table::

    +------------+------+----+-----+-----+-----+-----------+-----+
    |   Pclass   | Sex  |Age |SibSp|Parch|Fare | Embarked  |Title|
    +============+======+====+=====+=====+=====+===========+=====+
    |First class |female|48.0|    0|    0|25.93|Southampton|Mrs  |
    +------------+------+----+-----+-----+-----+-----------+-----+
    |Third class |male  |29.5|    0|    0| 7.90|Southampton|Mr   |
    +------------+------+----+-----+-----+-----+-----------+-----+
    |Second class|female|17.0|    0|    0|10.50|Southampton|Miss |
    +------------+------+----+-----+-----+-----+-----------+-----+
    |Third class |female|29.5|    0|    0| 8.14|Queenstown |Miss |
    +------------+------+----+-----+-----+-----+-----------+-----+
    |Second class|female| 7.0|    0|    2|26.25|Southampton|Miss |
    +------------+------+----+-----+-----+-----+-----------+-----+



.. code:: ipython

    xpl.x_init




.. table::

    +--------+--------+--------+------+----+-----+-----+-----+--------+-----+
    |Pclass_1|Pclass_2|Pclass_3| Sex  |Age |SibSp|Parch|Fare |Embarked|Title|
    +========+========+========+======+====+=====+=====+=====+========+=====+
    |       0|       1|       0|0.7420|48.0|    0|    0|25.93|       1|    2|
    +--------+--------+--------+------+----+-----+-----+-----+--------+-----+
    |       1|       0|       0|0.1889|29.5|    0|    0| 7.90|       1|    1|
    +--------+--------+--------+------+----+-----+-----+-----+--------+-----+
    |       0|       0|       1|0.7420|17.0|    0|    0|10.50|       1|    3|
    +--------+--------+--------+------+----+-----+-----+-----+--------+-----+
    |       1|       0|       0|0.7420|29.5|    0|    0| 8.14|       3|    3|
    +--------+--------+--------+------+----+-----+-----+-----+--------+-----+
    |       0|       0|       1|0.7420| 7.0|    0|    2|26.25|       1|    3|
    +--------+--------+--------+------+----+-----+-----+-----+--------+-----+


