ColumnTransformer tutorial
==========================

This tutorial shows how to use ColumnTransformer to reverse data
preprocessing and display explicit labels.

We used Kaggle's `Titanic <https://www.kaggle.com/c/titanic/data>`__ dataset.

Content :

- Encode data with ColumnTransformer
- Build a Binary Classifier (Random Forest)
- Using Shapash
- Show inversed data

We want to implement an inverse transform function for ColumnTransformer based
on column position.

The top-Transform feature obtained after the ColumnTransformer
shouldn’t be sampled.

.. code:: ipython

    import numpy as np
    import pandas as pd
    from xgboost import XGBClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.preprocessing import OneHotEncoder
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

Create Target :

.. code:: ipython

    y = titan_df['Survived']
    X = titan_df.drop('Survived', axis=1)

Train a columns transformer with multiple transformers :

.. code:: ipython

    enc_columntransfo = ColumnTransformer(
                transformers=[
                    ('onehot', OneHotEncoder(), ['Pclass','Sex']),
                    ('ordinal', OrdinalEncoder(), ['Embarked','Title'])
                ],
                remainder='passthrough')
    X_transform = pd.DataFrame(enc_columntransfo.fit_transform(X, y))

Reaffect columns names for the remainder part :

.. code:: ipython

    #find index that didn't get transformation
    idx_col = enc_columntransfo.transformers_[2][2]
    #give the N-last index, the remainder index name
    start = len(X_transform.columns)-len(idx_col)
    X_transform.columns = X_transform.columns.tolist()[:start]+X.columns[idx_col].tolist()
    X_transform.head(2)




.. table::

    +-+-+-+-+-+-+--+---+-----+-----+-----+
    |0|1|2|3|4|5|6 |Age|SibSp|Parch|Fare |
    +=+=+=+=+=+=+==+===+=====+=====+=====+
    |0|0|1|0|1|2|11| 22|    1|    0| 7.25|
    +-+-+-+-+-+-+--+---+-----+-----+-----+
    |1|0|0|1|0|0|12| 38|    1|    0|71.28|
    +-+-+-+-+-+-+--+---+-----+-----+-----+
    |0|0|1|1|0|2| 8| 26|    0|    0| 7.92|
    +-+-+-+-+-+-+--+---+-----+-----+-----+
    |1|0|0|1|0|2|12| 35|    1|    0|53.10|
    +-+-+-+-+-+-+--+---+-----+-----+-----+
    |0|0|1|0|1|2|11| 35|    0|    0| 8.05|
    +-+-+-+-+-+-+--+---+-----+-----+-----+



Fit a model
-----------

.. code:: ipython

    Xtrain, Xtest, ytrain, ytest = train_test_split(X_transform, y, train_size=0.75, random_state=1)
    
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
        preprocessing=enc_columntransfo,
        model=clf 
    )


.. parsed-literal::

    Backend: Shap TreeExplainer


Visualize data in pandas
------------------------

.. code:: ipython

    #Cause in ColumnsTransformer we apply multiple transformer on the same column.
    #the Pclass column is now : TransformersName +  Pclass
    xpl.x_pred.head(4)




.. table::

    +-------------+----------+----------------+-------------+----+-----+-----+-----+
    |onehot_Pclass|onehot_Sex|ordinal_Embarked|ordinal_Title|Age |SibSp|Parch|Fare |
    +=============+==========+================+=============+====+=====+=====+=====+
    |First class  |female    |Southampton     |Mrs          |48.0|    0|    0|25.93|
    +-------------+----------+----------------+-------------+----+-----+-----+-----+
    |Third class  |male      |Southampton     |Mr           |29.5|    0|    0| 7.90|
    +-------------+----------+----------------+-------------+----+-----+-----+-----+
    |Second class |female    |Southampton     |Miss         |17.0|    0|    0|10.50|
    +-------------+----------+----------------+-------------+----+-----+-----+-----+
    |Third class  |female    |Queenstown      |Miss         |29.5|    0|    0| 8.14|
    +-------------+----------+----------------+-------------+----+-----+-----+-----+



.. code:: ipython

    xpl.x_init.head(4)




.. table::

    +-+-+-+-+-+-+--+----+-----+-----+-----+
    |0|1|2|3|4|5|6 |Age |SibSp|Parch|Fare |
    +=+=+=+=+=+=+==+====+=====+=====+=====+
    |1|0|0|1|0|2|12|48.0|    0|    0|25.93|
    +-+-+-+-+-+-+--+----+-----+-----+-----+
    |0|0|1|0|1|2|11|29.5|    0|    0| 7.90|
    +-+-+-+-+-+-+--+----+-----+-----+-----+
    |0|1|0|1|0|2| 8|17.0|    0|    0|10.50|
    +-+-+-+-+-+-+--+----+-----+-----+-----+
    |0|0|1|1|0|1| 8|29.5|    0|    0| 8.14|
    +-+-+-+-+-+-+--+----+-----+-----+-----+


