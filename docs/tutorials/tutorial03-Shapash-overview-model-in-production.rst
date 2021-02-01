Shapash model in production - Overview
======================================

With this tutorial you: Understand how to create a Shapash
SmartPredictor to make prediction and have local explanation in
production with a simple use case.

This tutorial describes the different steps from training the model to
Shapash SmartPredictor deployment. A more detailed tutorial allows you
to know more about the SmartPredictor Object.

Contents: - Build a Regressor - Compile Shapash SmartExplainer - From
Shapash SmartExplainer to SmartPredictor - Save Shapash Smartpredictor
Object in pickle file - Make a prediction

Data from Kaggle `House
Prices <https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data>`__

.. code:: ipython3

    import pandas as pd
    from category_encoders import OrdinalEncoder
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import train_test_split

Step 1 : Exploration and training of the model
----------------------------------------------

Building Supervized Model
~~~~~~~~~~~~~~~~~~~~~~~~~

In this section, we train a Machine Learning supervized model with our
data House Prices.

.. code:: ipython3

    from shapash.data.data_loader import data_loading
    house_df, house_dict = data_loading('house_prices')

.. code:: ipython3

    y_df=house_df['SalePrice'].to_frame()
    X_df=house_df[house_df.columns.difference(['SalePrice'])]

Preprocessing step
^^^^^^^^^^^^^^^^^^

Encoding Categorical Features

.. code:: ipython3

    from category_encoders import OrdinalEncoder
    
    categorical_features = [col for col in X_df.columns if X_df[col].dtype == 'object']
    
    encoder = OrdinalEncoder(cols=categorical_features,
                             handle_unknown='ignore',
                             return_df=True).fit(X_df)
    
    X_encoded=encoder.transform(X_df)

Train / Test Split
^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    Xtrain, Xtest, ytrain, ytest = train_test_split(X_encoded, y_df, train_size=0.75, random_state=1)

Model Fitting
^^^^^^^^^^^^^

.. code:: ipython3

    regressor = LGBMRegressor(n_estimators=200).fit(Xtrain, ytrain)

.. code:: ipython3

    y_pred = pd.DataFrame(regressor.predict(Xtest), columns=['pred'], index=Xtest.index)

Understand my model with shapash
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this section, we use the SmartExplainer Object from shapash. - It
allows users to understand how the model works with the specified data.
- This object must be used only for data mining step. Shapash provides
another object for deployment. - In this tutorial, we are not exploring
possibilites of the SmartExplainer but others will. (see other
tutorials)

Declare and Compile SmartExplainer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    from shapash.explainer.smart_explainer import SmartExplainer

Use wording on features names to better understanding results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here, we use a wording to rename our features label with more
understandable terms. It’s usefull to make our local explainability more
operational and understandable for users. - To do this, we use the
house_dict dictionary which maps a description to each features. - We
can then use it features_dict as a parameter of the SmartExplainer.

.. code:: ipython3

    xpl = SmartExplainer(features_dict=house_dict)

**compile()** This method is the first step to understand model and
prediction. It performs the sorting of contributions, the reverse
preprocessing steps and all the calculations necessary for a quick
display of plots and efficient summary of explanation. (see
SmartExplainer documentation and tutorials)

.. code:: ipython3

    xpl.compile(
                x=Xtest,
                model=regressor,
                preprocessing=encoder, # Optional: compile step can use inverse_transform method
                y_pred=y_pred # Optional
                )


.. parsed-literal::

    Backend: Shap TreeExplainer


Understand results of your trained model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Then, we can easily get a first summary of the explanation of the model
results. - Here, we chose to get the 3 most contributive features for
each prediction. - We used a wording to get features names more
understandable in operationnal case.

.. code:: ipython3

    xpl.to_pandas(max_contrib=3).head()


.. parsed-literal::

    .. table:: 
    
        +--------+----------------------------------------+-------+--------------+----------------------------------------+-------+--------------+----------------------------------+-------------+--------------+
        |  pred  |               feature_1                |value_1|contribution_1|               feature_2                |value_2|contribution_2|            feature_3             |   value_3   |contribution_3|
        +========+========================================+=======+==============+========================================+=======+==============+==================================+=============+==============+
        |209141.3|Ground living area square feet          |   1792|       13710.4|Overall material and finish of the house|      7|       12776.3|Total square feet of basement area|          963|       -5103.0|
        +--------+----------------------------------------+-------+--------------+----------------------------------------+-------+--------------+----------------------------------+-------------+--------------+
        |178734.5|Ground living area square feet          |   2192|       29747.0|Overall material and finish of the house|      5|      -26151.3|Overall condition of the house    |            8|        9190.8|
        +--------+----------------------------------------+-------+--------------+----------------------------------------+-------+--------------+----------------------------------+-------------+--------------+
        |113950.8|Overall material and finish of the house|      5|      -24730.0|Ground living area square feet          |    900|      -16342.6|Total square feet of basement area|          882|       -5922.6|
        +--------+----------------------------------------+-------+--------------+----------------------------------------+-------+--------------+----------------------------------+-------------+--------------+
        | 74957.2|Overall material and finish of the house|      4|      -33927.7|Ground living area square feet          |    630|      -23234.4|Total square feet of basement area|          630|      -11687.9|
        +--------+----------------------------------------+-------+--------------+----------------------------------------+-------+--------------+----------------------------------+-------------+--------------+
        |135305.2|Overall material and finish of the house|      5|      -25445.7|Ground living area square feet          |   1188|      -11476.6|Condition of sale                 |Abnormal Sale|       -5071.8|
        +--------+----------------------------------------+-------+--------------+----------------------------------------+-------+--------------+----------------------------------+-------------+--------------+


Step 2 : SmartPredictor in production
-------------------------------------

Switch from SmartExplainer to SmartPredictor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you are satisfied by your results and the explainablity given by
Shapash, you can use the SmartPredictor object for deployment. - In this
section, we learn how to easily switch from SmartExplainer to a
SmartPredictor. - SmartPredictor allows you to make predictions, detail
and summarize contributions on new data automatically. - It only keeps
the attributes needed for deployment to be lighter than the
SmartExplainer object. - SmartPredictor performs additional consistency
checks before deployment. - SmartPredictor allows you to configure the
way of summary to suit your use cases. - It can be used with API or in
batch mode.

.. code:: ipython3

    predictor = xpl.to_smartpredictor()

Save and Load your SmartPredictor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can easily save and load your SmartPredictor Object in pickle.

Save your SmartPredictor in Pickle File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    predictor.save('./predictor.pkl')

Load your SmartPredictor in Pickle File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    from shapash.utils.load_smartpredictor import load_smartpredictor

.. code:: ipython3

    predictor_load = load_smartpredictor('./predictor.pkl')

Make a prediction with your SmartPredictor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to make new predictions and summarize local explainability of
your model on new datasets, you can use the method add_input of the
SmartPredictor. - The add_input method is the first step to add a
dataset for prediction and explainability. - It checks the structure of
the dataset, the prediction and the contribution if specified. - It
applies the preprocessing specified in the initialisation and reorder
the features with the order used by the model. (see the documentation of
this method) - In API mode, this method can handle dictionnaries data
which can be received from a GET or a POST request.

Add data
^^^^^^^^

The x input in add_input method doesn’t have to be encoded, add_input
applies preprocessing.

.. code:: ipython3

    predictor_load.add_input(x=X_df, ypred=y_df)

Make prediction
^^^^^^^^^^^^^^^

Then, we can see ypred is the one given in add_input method by checking
the attribute data[“ypred”]. If not specified, it will automatically be
computed in the method.

.. code:: ipython3

    predictor_load.data["ypred"].head()


.. parsed-literal::

    .. table:: 
    
        +---------+
        |SalePrice|
        +=========+
        |   208500|
        +---------+
        |   181500|
        +---------+
        |   223500|
        +---------+
        |   140000|
        +---------+
        |   250000|
        +---------+


Get detailed explanability associated to the prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use the method detail_contributions to see the detailed
contributions of each of your features for each row of your new dataset.
- For classification problems, it automatically associates contributions
with the right predicted label. - The predicted label can be computed
automatically in the method or you can specify an ypred with add_input
method.

.. code:: ipython3

    detailed_contributions = predictor_load.detail_contributions()

.. code:: ipython3

    detailed_contributions.head()


.. parsed-literal::

    .. table:: 
    
        +---------+--------+--------+---------+------------+--------+--------+------------+----------+----------+------------+------------+------------+------------+--------+---------+----------+----------+----------+----------+-------------+---------+---------+-----------+-----------+----------+----------+--------+----------+----------+----------+------------+----------+----------+-----------+---------+--------+-------+---------+----------+------------+-----------+-----------+---------+--------+---------+--------+------------+----------+--------+----------+----------+-------+-------+------------+-----------+-----------+-----------+----------+--------+--------+---------+-------------+--------+-----------+------+------------+-----------+---------+----------+---------+------------+-------+
        |SalePrice|1stFlrSF|2ndFlrSF|3SsnPorch|BedroomAbvGr|BldgType|BsmtCond|BsmtExposure|BsmtFinSF1|BsmtFinSF2|BsmtFinType1|BsmtFinType2|BsmtFullBath|BsmtHalfBath|BsmtQual|BsmtUnfSF|CentralAir|Condition1|Condition2|Electrical|EnclosedPorch|ExterCond|ExterQual|Exterior1st|Exterior2nd|Fireplaces|Foundation|FullBath|Functional|GarageArea|GarageCond|GarageFinish|GarageQual|GarageType|GarageYrBlt|GrLivArea|HalfBath|Heating|HeatingQC|HouseStyle|KitchenAbvGr|KitchenQual|LandContour|LandSlope|LotArea |LotConfig|LotShape|LowQualFinSF|MSSubClass|MSZoning|MasVnrArea|MasVnrType|MiscVal|MoSold |Neighborhood|OpenPorchSF|OverallCond|OverallQual|PavedDrive|PoolArea|RoofMatl|RoofStyle|SaleCondition|SaleType|ScreenPorch|Street|TotRmsAbvGrd|TotalBsmtSF|Utilities|WoodDeckSF|YearBuilt|YearRemodAdd|YrSold |
        +=========+========+========+=========+============+========+========+============+==========+==========+============+============+============+============+========+=========+==========+==========+==========+==========+=============+=========+=========+===========+===========+==========+==========+========+==========+==========+==========+============+==========+==========+===========+=========+========+=======+=========+==========+============+===========+===========+=========+========+=========+========+============+==========+========+==========+==========+=======+=======+============+===========+===========+===========+==========+========+========+=========+=============+========+===========+======+============+===========+=========+==========+=========+============+=======+
        |   208500| -1105.0|  1281.4|        0|       375.7|  12.260|   157.2|      -233.0|   -738.45|    -59.29|      1756.7|      -4.464|      1457.5|     -12.514| -156.91|   3769.6|     87.32|     406.3|         0|   -102.72|        64.69|    80.49|    36.93|      395.4|      457.4|    -684.7|     241.8|  -166.0|     335.0|    3107.9|     34.90|      -28.35|     304.5|     832.4|      226.1|   2706.5|   286.1| -17.38|    73.05|    14.206|       71.56|    -1032.4|     -7.534|        0|  -12.51|  -276.76| -109.91|           0|    2069.9|   175.0|     703.6|   -0.7997|-15.600| -629.7|      456.89|     1347.2|    -1507.9|     8248.8|     58.86|       0|       0|   -17.47|       385.57| -104.65|     -351.6|     0|      -498.2|    -5165.5|        0|    -944.0|   3871.0|      2219.3|  17.48|
        +---------+--------+--------+---------+------------+--------+--------+------------+----------+----------+------------+------------+------------+------------+--------+---------+----------+----------+----------+----------+-------------+---------+---------+-----------+-----------+----------+----------+--------+----------+----------+----------+------------+----------+----------+-----------+---------+--------+-------+---------+----------+------------+-----------+-----------+---------+--------+---------+--------+------------+----------+--------+----------+----------+-------+-------+------------+-----------+-----------+-----------+----------+--------+--------+---------+-------------+--------+-----------+------+------------+-----------+---------+----------+---------+------------+-------+
        |   181500|  2249.4|  -655.9|        0|       123.9|  -9.270|   139.4|      2699.2|   5102.47|    -84.77|      1047.8|      -3.002|      -590.0|      80.065|  498.26|    565.0|    231.49|   -1172.9|         0|    -71.37|        12.21|    28.83|  -779.36|      372.0|     -244.7|    4450.3|    -148.5|   624.4|     358.4|    -491.4|     49.45|      358.83|     177.8|     429.8|     -892.4|  -9238.1|  -302.1| -20.96|   -68.94|   -17.389|       87.89|    -1036.3|    217.966|        0| -877.56|  -495.82| -288.12|           0|     522.2|   365.1|    -466.9|  -90.6664|-17.083|  383.8|     3668.55|     -645.7|     6371.6|   -14419.4|     50.37|       0|       0|   -58.25|       263.49| -153.84|     -236.5|     0|      -705.1|     2989.0|        0|    2090.8|    323.9|     -3861.8| 424.38|
        +---------+--------+--------+---------+------------+--------+--------+------------+----------+----------+------------+------------+------------+------------+--------+---------+----------+----------+----------+----------+-------------+---------+---------+-----------+-----------+----------+----------+--------+----------+----------+----------+------------+----------+----------+-----------+---------+--------+-------+---------+----------+------------+-----------+-----------+---------+--------+---------+--------+------------+----------+--------+----------+----------+-------+-------+------------+-----------+-----------+-----------+----------+--------+--------+---------+-------------+--------+-----------+------+------------+-----------+---------+----------+---------+------------+-------+
        |   223500| -1426.8|  -616.1|        0|       369.5|   9.211|   199.2|      1032.3|    -92.18|    -93.17|      1302.4|      -2.108|      1777.8|     -14.254|  -70.43|   1140.2|     85.60|     400.1|         0|   -132.63|        47.11|    64.68|   598.39|      258.0|      323.2|    1686.9|     454.9|  -162.6|     407.6|    6259.5|     27.05|       13.04|     290.1|     547.6|     -334.6|  15880.4|   550.2| -16.54|    76.16|   -20.322|       56.49|     -553.5|    -11.249|        0| -396.04|  -247.38|  592.57|           0|    2459.9|   180.0|    -192.9|   -9.6968|-17.386| -622.0|       77.92|      353.9|    -1404.5|     9651.3|     67.89|       0|       0|    39.71|       752.11|  -91.18|     -280.8|     0|      -324.7|    -5338.3|        0|    -777.7|   3837.8|      2192.9| -98.97|
        +---------+--------+--------+---------+------------+--------+--------+------------+----------+----------+------------+------------+------------+------------+--------+---------+----------+----------+----------+----------+-------------+---------+---------+-----------+-----------+----------+----------+--------+----------+----------+----------+------------+----------+----------+-----------+---------+--------+-------+---------+----------+------------+-----------+-----------+---------+--------+---------+--------+------------+----------+--------+----------+----------+-------+-------+------------+-----------+-----------+-----------+----------+--------+--------+---------+-------------+--------+-----------+------+------------+-----------+---------+----------+---------+------------+-------+
        |   140000|  -653.9|   121.5|        0|       307.7|   9.720|   252.8|      -530.2|  -2987.65|    -77.04|      1517.8|      -7.561|       756.4|     -13.619| -581.85|   -395.2|    155.30|     401.9|         0|    -91.37|      -457.47|   101.75|  -941.60|     -125.1|     -833.8|    1464.7|    -268.0|  -285.0|     366.4|    4384.3|     23.39|      -40.75|     237.8|   -1527.6|      247.1|    714.0|  -275.6| -17.23|    57.35|     2.768|       66.91|    -1281.8|   -112.096|        0|-1259.79|    39.13|   85.86|           0|   -3676.3|   710.7|    -450.7|  -58.7978|-18.776| -651.3|     2445.47|      332.9|    -1668.0|     4330.5|     78.78|       0|       0|   184.16|     -2943.15| -114.61|     -338.4|     0|      -635.1|    -6548.5|        0|    -974.5|  -3386.4|     -5232.5|1633.76|
        +---------+--------+--------+---------+------------+--------+--------+------------+----------+----------+------------+------------+------------+------------+--------+---------+----------+----------+----------+----------+-------------+---------+---------+-----------+-----------+----------+----------+--------+----------+----------+----------+------------+----------+----------+-----------+---------+--------+-------+---------+----------+------------+-----------+-----------+---------+--------+---------+--------+------------+----------+--------+----------+----------+-------+-------+------------+-----------+-----------+-----------+----------+--------+--------+---------+-------------+--------+-----------+------+------------+-----------+---------+----------+---------+------------+-------+
        |   250000| -9531.6| -1097.6|        0|     -1575.0|   7.454|   130.5|       623.9|  -2396.57|    -92.93|       457.7|       1.101|       678.3|      -4.190| -428.88|   -523.6|    123.25|     216.1|         0|    -56.58|        24.05|    38.09| -1958.67|      338.7|      232.6|    1275.9|     188.7|  -302.3|     253.8|   14907.7|     21.33|       37.02|     121.3|     223.2|     -235.9|  17176.5|   551.0| -19.49|    75.31|  -106.781|       52.57|    -4658.3|   -105.551|        0| 8143.13|  -668.53|  526.70|           0|     837.5|   135.9|    6393.7|  276.8572| -8.900|-4439.1|     -728.98|     -827.2|    -2231.6|    55722.1|     42.89|       0|       0|   -74.76|       -58.91| -481.12|     -366.3|     0|     -4733.6|    -4675.7|        0|     165.7|   2334.7|      1355.4|-395.13|
        +---------+--------+--------+---------+------------+--------+--------+------------+----------+----------+------------+------------+------------+------------+--------+---------+----------+----------+----------+----------+-------------+---------+---------+-----------+-----------+----------+----------+--------+----------+----------+----------+------------+----------+----------+-----------+---------+--------+-------+---------+----------+------------+-----------+-----------+---------+--------+---------+--------+------------+----------+--------+----------+----------+-------+-------+------------+-----------+-----------+-----------+----------+--------+--------+---------+-------------+--------+-----------+------+------------+-----------+---------+----------+---------+------------+-------+


Summarize explanability of the predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  You can use the summarize method to summarize your local
   explainability
-  This summary can be configured with modify_mask method so that you
   have explainability that meets your operational needs.
-  When you initialize the SmartPredictor, you can also specify : >-
   postprocessing: to apply a wording to several values of your dataset.
   >- label_dict: to rename your label for classification problems. >-
   features_dict: to rename your features.

.. code:: ipython3

    predictor_load.modify_mask(max_contrib=3)

.. code:: ipython3

    explanation = predictor_load.summarize()

For example, here, we chose to build a summary with 3 most contributive
features of your dataset. - As you can see below, the wording defined in
the first step of this tutorial has been kept by the SmartPredictor and
used in the summarize method.

.. code:: ipython3

    explanation.head()


.. parsed-literal::

    .. table:: 
    
        +---------+----------------------------------------+-------+--------------+----------------------------------------+-------+--------------+------------------------------+-------+--------------+
        |SalePrice|               feature_1                |value_1|contribution_1|               feature_2                |value_2|contribution_2|          feature_3           |value_3|contribution_3|
        +=========+========================================+=======+==============+========================================+=======+==============+==============================+=======+==============+
        |   208500|Overall material and finish of the house|      7|        8248.8|Total square feet of basement area      |    856|       -5165.5|Original construction date    |   2003|        3871.0|
        +---------+----------------------------------------+-------+--------------+----------------------------------------+-------+--------------+------------------------------+-------+--------------+
        |   181500|Overall material and finish of the house|      6|      -14419.4|Ground living area square feet          |   1262|       -9238.1|Overall condition of the house|      8|        6371.6|
        +---------+----------------------------------------+-------+--------------+----------------------------------------+-------+--------------+------------------------------+-------+--------------+
        |   223500|Ground living area square feet          |   1786|       15880.4|Overall material and finish of the house|      7|        9651.3|Size of garage in square feet |    608|        6259.5|
        +---------+----------------------------------------+-------+--------------+----------------------------------------+-------+--------------+------------------------------+-------+--------------+
        |   140000|Total square feet of basement area      |    756|       -6548.5|Remodel date                            |   1970|       -5232.5|Size of garage in square feet |    642|        4384.3|
        +---------+----------------------------------------+-------+--------------+----------------------------------------+-------+--------------+------------------------------+-------+--------------+
        |   250000|Overall material and finish of the house|      8|       55722.1|Ground living area square feet          |   2198|       17176.5|Size of garage in square feet |    836|       14907.7|
        +---------+----------------------------------------+-------+--------------+----------------------------------------+-------+--------------+------------------------------+-------+--------------+

