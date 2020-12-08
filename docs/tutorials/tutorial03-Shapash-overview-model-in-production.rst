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
    
    encoder = OrdinalEncoder(cols = categorical_features,
                             handle_unknown = 'ignore',
                             return_df = True).fit(X_df)
    
    X_df=encoder.transform(X_df)

Train / Test Split
^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    Xtrain, Xtest, ytrain, ytest = train_test_split(X_df, y_df, train_size = 0.75, random_state = 1)

Model Fitting
^^^^^^^^^^^^^

.. code:: ipython3

    regressor = LGBMRegressor(n_estimators = 200).fit(Xtrain, ytrain)

.. code:: ipython3

    y_pred = pd.DataFrame(regressor.predict(Xtest), columns = ['pred'], index = Xtest.index)

Understand my model with shapash
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  In this section, we use the SmartExplainer Object from shapash which
   allow the users to understand how the model works with the specified
   data.
-  This object must be used only for data mining step. Shapash provides
   another object for deployment.
-  In this tutorial, we are not explore possibilites of the
   SmartExplainer but others will. (Look at them)

Declare and Compile SmartExplainer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    from shapash.explainer.smart_explainer import SmartExplainer

Use wording on features names to better understanding results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here, we use a wording to rename our features label with more
understandable terms. It’s usefull to make our local explainability more
operational and understandable for all users. - To do this, we use the
house_dict dictionary which maps a description to each features. - We
can then use it features_dict as a parameter of the SmartExplainer.

.. code:: ipython3

    xpl = SmartExplainer(features_dict = house_dict)

Then, we need to use the compile method of the SmartExplainer Object.
This method is the first step to understand model and prediction. It
performs the sorting of contributions, the reverse preprocessing steps
and performs all the calculations necessary for a quick display of plots
and efficient display of summary of explanation. (see the documentation
on SmartExplainer Object and the associated tutorials to go further)

.. code:: ipython3

    xpl.compile(
                x = Xtest,
                model = regressor,
                preprocessing = encoder, # Optional: compile step can use inverse_transform method
                y_pred = y_pred # Optional
                )


.. parsed-literal::

    Backend: Shap TreeExplainer


Understand results of your trained model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Then, we can easily get a first summary of the explanation of the model
results. - Here, we chose to get the 3 most contributive features for
each prediction - We used a wording to get features names more
understandable in operationnal case.

.. code:: ipython3

    xpl.to_pandas(max_contrib = 3).head()


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

-  When you are satisfied by your results and the explainablity given by
   Shapash, you can use the SmartPredictor object for deployment.
-  In this section, we learn how to easily switch from SmartExplainer to
   a SmartPredictor.
-  SmartPredictor allows you to make predictions, detailed and
   summarized contributions on new data automatically.
-  It only keeps the attributes needed for deployment to be lighter than
   the SmartExplainer object.
-  SmartPredictor performs additional consistency checks before
   deployment.
-  SmartPredictor allows you to configure the way of summary to suit
   your use cases.
-  It can be use with API or in batch mode.

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

-  In order to make new predictions and summarize local explainability
   of your model on new datasets, you can use the method add_input of
   the SmartPredictor.
-  The add_input method is the first step to add a dataset for
   prediction and explainability.
-  It checks the structure of the dataset, the prediction and the
   contribution if specified.
-  It applies the preprocessing specified in the initialisation and
   reorder the features with the order used by the model. (see the
   documentation on this method)
-  In API mode, this method can handle dictionnaries data which can be
   received from a GET or a POST request.

Add data
^^^^^^^^

.. code:: ipython3

    predictor_load.add_input(x = X_df, ypred = y_df)

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

-  You can use the method detail_contributions to see the detailed
   contributions of each of your features for each row of your new
   dataset.
-  For classification problems, it automatically associates
   contributions with the right predicted label.
-  The predicted label can be computed automatically with predict method
   or you can specify in add_input method an ypred

.. code:: ipython3

    detailed_contributions = predictor_load.detail_contributions()

.. code:: ipython3

    detailed_contributions.head()


.. parsed-literal::

    .. table:: 
    
        +---------+--------+--------+---------+------------+--------+--------+------------+----------+----------+------------+------------+------------+------------+--------+---------+----------+----------+----------+----------+-------------+---------+---------+-----------+-----------+----------+----------+--------+----------+----------+----------+------------+----------+----------+-----------+---------+--------+-------+---------+----------+------------+-----------+-----------+---------+-------+---------+--------+------------+----------+--------+----------+----------+-------+-------+------------+-----------+-----------+-----------+----------+--------+--------+---------+-------------+--------+-----------+------+------------+-----------+---------+----------+---------+------------+-------+
        |SalePrice|1stFlrSF|2ndFlrSF|3SsnPorch|BedroomAbvGr|BldgType|BsmtCond|BsmtExposure|BsmtFinSF1|BsmtFinSF2|BsmtFinType1|BsmtFinType2|BsmtFullBath|BsmtHalfBath|BsmtQual|BsmtUnfSF|CentralAir|Condition1|Condition2|Electrical|EnclosedPorch|ExterCond|ExterQual|Exterior1st|Exterior2nd|Fireplaces|Foundation|FullBath|Functional|GarageArea|GarageCond|GarageFinish|GarageQual|GarageType|GarageYrBlt|GrLivArea|HalfBath|Heating|HeatingQC|HouseStyle|KitchenAbvGr|KitchenQual|LandContour|LandSlope|LotArea|LotConfig|LotShape|LowQualFinSF|MSSubClass|MSZoning|MasVnrArea|MasVnrType|MiscVal|MoSold |Neighborhood|OpenPorchSF|OverallCond|OverallQual|PavedDrive|PoolArea|RoofMatl|RoofStyle|SaleCondition|SaleType|ScreenPorch|Street|TotRmsAbvGrd|TotalBsmtSF|Utilities|WoodDeckSF|YearBuilt|YearRemodAdd|YrSold |
        +=========+========+========+=========+============+========+========+============+==========+==========+============+============+============+============+========+=========+==========+==========+==========+==========+=============+=========+=========+===========+===========+==========+==========+========+==========+==========+==========+============+==========+==========+===========+=========+========+=======+=========+==========+============+===========+===========+=========+=======+=========+========+============+==========+========+==========+==========+=======+=======+============+===========+===========+===========+==========+========+========+=========+=============+========+===========+======+============+===========+=========+==========+=========+============+=======+
        |   208500| -1105.0| 1281.45|        0|       375.7|  12.260|   157.2|      -233.0|   -738.45|    -59.29|      1756.7|      -4.464|      1457.5|     -12.514| -156.91|   3769.6|     87.32|     406.3|         0|   -102.72|       64.689|    80.49|    36.93|     395.35|      457.4|    -684.7|     241.8|  -166.0|     335.0|    3107.9|     34.90|     -28.351|     304.5|     832.4|      226.1|   2706.5|   286.1| -17.38|    73.05|    14.206|       71.56|    -1032.4|     -7.534|        0| -12.51|   -276.8|  -109.9|           0|    2069.9|   175.0|     703.6|   -0.7997|-15.600| -629.7|       456.9|     1347.2|    -1507.9|     8248.8|     58.86|       0|       0|  -17.468|       385.57| -104.65|     -351.6|     0|      -498.2|    -5165.5|        0|    -944.0|   3871.0|      2219.3|  17.48|
        +---------+--------+--------+---------+------------+--------+--------+------------+----------+----------+------------+------------+------------+------------+--------+---------+----------+----------+----------+----------+-------------+---------+---------+-----------+-----------+----------+----------+--------+----------+----------+----------+------------+----------+----------+-----------+---------+--------+-------+---------+----------+------------+-----------+-----------+---------+-------+---------+--------+------------+----------+--------+----------+----------+-------+-------+------------+-----------+-----------+-----------+----------+--------+--------+---------+-------------+--------+-----------+------+------------+-----------+---------+----------+---------+------------+-------+
        |   181500|  1629.1| -683.69|        0|       127.2|   8.045|   166.5|     -1112.6|   5781.67|    -76.74|      1545.9|      -3.002|      -612.1|      80.065|  484.04|    611.0|    238.35|     513.5|         0|    -72.65|       -4.472|    34.11|  -217.79|     340.65|     -103.3|    4165.2|     436.3|   623.7|     356.6|    -711.4|     51.74|     335.442|     197.4|     288.4|     -962.5| -10016.3|  -294.7| -20.87|   -33.75|    25.084|       88.06|      114.2|     80.720|        0|-794.90|   -100.0|  -319.9|           0|     902.7|   343.6|    -511.0|   58.2999|-18.709|  364.7|      2753.1|     -532.2|     6899.3|   -14555.9|     50.87|       0|       0|  -57.006|       306.40| -229.80|     -217.5|     0|      -546.0|     2783.7|        0|    2388.1|    340.2|     -4310.0| 413.35|
        +---------+--------+--------+---------+------------+--------+--------+------------+----------+----------+------------+------------+------------+------------+--------+---------+----------+----------+----------+----------+-------------+---------+---------+-----------+-----------+----------+----------+--------+----------+----------+----------+------------+----------+----------+-----------+---------+--------+-------+---------+----------+------------+-----------+-----------+---------+-------+---------+--------+------------+----------+--------+----------+----------+-------+-------+------------+-----------+-----------+-----------+----------+--------+--------+---------+-------------+--------+-----------+------+------------+-----------+---------+----------+---------+------------+-------+
        |   223500| -1321.1| -556.40|        0|       361.5|  10.475|   197.2|      -532.0|     61.50|    -84.60|      1440.2|      -2.108|      1806.2|     -14.254|  -65.43|    927.8|     89.36|     399.9|         0|   -132.47|       28.185|    69.26|   656.77|     114.67|      440.1|    1218.0|     456.0|  -171.0|     415.1|    5998.6|     29.34|      20.654|     290.1|     518.2|     -168.8|  15708.3|   577.7| -15.56|    59.28|   -24.845|       56.33|     -519.5|    -28.963|        0|-402.46|   -248.8|  -506.4|           0|    2473.1|   175.7|    -295.7|  -12.2395|-18.589| -393.4|       260.4|      207.8|    -1630.0|    11084.5|     67.35|       0|       0|   48.150|       759.31|  -91.18|     -323.3|     0|      -178.8|    -5157.3|        0|    -919.5|   3877.0|      2141.7| -72.95|
        +---------+--------+--------+---------+------------+--------+--------+------------+----------+----------+------------+------------+------------+------------+--------+---------+----------+----------+----------+----------+-------------+---------+---------+-----------+-----------+----------+----------+--------+----------+----------+----------+------------+----------+----------+-----------+---------+--------+-------+---------+----------+------------+-----------+-----------+---------+-------+---------+--------+------------+----------+--------+----------+----------+-------+-------+------------+-----------+-----------+-----------+----------+--------+--------+---------+-------------+--------+-----------+------+------------+-----------+---------+----------+---------+------------+-------+
        |   140000|  -991.6|   20.08|        0|       310.4|   9.720|   226.6|      -502.5|  -3170.03|    -95.89|      1441.0|      -4.973|       963.5|     -13.619| -234.37|   -289.7|    158.14|     432.3|         0|   -103.34|     -707.714|   114.40|   -80.38|      82.37|      211.0|    1462.0|     206.6|  -294.7|     387.1|    6651.6|     23.95|      -2.171|     290.4|     679.0|      315.7|   2969.7|  -263.4| -17.00|   419.86|    -2.777|       68.04|    -1288.9|    -86.747|        0|-825.75|   -245.6|  -291.1|           0|    2767.3|   415.8|    -709.2|   13.9822|-18.257| -889.9|      1585.2|      452.0|    -1875.1|     8188.4|     69.15|       0|       0|   86.058|       345.70|  -89.32|     -344.8|     0|      -608.0|    -5882.2|        0|    -853.1|  -3740.8|     -4930.9| 555.38|
        +---------+--------+--------+---------+------------+--------+--------+------------+----------+----------+------------+------------+------------+------------+--------+---------+----------+----------+----------+----------+-------------+---------+---------+-----------+-----------+----------+----------+--------+----------+----------+----------+------------+----------+----------+-----------+---------+--------+-------+---------+----------+------------+-----------+-----------+---------+-------+---------+--------+------------+----------+--------+----------+----------+-------+-------+------------+-----------+-----------+-----------+----------+--------+--------+---------+-------------+--------+-----------+------+------------+-----------+---------+----------+---------+------------+-------+
        |   250000| -8807.7|-1061.02|        0|     -1580.4|   7.868|   124.9|      -237.6|  -2109.99|    -95.46|       603.6|       1.101|       833.5|      -4.190| -392.37|   -477.5|    125.15|     200.8|         0|    -56.36|       18.642|    39.93| -1889.29|     253.88|      259.9|     886.0|     190.1|  -309.1|     252.5|   15161.9|     21.99|      22.500|     121.3|     218.2|     -361.6|  16891.9|   577.7| -18.30|    72.30|  -113.239|       52.48|    -4611.8|    -97.218|        0|7905.51|   -412.6|  -498.7|           0|     875.5|   129.9|    6318.0|  266.8708| -9.056|-4240.1|      -214.7|     -828.3|    -2403.3|    58568.4|     43.47|       0|       0|   -9.469|       -50.49| -481.12|     -384.1|     0|     -4071.6|    -4866.8|        0|     270.9|   2394.7|      1533.3|-233.44|
        +---------+--------+--------+---------+------------+--------+--------+------------+----------+----------+------------+------------+------------+------------+--------+---------+----------+----------+----------+----------+-------------+---------+---------+-----------+-----------+----------+----------+--------+----------+----------+----------+------------+----------+----------+-----------+---------+--------+-------+---------+----------+------------+-----------+-----------+---------+-------+---------+--------+------------+----------+--------+----------+----------+-------+-------+------------+-----------+-----------+-----------+----------+--------+--------+---------+-------------+--------+-----------+------+------------+-----------+---------+----------+---------+------------+-------+


Summarize explanability of the predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  You can use the summarize method to summarize your local
   explainability
-  This summary can be configured with modify_mask method so that you
   have explainability that meets your operational needs.
-  You can also specify : >- a postprocessing when you initialize your
   SmartPredictor to apply a wording to several values of your dataset.
   >- a label_dict to rename your label in classification problems
   (during the initialisation of your SmartPredictor). >- a
   features_dict to rename your features.

.. code:: ipython3

    predictor_load.modify_mask(max_contrib = 3)

.. code:: ipython3

    explanation = predictor_load.summarize()

For example, here, we chose to only build a summary with 3 most
contributive features of your dataset. - As you can see below, the
wording defined in the first step of this tutorial has been kept by the
SmartPredictor and used in the summarize method.

.. code:: ipython3

    explanation.head()


.. parsed-literal::

    .. table:: 
    
        +---------+----------------------------------------+-------+--------------+----------------------------------------+-------+--------------+----------------------------------+-------+--------------+
        |SalePrice|               feature_1                |value_1|contribution_1|               feature_2                |value_2|contribution_2|            feature_3             |value_3|contribution_3|
        +=========+========================================+=======+==============+========================================+=======+==============+==================================+=======+==============+
        |   208500|Overall material and finish of the house|      7|        8248.8|Total square feet of basement area      |    856|       -5165.5|Original construction date        |   2003|        3871.0|
        +---------+----------------------------------------+-------+--------------+----------------------------------------+-------+--------------+----------------------------------+-------+--------------+
        |   181500|Overall material and finish of the house|      6|      -14555.9|Ground living area square feet          |   1262|      -10016.3|Overall condition of the house    |      8|        6899.3|
        +---------+----------------------------------------+-------+--------------+----------------------------------------+-------+--------------+----------------------------------+-------+--------------+
        |   223500|Ground living area square feet          |   1786|       15708.3|Overall material and finish of the house|      7|       11084.5|Size of garage in square feet     |    608|        5998.6|
        +---------+----------------------------------------+-------+--------------+----------------------------------------+-------+--------------+----------------------------------+-------+--------------+
        |   140000|Overall material and finish of the house|      7|        8188.4|Size of garage in square feet           |    642|        6651.6|Total square feet of basement area|    756|       -5882.2|
        +---------+----------------------------------------+-------+--------------+----------------------------------------+-------+--------------+----------------------------------+-------+--------------+
        |   250000|Overall material and finish of the house|      8|       58568.4|Ground living area square feet          |   2198|       16891.9|Size of garage in square feet     |    836|       15161.9|
        +---------+----------------------------------------+-------+--------------+----------------------------------------+-------+--------------+----------------------------------+-------+--------------+

