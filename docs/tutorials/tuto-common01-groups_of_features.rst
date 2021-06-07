Groups of features
==================

| Shapash allows the use of groups of features.
| With groups of features you can regroup variables together and use the
  different functions of Shapash to analyze these groups.

| For example if your model uses a lot of features you may want to
  regroup features that share a common theme.
| This way **you can visualize and compare the importance of these
  themes and how they are used by your model.**

Contents of this tutorial:

- Build a model
- Contruct groups of features
- Compile Shapash SmartExplainer with the groups - Start Shapash WebApp
- Explore the functions of Shapash using groups

Data from Kaggle `House
Prices <https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data>`__

Motivation
----------

In this use case, we have a lot of features that describe the house very
precisely.

| However, when analyzing our model, **you may want to get more general
  insights of the themes that are most important in setting the price of
  a property**.
| This way, rather than having to check the 6 features describing a
  garage, you can have a more general idea of how important the garage
  is by grouping these 6 features together. Shapash allows you to
  visualize the role of each group in the features importance plot.

Also, you may want to understand why your model predicted such an
important price for a specific house. If many features describing the
location of the house are contributing slightly more than usual to a
higher price, **it may not be visible directly that the price is due to
the location because of the number of features**. But grouping these
variables together allows to easily understand a specific prediction.
Shapash also allows you to group features together in local plots.

.. code:: ipython3

    import pandas as pd
    from category_encoders import OrdinalEncoder
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import train_test_split

Building a supervized model
---------------------------

Load House Prices data
~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    from shapash.data.data_loader import data_loading
    house_df, house_dict = data_loading('house_prices')

.. code:: ipython3

    house_df.head()

.. table::

    +-------------------------------+-----------------------+-------+------+------------------+---------------+--------------------------------+-------------------------------+------------+-------------+-------------------------+----------+----------------------+----------+-----------+-----------+---------+------------+---------+----------------------------+------------+-------------+----------+----------+---------------+---------------+---------------+----------------------+---------------------------------+-----------------------+-----------------------+----------+----------------------+----------+---------+-----------+---------------------------+---------+----------+---------------------------------+--------+--------+------------+---------+------------+------------+--------+--------+------------+------------+---------------+------------+---------------------+----------+------------------+-----------+--------------------+----------+---------------+---------------+----------+----------+-----------+-------------+---------+-----------+--------+-------+------+------+----------------------------+-------------+---------+
    |          MSSubClass           |       MSZoning        |LotArea|Street|     LotShape     |  LandContour  |           Utilities            |           LotConfig           | LandSlope  |Neighborhood |       Condition1        |Condition2|       BldgType       |HouseStyle|OverallQual|OverallCond|YearBuilt|YearRemodAdd|RoofStyle|          RoofMatl          |Exterior1st | Exterior2nd |MasVnrType|MasVnrArea|   ExterQual   |   ExterCond   |  Foundation   |       BsmtQual       |            BsmtCond             |     BsmtExposure      |     BsmtFinType1      |BsmtFinSF1|     BsmtFinType2     |BsmtFinSF2|BsmtUnfSF|TotalBsmtSF|          Heating          |HeatingQC|CentralAir|           Electrical            |1stFlrSF|2ndFlrSF|LowQualFinSF|GrLivArea|BsmtFullBath|BsmtHalfBath|FullBath|HalfBath|BedroomAbvGr|KitchenAbvGr|  KitchenQual  |TotRmsAbvGrd|     Functional      |Fireplaces|    GarageType    |GarageYrBlt|    GarageFinish    |GarageArea|  GarageQual   |  GarageCond   |PavedDrive|WoodDeckSF|OpenPorchSF|EnclosedPorch|3SsnPorch|ScreenPorch|PoolArea|MiscVal|MoSold|YrSold|          SaleType          |SaleCondition|SalePrice|
    +===============================+=======================+=======+======+==================+===============+================================+===============================+============+=============+=========================+==========+======================+==========+===========+===========+=========+============+=========+============================+============+=============+==========+==========+===============+===============+===============+======================+=================================+=======================+=======================+==========+======================+==========+=========+===========+===========================+=========+==========+=================================+========+========+============+=========+============+============+========+========+============+============+===============+============+=====================+==========+==================+===========+====================+==========+===============+===============+==========+==========+===========+=============+=========+===========+========+=======+======+======+============================+=============+=========+
    |2-Story 1946 & Newer           |Residential Low Density|   8450|Paved |Regular           |Near Flat/Level|All public Utilities (E,G,W,& S)|Inside lot                     |Gentle slope|College Creek|Normal                   |Normal    |Single-family Detached|Two story |          7|          5|     2003|        2003|Gable    |Standard (Composite) Shingle|Vinyl Siding|Vinyl Siding |Brick Face|       196|Good           |Average/Typical|Poured Contrete|Good (90-99 inches)   |Typical - slight dampness allowed|No Exposure/No Basement|Good Living Quarters   |       706|Unfinished/No Basement|         0|      150|        856|Gas forced warm air furnace|Excellent|Yes       |Standard Circuit Breakers & Romex|     856|     854|           0|     1710|           1|           0|       2|       1|           3|           1|Good           |           8|Typical Functionality|         0|Attached to home  |       2003|Rough Finished      |       548|Typical/Average|Typical/Average|Paved     |         0|         61|            0|        0|          0|       0|      0|     2|  2008|Warranty Deed - Conventional|Normal Sale  |   208500|
    +-------------------------------+-----------------------+-------+------+------------------+---------------+--------------------------------+-------------------------------+------------+-------------+-------------------------+----------+----------------------+----------+-----------+-----------+---------+------------+---------+----------------------------+------------+-------------+----------+----------+---------------+---------------+---------------+----------------------+---------------------------------+-----------------------+-----------------------+----------+----------------------+----------+---------+-----------+---------------------------+---------+----------+---------------------------------+--------+--------+------------+---------+------------+------------+--------+--------+------------+------------+---------------+------------+---------------------+----------+------------------+-----------+--------------------+----------+---------------+---------------+----------+----------+-----------+-------------+---------+-----------+--------+-------+------+------+----------------------------+-------------+---------+
    |1-Story 1946 & Newer All Styles|Residential Low Density|   9600|Paved |Regular           |Near Flat/Level|All public Utilities (E,G,W,& S)|Frontage on 2 sides of property|Gentle slope|Veenker      |Adjacent to feeder street|Normal    |Single-family Detached|One story |          6|          8|     1976|        1976|Gable    |Standard (Composite) Shingle|Metal Siding|Metal Siding |None      |         0|Average/Typical|Average/Typical|Cinder Block   |Good (90-99 inches)   |Typical - slight dampness allowed|Good Exposure          |Average Living Quarters|       978|Unfinished/No Basement|         0|      284|       1262|Gas forced warm air furnace|Excellent|Yes       |Standard Circuit Breakers & Romex|    1262|       0|           0|     1262|           0|           1|       2|       0|           3|           1|Typical/Average|           6|Typical Functionality|         1|Attached to home  |       1976|Rough Finished      |       460|Typical/Average|Typical/Average|Paved     |       298|          0|            0|        0|          0|       0|      0|     5|  2007|Warranty Deed - Conventional|Normal Sale  |   181500|
    +-------------------------------+-----------------------+-------+------+------------------+---------------+--------------------------------+-------------------------------+------------+-------------+-------------------------+----------+----------------------+----------+-----------+-----------+---------+------------+---------+----------------------------+------------+-------------+----------+----------+---------------+---------------+---------------+----------------------+---------------------------------+-----------------------+-----------------------+----------+----------------------+----------+---------+-----------+---------------------------+---------+----------+---------------------------------+--------+--------+------------+---------+------------+------------+--------+--------+------------+------------+---------------+------------+---------------------+----------+------------------+-----------+--------------------+----------+---------------+---------------+----------+----------+-----------+-------------+---------+-----------+--------+-------+------+------+----------------------------+-------------+---------+
    |2-Story 1946 & Newer           |Residential Low Density|  11250|Paved |Slightly irregular|Near Flat/Level|All public Utilities (E,G,W,& S)|Inside lot                     |Gentle slope|College Creek|Normal                   |Normal    |Single-family Detached|Two story |          7|          5|     2001|        2002|Gable    |Standard (Composite) Shingle|Vinyl Siding|Vinyl Siding |Brick Face|       162|Good           |Average/Typical|Poured Contrete|Good (90-99 inches)   |Typical - slight dampness allowed|Mimimum Exposure       |Good Living Quarters   |       486|Unfinished/No Basement|         0|      434|        920|Gas forced warm air furnace|Excellent|Yes       |Standard Circuit Breakers & Romex|     920|     866|           0|     1786|           1|           0|       2|       1|           3|           1|Good           |           6|Typical Functionality|         1|Attached to home  |       2001|Rough Finished      |       608|Typical/Average|Typical/Average|Paved     |         0|         42|            0|        0|          0|       0|      0|     9|  2008|Warranty Deed - Conventional|Normal Sale  |   223500|
    +-------------------------------+-----------------------+-------+------+------------------+---------------+--------------------------------+-------------------------------+------------+-------------+-------------------------+----------+----------------------+----------+-----------+-----------+---------+------------+---------+----------------------------+------------+-------------+----------+----------+---------------+---------------+---------------+----------------------+---------------------------------+-----------------------+-----------------------+----------+----------------------+----------+---------+-----------+---------------------------+---------+----------+---------------------------------+--------+--------+------------+---------+------------+------------+--------+--------+------------+------------+---------------+------------+---------------------+----------+------------------+-----------+--------------------+----------+---------------+---------------+----------+----------+-----------+-------------+---------+-----------+--------+-------+------+------+----------------------------+-------------+---------+
    |2-Story 1945 & Older           |Residential Low Density|   9550|Paved |Slightly irregular|Near Flat/Level|All public Utilities (E,G,W,& S)|Corner lot                     |Gentle slope|Crawford     |Normal                   |Normal    |Single-family Detached|Two story |          7|          5|     1915|        1970|Gable    |Standard (Composite) Shingle|Wood Siding |Wood Shingles|None      |         0|Average/Typical|Average/Typical|Brick & Tile   |Typical (80-89 inches)|Good                             |No Exposure/No Basement|Average Living Quarters|       216|Unfinished/No Basement|         0|      540|        756|Gas forced warm air furnace|Good     |Yes       |Standard Circuit Breakers & Romex|     961|     756|           0|     1717|           1|           0|       1|       0|           3|           1|Good           |           7|Typical Functionality|         1|Detached from home|       1998|Unfinished/No Garage|       642|Typical/Average|Typical/Average|Paved     |         0|         35|          272|        0|          0|       0|      0|     2|  2006|Warranty Deed - Conventional|Abnormal Sale|   140000|
    +-------------------------------+-----------------------+-------+------+------------------+---------------+--------------------------------+-------------------------------+------------+-------------+-------------------------+----------+----------------------+----------+-----------+-----------+---------+------------+---------+----------------------------+------------+-------------+----------+----------+---------------+---------------+---------------+----------------------+---------------------------------+-----------------------+-----------------------+----------+----------------------+----------+---------+-----------+---------------------------+---------+----------+---------------------------------+--------+--------+------------+---------+------------+------------+--------+--------+------------+------------+---------------+------------+---------------------+----------+------------------+-----------+--------------------+----------+---------------+---------------+----------+----------+-----------+-------------+---------+-----------+--------+-------+------+------+----------------------------+-------------+---------+
    |2-Story 1946 & Newer           |Residential Low Density|  14260|Paved |Slightly irregular|Near Flat/Level|All public Utilities (E,G,W,& S)|Frontage on 2 sides of property|Gentle slope|Northridge   |Normal                   |Normal    |Single-family Detached|Two story |          8|          5|     2000|        2000|Gable    |Standard (Composite) Shingle|Vinyl Siding|Vinyl Siding |Brick Face|       350|Good           |Average/Typical|Poured Contrete|Good (90-99 inches)   |Typical - slight dampness allowed|Average Exposure       |Good Living Quarters   |       655|Unfinished/No Basement|         0|      490|       1145|Gas forced warm air furnace|Excellent|Yes       |Standard Circuit Breakers & Romex|    1145|    1053|           0|     2198|           1|           0|       2|       1|           4|           1|Good           |           9|Typical Functionality|         1|Attached to home  |       2000|Rough Finished      |       836|Typical/Average|Typical/Average|Paved     |       192|         84|            0|        0|          0|       0|      0|    12|  2008|Warranty Deed - Conventional|Normal Sale  |   250000|
    +-------------------------------+-----------------------+-------+------+------------------+---------------+--------------------------------+-------------------------------+------------+-------------+-------------------------+----------+----------------------+----------+-----------+-----------+---------+------------+---------+----------------------------+------------+-------------+----------+----------+---------------+---------------+---------------+----------------------+---------------------------------+-----------------------+-----------------------+----------+----------------------+----------+---------+-----------+---------------------------+---------+----------+---------------------------------+--------+--------+------------+---------+------------+------------+--------+--------+------------+------------+---------------+------------+---------------------+----------+------------------+-----------+--------------------+----------+---------------+---------------+----------+----------+-----------+-------------+---------+-----------+--------+-------+------+------+----------------------------+-------------+---------+


.. code:: ipython3

    y = house_df['SalePrice']
    X = house_df.drop('SalePrice', axis=1)

Encoding Categorical Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    categorical_features = [col for col in X.columns if X[col].dtype == 'object']
    
    encoder = OrdinalEncoder(
        cols=categorical_features,
        handle_unknown='ignore',
        return_df=True
    ).fit(X)
    
    X = encoder.transform(X)

Train / Test Split
~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=1)

Model fitting
~~~~~~~~~~~~~

.. code:: ipython3

    regressor = LGBMRegressor(n_estimators=200).fit(X_train, y_train)

Construct groups of features
----------------------------

There are quite a lot of features used by the model and it can be hard
to compare them.

**We can regroup the features that share similarities in order to
identify which topic is important.**

In our example we constructed the following new groups :

- ``location``: features related to the location of the house
- ``size``: features that measure part of the house
- ``aspect``: features that evaluate the style of any part of the house
- ``condition``: features related to the quality of anything in the house
- ``configuration``: features about the general configuration / shape of the house
- ``equipment``: features that describe the equipment of the house (electricity, gas, heating…)
- ``garage``: features related to the garage (style, …)
- ``sale``: features related to the sale of the house

.. code:: ipython3

    # We construct the groups as a dictionary of string keys and list of string values
    # All the features inside the list will belong to the same group
    features_groups = {
        "location": ["MSZoning", "Neighborhood", "Condition1", "Condition2"],
        "size": [
            "LotArea",
            "MasVnrArea",
            "BsmtQual",
            "BsmtFinSF2",
            "BsmtUnfSF",
            "TotalBsmtSF",
            "1stFlrSF",
            "2ndFlrSF",
            "GrLivArea",
            "WoodDeckSF",
            "OpenPorchSF",
            "EnclosedPorch",
            "3SsnPorch",
            "ScreenPorch",
            "PoolArea",
            "BsmtFinSF1"
        ],
        "aspect": [
            "LotShape",
            "LandContour",
            "RoofStyle",
            "RoofMatl",
            "Exterior1st",
            "MasVnrType",
        ],
        "condition": [
            "OverallQual",
            "OverallCond",
            "ExterQual",
            "ExterCond",
            "BsmtCond",
            "BsmtFinType1",
            "BsmtFinType2",
            "HeatingQC",
            "KitchenQual"
        ],
        "configuration": ["LotConfig", "LandSlope", "BldgType", "HouseStyle"],
        "equipment": ["Heating", "CentralAir", "Electrical"],
        "garage": [
            "GarageType",
            "GarageYrBlt",
            "GarageFinish",
            "GarageArea",
            "GarageQual",
            "GarageCond",
        ],
        "sale": ["SaleType", "SaleCondition", "MoSold", "YrSold"]
    }

**Optional : we can also give labels to groups names**

.. code:: ipython3

    groups_labels = {
        'location': 'Location of the property',
        'size' : 'Size of different elements in the house',
        'aspect': 'Aspect of the house',
        'condition': 'Quality of the materials and parts of the property',
        'configuration': 'Configuration of the house',
        'equipment': 'All equipments',
        'garage': 'Garage group of features',
        'sale': 'Sale information'
    }
    house_dict.update(groups_labels)

Compile Shapash SmartExplainer object using groups
--------------------------------------------------

.. code:: ipython3

    from shapash.explainer.smart_explainer import SmartExplainer
    # optional parameter, specifies label for features and groups name
    xpl = SmartExplainer(features_dict=house_dict)  

.. code:: ipython3

    xpl.compile(
        x=X_test,
        model=regressor,
        preprocessing=encoder,
        features_groups=features_groups
    )


Start WebApp
------------

We can now start the webapp using the following cell.

| The groups of features are visible by default on the features
  importance plot.
| You can disable the groups using the ``groups`` switch button.

Also you can click on a group’s bar to display the features importance
of the features inside the group.

.. code:: ipython3

    app = xpl.run_app(title_story='House Prices')

**Stop the WebApp after using it**

.. code:: ipython3

    app.kill()

Explore the functions of Shapash using groups
---------------------------------------------

Features importance plot
~~~~~~~~~~~~~~~~~~~~~~~~

**Display the features importance plot that includes the groups and
excludes the features inside each group**

.. code:: ipython3

    xpl.plot.features_importance(selection=[259, 268])



.. image:: tuto-common01-groups_of_features_files/tuto-common01-groups_of_features_30_0.png


**Display the features importance plot of the features inside one
group**

.. code:: ipython3

    xpl.plot.features_importance(group_name='size')



.. image:: tuto-common01-groups_of_features_files/tuto-common01-groups_of_features_32_0.png


Contribution plot
~~~~~~~~~~~~~~~~~

| **Plot the shap values of each observation of a group of features**
| The features values were projected on the x axis using t-SNE.
| The values of the features (top 4 features only) can be visualized
  using the hover text.

.. code:: ipython3

    xpl.plot.contribution_plot('size')



.. image:: tuto-common01-groups_of_features_files/tuto-common01-groups_of_features_35_0.png


Local plot
~~~~~~~~~~

By default, Shapash will display the groups in the local plot.

You can directly see the impact of the different groups of features for
the given observation.

.. code:: ipython3

    xpl.filter(max_contrib=8)

.. code:: ipython3

    xpl.plot.local_plot(index=629)



.. image:: tuto-common01-groups_of_features_files/tuto-common01-groups_of_features_39_0.png


You can also display the features without the groups using the following
parameters :

.. code:: ipython3

    xpl.filter(max_contrib=6, display_groups=False)

.. code:: ipython3

    xpl.plot.local_plot(index=629, display_groups=False)



.. image:: tuto-common01-groups_of_features_files/tuto-common01-groups_of_features_42_0.png

