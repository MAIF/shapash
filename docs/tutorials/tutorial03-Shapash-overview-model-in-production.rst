Shapash in Jupyter - Overview
=============================

With this tutorial you: Understand how Shapash works in Jupyter Notebook
with a simple use case

Contents: - Build a Regressor - Compile Shapash SmartExplainer - Compile
Shapash SmartExplainer to SmartPredictor - Save Shapash Smartpredictor
Object in pickle file - Make a prediction

Data from Kaggle `House
Prices <https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data>`__

.. code:: ipython3

    import pandas as pd
    from category_encoders import OrdinalEncoder
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import train_test_split

Building Supervized Model
-------------------------

.. code:: ipython3

    import sys
    sys.path.insert(0,'/home/78257d/shapash/')
    from shapash.explainer.smart_predictor import SmartPredictor
    from shapash.explainer.smart_explainer import SmartExplainer
    from shapash.data.data_loader import data_loading
    from shapash.utils.load_smartpredictor import load_smartpredictor
    #from shapash.data.data_loader import data_loading
    house_df, house_dict = data_loading('house_prices')

.. code:: ipython3

    y_df=house_df['SalePrice'].to_frame()
    X_df=house_df[house_df.columns.difference(['SalePrice'])]

.. code:: ipython3

    house_df.head()




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
          <th>MSSubClass</th>
          <th>MSZoning</th>
          <th>LotArea</th>
          <th>Street</th>
          <th>LotShape</th>
          <th>LandContour</th>
          <th>Utilities</th>
          <th>LotConfig</th>
          <th>LandSlope</th>
          <th>Neighborhood</th>
          <th>...</th>
          <th>EnclosedPorch</th>
          <th>3SsnPorch</th>
          <th>ScreenPorch</th>
          <th>PoolArea</th>
          <th>MiscVal</th>
          <th>MoSold</th>
          <th>YrSold</th>
          <th>SaleType</th>
          <th>SaleCondition</th>
          <th>SalePrice</th>
        </tr>
        <tr>
          <th>Id</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
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
          <td>2-Story 1946 &amp; Newer</td>
          <td>Residential Low Density</td>
          <td>8450</td>
          <td>Paved</td>
          <td>Regular</td>
          <td>Near Flat/Level</td>
          <td>All public Utilities (E,G,W,&amp; S)</td>
          <td>Inside lot</td>
          <td>Gentle slope</td>
          <td>College Creek</td>
          <td>...</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>2</td>
          <td>2008</td>
          <td>Warranty Deed - Conventional</td>
          <td>Normal Sale</td>
          <td>208500</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1-Story 1946 &amp; Newer All Styles</td>
          <td>Residential Low Density</td>
          <td>9600</td>
          <td>Paved</td>
          <td>Regular</td>
          <td>Near Flat/Level</td>
          <td>All public Utilities (E,G,W,&amp; S)</td>
          <td>Frontage on 2 sides of property</td>
          <td>Gentle slope</td>
          <td>Veenker</td>
          <td>...</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>5</td>
          <td>2007</td>
          <td>Warranty Deed - Conventional</td>
          <td>Normal Sale</td>
          <td>181500</td>
        </tr>
        <tr>
          <th>3</th>
          <td>2-Story 1946 &amp; Newer</td>
          <td>Residential Low Density</td>
          <td>11250</td>
          <td>Paved</td>
          <td>Slightly irregular</td>
          <td>Near Flat/Level</td>
          <td>All public Utilities (E,G,W,&amp; S)</td>
          <td>Inside lot</td>
          <td>Gentle slope</td>
          <td>College Creek</td>
          <td>...</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>9</td>
          <td>2008</td>
          <td>Warranty Deed - Conventional</td>
          <td>Normal Sale</td>
          <td>223500</td>
        </tr>
        <tr>
          <th>4</th>
          <td>2-Story 1945 &amp; Older</td>
          <td>Residential Low Density</td>
          <td>9550</td>
          <td>Paved</td>
          <td>Slightly irregular</td>
          <td>Near Flat/Level</td>
          <td>All public Utilities (E,G,W,&amp; S)</td>
          <td>Corner lot</td>
          <td>Gentle slope</td>
          <td>Crawford</td>
          <td>...</td>
          <td>272</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>2</td>
          <td>2006</td>
          <td>Warranty Deed - Conventional</td>
          <td>Abnormal Sale</td>
          <td>140000</td>
        </tr>
        <tr>
          <th>5</th>
          <td>2-Story 1946 &amp; Newer</td>
          <td>Residential Low Density</td>
          <td>14260</td>
          <td>Paved</td>
          <td>Slightly irregular</td>
          <td>Near Flat/Level</td>
          <td>All public Utilities (E,G,W,&amp; S)</td>
          <td>Frontage on 2 sides of property</td>
          <td>Gentle slope</td>
          <td>Northridge</td>
          <td>...</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>12</td>
          <td>2008</td>
          <td>Warranty Deed - Conventional</td>
          <td>Normal Sale</td>
          <td>250000</td>
        </tr>
      </tbody>
    </table>
    <p>5 rows × 73 columns</p>
    </div>



Encoding Categorical Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    from category_encoders import OrdinalEncoder
    
    categorical_features = [col for col in X_df.columns if X_df[col].dtype == 'object']
    
    encoder = OrdinalEncoder(
        cols=categorical_features,
        handle_unknown='ignore',
        return_df=True).fit(X_df)
    
    X_df=encoder.transform(X_df)


.. parsed-literal::

    /home/78257d/.conda/envs/test_env_shapash/lib/python3.6/site-packages/category_encoders/utils.py:21: FutureWarning:
    
    is_categorical is deprecated and will be removed in a future version.  Use is_categorical_dtype instead
    


Train / Test Split
^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    Xtrain, Xtest, ytrain, ytest = train_test_split(X_df, y_df, train_size=0.75, random_state=1)

Model Fitting
^^^^^^^^^^^^^

.. code:: ipython3

    regressor = LGBMRegressor(n_estimators=200).fit(Xtrain,ytrain)

.. code:: ipython3

    y_pred = pd.DataFrame(regressor.predict(Xtest),columns=['pred'],index=Xtest.index)

Understand my model with shapash
--------------------------------

Declare and Compile SmartExplainer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    from shapash.explainer.smart_explainer import SmartExplainer

.. code:: ipython3

    xpl = SmartExplainer(features_dict=house_dict) # Optional parameter, dict specifies label for features name 

.. code:: ipython3

    xpl.compile(
        x=Xtest,
        model=regressor,
        preprocessing=encoder, # Optional: compile step can use inverse_transform method
        y_pred=y_pred # Optional
    )


.. parsed-literal::

    Backend: Shap TreeExplainer


Compile SmartExplainer to SmartPredictor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    predictor = xpl.to_smartpredictor()

Save and Load your Predictor
----------------------------

Save your predictor in Pickle File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    predictor.save('./predictor.pkl')

Load your predictor in Pickle File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    predictor_load = load_smartpredictor('./predictor.pkl')

Make a prediction with your Predictor
-------------------------------------

Add data
^^^^^^^^

.. code:: ipython3

    predictor_load.add_input(x=X_df, ypred=y_df)

Make prediction
^^^^^^^^^^^^^^^

.. code:: ipython3

    prediction = predictor_load.predict()

.. code:: ipython3

    prediction.head()




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
          <th>ypred</th>
        </tr>
        <tr>
          <th>Id</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>1</th>
          <td>206462.878757</td>
        </tr>
        <tr>
          <th>2</th>
          <td>181127.963794</td>
        </tr>
        <tr>
          <th>3</th>
          <td>221478.052244</td>
        </tr>
        <tr>
          <th>4</th>
          <td>184788.423141</td>
        </tr>
        <tr>
          <th>5</th>
          <td>256637.518234</td>
        </tr>
      </tbody>
    </table>
    </div>



Get detailed explanability associated to the prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    detailed_contributions = predictor_load.detail_contributions()

.. code:: ipython3

    detailed_contributions.head()

Summarize explainability of the predictions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    predictor_load.modify_mask(max_contrib=10)

.. code:: ipython3

    explanation = predictor_load.summarize()

.. code:: ipython3

    explanation.head()




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
          <th>ypred</th>
          <th>feature_1</th>
          <th>value_1</th>
          <th>contribution_1</th>
          <th>feature_2</th>
          <th>value_2</th>
          <th>contribution_2</th>
          <th>feature_3</th>
          <th>value_3</th>
          <th>contribution_3</th>
          <th>...</th>
          <th>contribution_30</th>
          <th>feature_31</th>
          <th>value_31</th>
          <th>contribution_31</th>
          <th>feature_32</th>
          <th>value_32</th>
          <th>contribution_32</th>
          <th>feature_33</th>
          <th>value_33</th>
          <th>contribution_33</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>1</th>
          <td>206462.878757</td>
          <td>Overall material and finish of the house</td>
          <td>7</td>
          <td>8248.82</td>
          <td>Total square feet of basement area</td>
          <td>856</td>
          <td>-5165.5</td>
          <td>Original construction date</td>
          <td>2003</td>
          <td>3870.96</td>
          <td>...</td>
          <td>334.984</td>
          <td>Garage quality</td>
          <td>0</td>
          <td>304.462</td>
          <td>Half baths above grade</td>
          <td>0</td>
          <td>286.121</td>
          <td>Lot configuration</td>
          <td>0</td>
          <td>-276.762</td>
        </tr>
        <tr>
          <th>2</th>
          <td>181127.963794</td>
          <td>Overall material and finish of the house</td>
          <td>6</td>
          <td>-14555.9</td>
          <td>Ground living area square feet</td>
          <td>1262</td>
          <td>-10016.3</td>
          <td>Overall condition of the house</td>
          <td>8</td>
          <td>6899.3</td>
          <td>...</td>
          <td>343.581</td>
          <td>Exterior covering on house</td>
          <td>0</td>
          <td>340.65</td>
          <td>Original construction date</td>
          <td>0</td>
          <td>340.16</td>
          <td>Interior finish of the garage?</td>
          <td>0</td>
          <td>335.442</td>
        </tr>
        <tr>
          <th>3</th>
          <td>221478.052244</td>
          <td>Ground living area square feet</td>
          <td>1786</td>
          <td>15708.3</td>
          <td>Overall material and finish of the house</td>
          <td>7</td>
          <td>11084.5</td>
          <td>Size of garage in square feet</td>
          <td>608</td>
          <td>5998.61</td>
          <td>...</td>
          <td>-323.291</td>
          <td>Masonry veneer area in square feet</td>
          <td>0</td>
          <td>-295.708</td>
          <td>Garage quality</td>
          <td>0</td>
          <td>290.116</td>
          <td>Physical locations within Ames city limits</td>
          <td>0</td>
          <td>260.384</td>
        </tr>
        <tr>
          <th>4</th>
          <td>184788.423141</td>
          <td>Overall material and finish of the house</td>
          <td>7</td>
          <td>8188.35</td>
          <td>Size of garage in square feet</td>
          <td>642</td>
          <td>6651.57</td>
          <td>Total square feet of basement area</td>
          <td>756</td>
          <td>-5882.2</td>
          <td>...</td>
          <td>345.697</td>
          <td>Screen porch area in square feet</td>
          <td>0</td>
          <td>-344.762</td>
          <td>Year garage was built</td>
          <td>0</td>
          <td>315.665</td>
          <td>Bedrooms above grade</td>
          <td>0</td>
          <td>310.41</td>
        </tr>
        <tr>
          <th>5</th>
          <td>256637.518234</td>
          <td>Overall material and finish of the house</td>
          <td>8</td>
          <td>58568.4</td>
          <td>Ground living area square feet</td>
          <td>2198</td>
          <td>16891.9</td>
          <td>Size of garage in square feet</td>
          <td>836</td>
          <td>15161.9</td>
          <td>...</td>
          <td>-361.637</td>
          <td>Full bathrooms above grade</td>
          <td>0</td>
          <td>-309.068</td>
          <td>Wood deck area in square feet</td>
          <td>0</td>
          <td>270.882</td>
          <td>Masonry veneer type</td>
          <td>0</td>
          <td>266.871</td>
        </tr>
      </tbody>
    </table>
    <p>5 rows × 100 columns</p>
    </div>



