How to use filter and local_plot methods
========================================

This tutorial presents the different parameters you can use to summarize
and display local explanations. It also shows how to export this summary
into pandas DataFrame

Contents: - Work with filter and local_plot method to tune output -
display Positive or Negative contributions - mask hidden contrib or
prediction - hide some specific features - Use query parameter to select
without index or row number - Classification: How can you select the
label value to display? - print the summary params - export local
explanation with to_pandas

Data from Kaggle `House
Prices <https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data>`__

.. code:: ipython3

    import pandas as pd
    from category_encoders import OrdinalEncoder
    from catboost import CatBoostRegressor, CatBoostClassifier
    from sklearn.model_selection import train_test_split

Building Supervized Model
-------------------------

.. code:: ipython3

    from shapash.data.data_loader import data_loading
    house_df, house_dict = data_loading('house_prices')
    y_df=house_df['SalePrice'].to_frame()
    X_df=house_df[house_df.columns.difference(['SalePrice'])]

.. code:: ipython3

    from category_encoders import OrdinalEncoder
    
    categorical_features = [col for col in X_df.columns if X_df[col].dtype == 'object']
    
    encoder = OrdinalEncoder(
        cols=categorical_features,
        handle_unknown='ignore',
        return_df=True).fit(X_df)
    
    X_df=encoder.transform(X_df)

.. code:: ipython3

    Xtrain, Xtest, ytrain, ytest = train_test_split(X_df, y_df, train_size=0.75, random_state=1)

.. code:: ipython3

    regressor = CatBoostRegressor(n_estimators=50).fit(Xtrain,ytrain,verbose=False)

.. code:: ipython3

    y_pred = pd.DataFrame(regressor.predict(Xtest),columns=['pred'],index=Xtest.index)

Work With filter and local_plot methods
---------------------------------------

First step: You need to Declare and Compile SmartExplainer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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


Filter method
^^^^^^^^^^^^^

Use the filter method to specify how to synthesize local explainability
you have 4 parameters to customize your summary: - max_contrib : maximum
number of criteria to display - threshold : minimum value of the
contribution (in absolute value) necessary to display a criterion -
positive : display only positive contribution? Negative?(default None) -
features_to_hide : list of features you don’t want to display

.. code:: ipython3

    xpl.filter(max_contrib=5)

Local_plot
^^^^^^^^^^

.. code:: ipython3

    xpl.plot.local_plot(index=268)



.. image:: tuto-plot01-local_plot-and-to_pandas_files/tuto-plot01-local_plot-and-to_pandas_16_0.png


Threshold parameter to focus on significant contributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    xpl.filter(max_contrib=5,threshold=10000)
    xpl.plot.local_plot(index=268)



.. image:: tuto-plot01-local_plot-and-to_pandas_files/tuto-plot01-local_plot-and-to_pandas_18_0.png


Don’t display hidden contributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    xpl.plot.local_plot(index=268,show_masked=False)



.. image:: tuto-plot01-local_plot-and-to_pandas_files/tuto-plot01-local_plot-and-to_pandas_20_0.png


You can also hide the predict value with parameter show_predict=False

Focus on Negative contribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    xpl.filter(max_contrib=8,positive=False)
    xpl.plot.local_plot(index=268)



.. image:: tuto-plot01-local_plot-and-to_pandas_files/tuto-plot01-local_plot-and-to_pandas_23_0.png


You can also focus positive contribution using positive=True

Hide specific features:
^^^^^^^^^^^^^^^^^^^^^^^

Because: - some features can be too complex - end user don’t want know
unnecessary information

You can use features_to_hide parameter in filter method

.. code:: ipython3

    xpl.filter(max_contrib=8,positive=False,features_to_hide=['BsmtFullBath','GarageType'])
    xpl.plot.local_plot(index=268)



.. image:: tuto-plot01-local_plot-and-to_pandas_files/tuto-plot01-local_plot-and-to_pandas_26_0.png


Select a row with a query
^^^^^^^^^^^^^^^^^^^^^^^^^

You can selct with an index or a row number. You can also use a query:

.. code:: ipython3

    xpl.filter(max_contrib=3,positive=False)
    xpl.plot.local_plot(query="LotArea == 8400 and LotShape == 'Regular' and TotalBsmtSF == 720")



.. image:: tuto-plot01-local_plot-and-to_pandas_files/tuto-plot01-local_plot-and-to_pandas_28_0.png


Classification Case
-------------------

transform our use case into classification:

.. code:: ipython3

    ytrain['PriceClass'] = ytrain['SalePrice'].apply(lambda x: 1 if x < 150000 else (3 if x > 300000 else 2))
    label_dict = { 1 : 'Cheap', 2 : 'Moderately Expensive', 3 : 'Expensive' }

.. code:: ipython3

    clf = CatBoostClassifier(n_estimators=50).fit(Xtrain,ytrain['PriceClass'],verbose=False)
    y_pred_clf = pd.DataFrame(clf.predict(Xtest),columns=['pred'],index=Xtest.index)

Declare new SmartExplainer dedicated to classification problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    xplclf = SmartExplainer(features_dict=house_dict,label_dict=label_dict) # Optional parameters: display explicit output

.. code:: ipython3

    xplclf.compile(
        x=Xtest,
        model=clf,
        preprocessing=encoder,
        y_pred=y_pred_clf
    )


.. parsed-literal::

    Backend: Shap TreeExplainer


Use label parameter of local_plot parameter to select the explanation you want
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

with label parameter, you can specify explicit label or label number

.. code:: ipython3

    xplclf.filter(max_contrib=7,positive=True)
    xplclf.plot.local_plot(index=268,label='Moderately Expensive')



.. image:: tuto-plot01-local_plot-and-to_pandas_files/tuto-plot01-local_plot-and-to_pandas_36_0.png


See the summary parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    xplclf.mask_params




.. parsed-literal::

    {'features_to_hide': None,
     'threshold': None,
     'positive': True,
     'max_contrib': 7}



Export explanations
-------------------

Export your local explanation in pd.DataFrame with to_pandas method :
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  The to_pandas method has the same parameters as the filter method
-  if you don’t specify any parameter, to_pandas use the same params you
   specified when you call filter method
-  When you work on classification problem, parameter proba=True output
   predict probability

.. code:: ipython3

    summary_df= xplclf.to_pandas(proba=True)


.. parsed-literal::

    to_pandas params: {'features_to_hide': None, 'threshold': None, 'positive': True, 'max_contrib': 7}


.. code:: ipython3

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
          <th>...</th>
          <th>contribution_4</th>
          <th>feature_5</th>
          <th>value_5</th>
          <th>contribution_5</th>
          <th>feature_6</th>
          <th>value_6</th>
          <th>contribution_6</th>
          <th>feature_7</th>
          <th>value_7</th>
          <th>contribution_7</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>259</th>
          <td>Moderately Expensive</td>
          <td>0.994917</td>
          <td>Ground living area square feet</td>
          <td>1792</td>
          <td>0.309308</td>
          <td>Interior finish of the garage?</td>
          <td>Rough Finished</td>
          <td>0.275467</td>
          <td>Size of garage in square feet</td>
          <td>564</td>
          <td>...</td>
          <td>0.182722</td>
          <td>Physical locations within Ames city limits</td>
          <td>College Creek</td>
          <td>0.170888</td>
          <td>Overall material and finish of the house</td>
          <td>7</td>
          <td>0.164045</td>
          <td>Height of the basement</td>
          <td>Good (90-99 inches)</td>
          <td>0.139618</td>
        </tr>
        <tr>
          <th>268</th>
          <td>Moderately Expensive</td>
          <td>0.876916</td>
          <td>Second floor square feet</td>
          <td>720</td>
          <td>0.183251</td>
          <td>Full bathrooms above grade</td>
          <td>2</td>
          <td>0.155086</td>
          <td>Ground living area square feet</td>
          <td>2192</td>
          <td>...</td>
          <td>0.143119</td>
          <td>Type 1 finished square feet</td>
          <td>378</td>
          <td>0.142439</td>
          <td>First Floor square feet</td>
          <td>1052</td>
          <td>0.127817</td>
          <td>Half baths above grade</td>
          <td>1</td>
          <td>0.127717</td>
        </tr>
        <tr>
          <th>289</th>
          <td>Cheap</td>
          <td>0.997304</td>
          <td>Ground living area square feet</td>
          <td>900</td>
          <td>0.818922</td>
          <td>Size of garage in square feet</td>
          <td>280</td>
          <td>0.561631</td>
          <td>Total square feet of basement area</td>
          <td>882</td>
          <td>...</td>
          <td>0.349033</td>
          <td>Full bathrooms above grade</td>
          <td>1</td>
          <td>0.324806</td>
          <td>Overall material and finish of the house</td>
          <td>5</td>
          <td>0.318031</td>
          <td>First Floor square feet</td>
          <td>900</td>
          <td>0.247826</td>
        </tr>
        <tr>
          <th>650</th>
          <td>Cheap</td>
          <td>0.998653</td>
          <td>Ground living area square feet</td>
          <td>630</td>
          <td>0.816398</td>
          <td>Size of garage in square feet</td>
          <td>0</td>
          <td>0.587745</td>
          <td>Total square feet of basement area</td>
          <td>630</td>
          <td>...</td>
          <td>0.355685</td>
          <td>Overall material and finish of the house</td>
          <td>4</td>
          <td>0.317549</td>
          <td>Full bathrooms above grade</td>
          <td>1</td>
          <td>0.31303</td>
          <td>General zoning classification</td>
          <td>Residential Medium Density</td>
          <td>0.178395</td>
        </tr>
        <tr>
          <th>1234</th>
          <td>Cheap</td>
          <td>0.852389</td>
          <td>Ground living area square feet</td>
          <td>1188</td>
          <td>0.942118</td>
          <td>Remodel date</td>
          <td>1959</td>
          <td>0.423368</td>
          <td>Overall material and finish of the house</td>
          <td>5</td>
          <td>...</td>
          <td>0.373812</td>
          <td>Number of fireplaces</td>
          <td>0</td>
          <td>0.168725</td>
          <td>Rating of basement finished area</td>
          <td>Average Rec Room</td>
          <td>0.130175</td>
          <td>Wood deck area in square feet</td>
          <td>0</td>
          <td>0.12249</td>
        </tr>
      </tbody>
    </table>
    <p>5 rows × 23 columns</p>
    </div>



It is also possible to calculate the probability relating to one of the
target modality for all the dataset, and to display the elements of
explainability associated with this target modality

.. code:: ipython3

    #Create One column pd.DataFrame with constant value
    constantpred=pd.DataFrame([3 for x in range(Xtest.shape[0])],columns=['pred'],index=Xtest.index)
    xplclf.add(y_pred=constantpred)
    summary_df = xplclf.to_pandas(proba=True,max_contrib=3,threshold=0.1,positive=True)

.. code:: ipython3

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
          <th>259</th>
          <td>Expensive</td>
          <td>0.003081</td>
          <td>Ground living area square feet</td>
          <td>1792</td>
          <td>0.327986</td>
          <td>Overall material and finish of the house</td>
          <td>7</td>
          <td>0.197494</td>
          <td>Rating of basement finished area</td>
          <td>Good Living Quarters</td>
          <td>0.181953</td>
        </tr>
        <tr>
          <th>268</th>
          <td>Expensive</td>
          <td>0.007627</td>
          <td>Ground living area square feet</td>
          <td>2192</td>
          <td>0.825571</td>
          <td>Wood deck area in square feet</td>
          <td>262</td>
          <td>0.251474</td>
          <td>Remodel date</td>
          <td>1997</td>
          <td>0.157067</td>
        </tr>
        <tr>
          <th>289</th>
          <td>Expensive</td>
          <td>0.000024</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>650</th>
          <td>Expensive</td>
          <td>0.000056</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1234</th>
          <td>Expensive</td>
          <td>0.000623</td>
          <td>Type of sale</td>
          <td>Court Officer Deed/Estate</td>
          <td>0.114506</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
    </div>



NB: The to_pandas method returns Nan for lines that do not meet your
conditions
