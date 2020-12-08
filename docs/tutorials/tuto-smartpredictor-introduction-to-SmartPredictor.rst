From model training to deployment - an introduction to the SmartPredictor object
================================================================================

Shapash provides a SmartPredictor Object to make prediction and local
explainability for operational needs in deployment context. It gives a
summary of the local explanation of your prediction. SmartPredictor
allows users to configure the summary to suit their use. It is an object
dedicated to deployment, lighter than SmartExplainer Object with
additionnal consistency checks. SmartPredictor can be used with an API
or in batch mode.

This tutorial provides more information to help you getting started with
the SmartPredictor Object of Shapash.

Contents: - Build a SmartPredictor - Save and Load a Smartpredictor -
Add input - Use label and wording - Summarize explaination

We used Kaggle’s `Titanic <https://www.kaggle.com/c/titanic>`__ dataset

Step 1: Exploration and training of the model
---------------------------------------------

Import Dataset
~~~~~~~~~~~~~~

First, we need to import a dataset. Here we chose the famous dataset
Titanic from Kaggle.

.. code:: ipython3

    import numpy as np
    import pandas as pd
    from category_encoders import OrdinalEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import shap

.. code:: ipython3

    from shapash.explainer.smart_predictor import SmartPredictor
    from shapash.utils.load_smartpredictor import load_smartpredictor
    from shapash.data.data_loader import data_loading

.. code:: ipython3

    titan_df, titan_dict = data_loading('titanic')
    del titan_df['Name']

.. code:: ipython3

    titan_df.head()


.. parsed-literal::

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


Create Classification Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this section, we train a Machine Learning supervized model with our
data. In our example, we are confronted to a classification problem.

.. code:: ipython3

    y = titan_df['Survived']
    X = titan_df.drop('Survived', axis = 1)

.. code:: ipython3

    varcat=['Pclass', 'Sex', 'Embarked', 'Title']

Encoding Categorical Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We use a preprocessing on our data for handling categorical features
before the training step.

.. code:: ipython3

    categ_encoding = OrdinalEncoder(cols = varcat, \
                                    handle_unknown = 'ignore', \
                                    return_df = True).fit(X)
    X = categ_encoding.transform(X)

Train Test split + Random Forest fit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size = 0.75, random_state = 1)
    
    rf = RandomForestClassifier(n_estimators = 100, min_samples_leaf = 3)
    rf.fit(Xtrain, ytrain)




.. parsed-literal::

    RandomForestClassifier(min_samples_leaf=3)



.. code:: ipython3

    ypred=pd.DataFrame(rf.predict(Xtest), columns = ['pred'], index = Xtest.index)

Explore your trained model results Step with SmartExplainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    from shapash.explainer.smart_explainer import SmartExplainer

Use Label and Wording
^^^^^^^^^^^^^^^^^^^^^

Here, we use labels and wording to get a more understandable
explainabily. - features_dict : allow users to rename the features of
their datasets with the one needed - label_dict : allow users in
classification problems to rename label predicted with the one needed -
postprocessing : allow users to apply some wording to the features
wanted

.. code:: ipython3

    feature_dict = {
                    'Pclass': 'Ticket class',
                     'Sex': 'Sex',
                     'Age': 'Age',
                     'SibSp': 'Relatives such as brother or wife',
                     'Parch': 'Relatives like children or parents',
                     'Fare': 'Passenger fare',
                     'Embarked': 'Port of embarkation',
                     'Title': 'Title of passenger'
                   }

.. code:: ipython3

    label_dict = {0: "Not Survived", 1: "Survived"}

.. code:: ipython3

    postprocessing = {"Pclass": {'type': 'transcoding', 'rule': { 'First class': '1st class', 'Second class': '2nd class', "Third class": "3rd class"}}}

Define a SmartExplainer
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    xpl = SmartExplainer(label_dict = label_dict, features_dict = feature_dict)

We use the compile method of the SmartExplainer Object. This method is
the first step to understand model and prediction. It performs the
sorting of contributions, the reverse preprocessing steps and performs
all the calculations necessary for a quick display of plots and
efficient display of summary of explanation. (see the documentation on
SmartExplainer Object and the associated tutorials for more details)

.. code:: ipython3

    xpl.compile(
                x = Xtest,
                model = rf,
                preprocessing = categ_encoding,
                y_pred = ypred,
                postprocessing = postprocessing
               )


.. parsed-literal::

    Backend: Shap TreeExplainer


Understand results of your trained model with SmartExplainer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can easily get a first summary of the explanation of the model
results. - We choose to get the 3 most contributive features for each
prediction. - We use a wording to get features names more understandable
in operationnal case. - We rename the predicted label to show a more
explicit prediction. - We apply a post-processing to transform some
feature’s values.

.. code:: ipython3

    xpl.to_pandas(max_contrib = 3).head()


.. parsed-literal::

    .. table:: 
    
        +------------+------------------+-------+--------------+------------------+---------+--------------+-------------------+----------+--------------+
        |    pred    |    feature_1     |value_1|contribution_1|    feature_2     | value_2 |contribution_2|     feature_3     | value_3  |contribution_3|
        +============+==================+=======+==============+==================+=========+==============+===================+==========+==============+
        |Survived    |Sex               |female |       0.21521|Title of passenger|Mrs      |       0.17187|Ticket class       |1st class |       0.12316|
        +------------+------------------+-------+--------------+------------------+---------+--------------+-------------------+----------+--------------+
        |Not Survived|Sex               |male   |       0.08676|Title of passenger|Mr       |       0.08308|Passenger fare     |       7.9|       0.06384|
        +------------+------------------+-------+--------------+------------------+---------+--------------+-------------------+----------+--------------+
        |Survived    |Sex               |female |       0.19852|Title of passenger|Miss     |       0.18208|Ticket class       |2nd class |       0.10237|
        +------------+------------------+-------+--------------+------------------+---------+--------------+-------------------+----------+--------------+
        |Survived    |Sex               |female |       0.17773|Title of passenger|Miss     |       0.16248|Port of embarkation|Queenstown|       0.10887|
        +------------+------------------+-------+--------------+------------------+---------+--------------+-------------------+----------+--------------+
        |Survived    |Title of passenger|Miss   |       0.18741|Ticket class      |2nd class|       0.13504|Sex                |female    |       0.10926|
        +------------+------------------+-------+--------------+------------------+---------+--------------+-------------------+----------+--------------+


Step 2: SmartPredictor in production
------------------------------------

-  to_smartpredictor () is a method to get a SmartPredictor object.
-  It allows users to switch from a SmartExplainer used for data mining
   to the SmartPredictor.
-  It only keeps the attributes needed for deployment to be lighter than
   the SmartExplainer object.
-  Smartpredictor performs additional consistency checks before
   deployment.
-  This object is dedicated to the deployment.
-  In this section, we learn how to initialize a SmartPredictor.
-  It makes new predictions and summarize explainability that you
   configured to make it operational to your needs.
-  SmartPredictor can be use with API or in batch mode.
-  It handles dataframes and dictionnaries input data.

Switch from SmartExplainer Object to SmartPredictor Object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    predictor = xpl.to_smartpredictor()

Save your predictor in Pickle File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    predictor.save('./predictor.pkl')

Load your predictor in Pickle File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    predictor_load = load_smartpredictor('./predictor.pkl')

Make a prediction with your SmartPredictor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Once our SmartPredictor has been initialized, we can compute new
   predictions and explain them.
-  First, we specify a new dataset which can be a pandas.DataFrame or a
   dictionnary (usefull when you decide to use an API in your deployment
   process)
-  We use the add_input method of the SmartPredictor. (see the
   documentation for this method)

Add data
^^^^^^^^

.. code:: ipython3

    person_x = {'Pclass': 'First class',
                 'Sex': 'female',
                 'Age': 36,
                 'SibSp': 1,
                 'Parch': 0,
                 'Fare': 7.25,
                 'Embarked': 'Cherbourg',
                 'Title': 'Miss'
               }

.. code:: ipython3

    predictor_load.add_input(x = person_x)

If you don’t specify an ypred in the add_input method, SmartPredictor
use its predict method to automatically affect the predicted value to
ypred.

Make prediction
^^^^^^^^^^^^^^^

Let’s display ypred which has been automatically computed in add_input
method.

.. code:: ipython3

    predictor_load.data["ypred"]


.. parsed-literal::

    .. table:: 
    
        +--------+------+
        | ypred  |proba |
        +========+======+
        |Survived|0.7475|
        +--------+------+


The predict_proba method of Smartpredictor compute the probabilties
associated to each label.

.. code:: ipython3

    prediction_proba = predictor_load.predict_proba()

.. code:: ipython3

    prediction_proba


.. parsed-literal::

    .. table:: 
    
        +-------+-------+
        |class_0|class_1|
        +=======+=======+
        | 0.2525| 0.7475|
        +-------+-------+


Get detailed explanability associated to the prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  You can use the method detail_contributions for detailed
   contributions of each of your features for each row of your new
   dataset.
-  For classification problems, it automatically associates
   contributions with the right predicted label.
-  The predicted label are computed automatically with predict method or
   you can specify in add_input method an ypred.

.. code:: ipython3

    detailed_contributions = predictor_load.detail_contributions()

The ypred has already been renamed with the value that we have given in
the label_dict.

.. code:: ipython3

    detailed_contributions


.. parsed-literal::

    .. table:: 
    
        +--------+------+------+------+--------+---------+---------+--------+--------+------+
        | ypred  |proba |Pclass| Sex  |  Age   |  SibSp  |  Parch  |  Fare  |Embarked|Title |
        +========+======+======+======+========+=========+=========+========+========+======+
        |Survived|0.7475|0.1019|0.1846|-0.01438|-0.001680|-0.005304|-0.08279| 0.03828|0.1584|
        +--------+------+------+------+--------+---------+---------+--------+--------+------+


Summarize explanability of the predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  You can use the summarize method to summarize your local
   explainability
-  This summary can be configured with the modify_mask method to suit
   your use case
-  You can also specify : >- post-processing when you initialize your
   SmartPredictor to tune the wording >- label_dict to rename your label
   in classification problems (during the initialisation of your
   SmartPredictor). >- features_dict to rename your features.

We use modify_mask method to only get the 4 most contributives features
in our local summary.

.. code:: ipython3

    predictor_load.modify_mask(max_contrib = 4)

.. code:: ipython3

    explanation = predictor_load.summarize()

-  The dictionnary of mapping given to the SmartExplainer Object allow
   us to rename the ‘Title’ feature into ‘Title of passenger’.
-  The value of this features has been worded correctly has we
   configured it : First class became 1st class.
-  Our explanability is focused on the 4 most contributives features.

.. code:: ipython3

    explanation


.. parsed-literal::

    .. table:: 
    
        +--------+------+---------+-------+--------------+------------------+-------+--------------+------------+---------+--------------+--------------+-------+--------------+
        | ypred  |proba |feature_1|value_1|contribution_1|    feature_2     |value_2|contribution_2| feature_3  | value_3 |contribution_3|  feature_4   |value_4|contribution_4|
        +========+======+=========+=======+==============+==================+=======+==============+============+=========+==============+==============+=======+==============+
        |Survived|0.7475|Sex      |female |        0.1846|Title of passenger|Miss   |        0.1584|Ticket class|1st class|        0.1019|Passenger fare|   7.25|      -0.08279|
        +--------+------+---------+-------+--------------+------------------+-------+--------------+------------+---------+--------------+--------------+-------+--------------+


Classification - choose the predicted value and customize the summary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if you want to display the contributions associated with class 0
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can change the ypred or the x given to the add_input to make new
prediction and summary of your explanability.

.. code:: ipython3

    predictor_load.add_input(x = person_x, ypred = pd.DataFrame({"ypred": [0]}))

.. code:: ipython3

    predictor_load.modify_mask(max_contrib = 3)

.. code:: ipython3

    explanation = predictor_load.summarize()

The displayed contributions and summary adapt to changing the predicted
value of y_pred from 1 to 0.

.. code:: ipython3

    explanation


.. parsed-literal::

    .. table:: 
    
        +------------+------+---------+-------+--------------+------------------+-------+--------------+------------+---------+--------------+
        |   ypred    |proba |feature_1|value_1|contribution_1|    feature_2     |value_2|contribution_2| feature_3  | value_3 |contribution_3|
        +============+======+=========+=======+==============+==================+=======+==============+============+=========+==============+
        |Not Survived|0.2525|Sex      |female |       -0.1846|Title of passenger|Miss   |       -0.1584|Ticket class|1st class|       -0.1019|
        +------------+------+---------+-------+--------------+------------------+-------+--------------+------------+---------+--------------+


Configure summary: mask one feature, select positive contributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  The modify_mask method allows us to configure the summary parameters
   of your explainability.
-  Here, we can choose to hide some features from our explanability and
   only get the one which has positive contributions.

.. code:: ipython3

    predictor_load.modify_mask(features_to_hide = ["Fare"], positive = True)

.. code:: ipython3

    explanation = predictor_load.summarize()

.. code:: ipython3

    explanation


.. parsed-literal::

    .. table:: 
    
        +------------+------+---------+-------+--------------+----------------------------------+-------+--------------+---------------------------------+-------+--------------+
        |   ypred    |proba |feature_1|value_1|contribution_1|            feature_2             |value_2|contribution_2|            feature_3            |value_3|contribution_3|
        +============+======+=========+=======+==============+==================================+=======+==============+=================================+=======+==============+
        |Not Survived|0.2525|Age      |     36|       0.01438|Relatives like children or parents|      0|      0.005304|Relatives such as brother or wife|      1|      0.001680|
        +------------+------+---------+-------+--------------+----------------------------------+-------+--------------+---------------------------------+-------+--------------+


Configure summary: the threshold parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We choose to only show the features which has a contribution greater
than 0.01.

.. code:: ipython3

    predictor_load.modify_mask(threshold = 0.01)

.. code:: ipython3

    explanation = predictor_load.summarize()

.. code:: ipython3

    explanation


.. parsed-literal::

    .. table:: 
    
        +------------+------+---------+-------+--------------+
        |   ypred    |proba |feature_1|value_1|contribution_1|
        +============+======+=========+=======+==============+
        |Not Survived|0.2525|Age      |     36|       0.01438|
        +------------+------+---------+-------+--------------+

