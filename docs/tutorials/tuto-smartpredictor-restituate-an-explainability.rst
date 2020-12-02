From model training to deployment - an introduction to the SmartPredictor object
================================================================================

Shapash create a SmartPredictor to make prediction and have
explainability for operational needs in deployment context.
Explainability can be restitutate to users to have a simple synthetic
explanation. SmartPredictor allows users to configure the summary to
satisfy their operational needs. It is an object dedicated to
deployment, lighter and more consistent than Smartexplainer.
SmartPredictor can be used with an API or in batch mode.

In this tutorial, we will go further to help you getting started with
the SmartPredictor Object of Shapash.

Contents: - Build a SmartPredictor - Save and Load a Smartpredictor -
Add input - Use label and wording - Summarize explaination

We used Kaggle’s `Titanic <https://www.kaggle.com/c/titanic>`__ dataset

Import Dataset
--------------

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

    from shapash.explainer.smart_explainer import SmartExplainer
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
---------------------------

In this section, we will train a Machine Learning supervized model with
our data. In our example, we are confronted to a classification problem.

.. code:: ipython3

    y = titan_df['Survived']
    X = titan_df.drop('Survived', axis=1)

.. code:: ipython3

    varcat=['Pclass','Sex','Embarked','Title']

Encoding Categorical Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We need to use a preprocessing on our data for handling categorical
features before the training step.

.. code:: ipython3

    categ_encoding = OrdinalEncoder(cols=varcat, \
                                    handle_unknown='ignore', \
                                    return_df=True).fit(X)
    X = categ_encoding.transform(X)

Train Test split + Random Forest fit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.75, random_state=1)
    
    rf = RandomForestClassifier(n_estimators=100,min_samples_leaf=3)
    rf.fit(Xtrain, ytrain)




.. parsed-literal::

    RandomForestClassifier(min_samples_leaf=3)



.. code:: ipython3

    ypred=pd.DataFrame(rf.predict(Xtest),columns=['pred'],index=Xtest.index)

Create a Smarpredictor from you SmartExplainer
----------------------------------------------

When the training step is done, we can start to initialize our
SmartExplainer Object.

.. code:: ipython3

    from shapash.explainer.smart_explainer import SmartExplainer

SmartExplainer takes only necessary dicts of the model features

Use Label and Wording
^^^^^^^^^^^^^^^^^^^^^

Here, we will use labels and wording to get a more understandable
explanabily. - features_dict : allow users to rename the features of
their datasets with the one needed - label_dict : allow users in
classification problems to rename label predicted with the one needed -
postprocessing : allow users to apply some wording to the features
wanted

.. code:: ipython3

    feature_dict = {'Pclass': 'Ticket class',
     'Sex': 'Sex',
     'Age': 'Age',
     'SibSp': 'Relatives such as brother or wife',
     'Parch': 'Relatives like children or parents',
     'Fare': 'Passenger fare',
     'Embarked': 'Port of embarkation',
     'Title': 'Title of passenger'}

.. code:: ipython3

    label_dict = {0: "Not Survived",1: "Survived"}

.. code:: ipython3

    postprocessing = {"Pclass": {'type': 'transcoding', 'rule': { 'First class' : '1st class', 'Second class' : '2nd class', "Third class" : "3rd class"}}}

.. code:: ipython3

    xpl = SmartExplainer(label_dict = label_dict, features_dict=feature_dict)

.. code:: ipython3

    xpl.compile(
        x=Xtest,
        model=rf,
        preprocessing=categ_encoding,
        y_pred=ypred,
        postprocessing = postprocessing
    )


.. parsed-literal::

    Backend: Shap TreeExplainer


Switch to SmartPredictor Object
-------------------------------

-  to_smartpredictor() is a method create to get a SmartPredictor
   object.
-  It allows users to switch from a SmartExplainer used for data mining
   to the SmartPredictor.
-  SmartPredictor takes only neccessary attribute to be lighter and more
   consistent than Smartexplainer.
-  SmartPredictor object is specific for deployement.
-  In this section, we will learn how to initialize a SmartPredictor.
-  SmartPredictor allows you not to only understand results of your
   models but also to produce those results on new data automatically.
-  It will make new predictions and summarize explainability that you
   configured to make it operational to your needs.
-  SmartPredictor take only neccessary attribute to be lighter and more
   consistent than Smartexplainer for deployment context.
-  SmartPredictor can be use with API or in batch mode.

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
------------------------------------------

-  Once our SmartPredictor has been initialized, we can easily apply
   predictions and summary to new datasets.
-  First, we have to specify a new dataset which can be a
   pandas.DataFrame or a dictionnary (usefull when you decide to use an
   API in your deployment process)
-  We will use the add_input method of the SmartPredictor. (see the
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
     'Title': 'Miss'}

.. code:: ipython3

    predictor_load.add_input(x=person_x)

If you don’t specify an ypred in the add_input method, SmartPredictor
will use its predict method to automatically affect the predicted value
to ypred.

Make prediction
^^^^^^^^^^^^^^^

Then, we can see that ypred is automatically computed in add_input
method by checking the attribute data[“ypred”] thanks to our model
trained and the new dataset given.

.. code:: ipython3

    predictor_load.data["ypred"].head()


.. parsed-literal::

    .. table:: 
    
        +--------+------+
        | ypred  |proba |
        +========+======+
        |Survived|0.7156|
        +--------+------+


We can also use the predict_proba method of the SmartPredictor to
automatically compute the probabilties associated to each label possible
with our model and the new dataset.

.. code:: ipython3

    prediction_proba = predictor_load.predict_proba()

.. code:: ipython3

    prediction_proba


.. parsed-literal::

    .. table:: 
    
        +-------+-------+
        |class_0|class_1|
        +=======+=======+
        | 0.2376| 0.7624|
        +-------+-------+


Get detailed explanability associated to the prediction
-------------------------------------------------------

-  You can use the method detail_contributions to see the detailed
   contributions of each of your features for each row of your new
   dataset.
-  For classification problems, it will automatically associated
   contributions with the right predicted label. (like you can see
   below)
-  The predicted label can be compute automatically with predict method
   or you can specify in add_input method an ypred

.. code:: ipython3

    detailed_contributions = predictor_load.detail_contributions()

You can notice here that the ypred has already been renamed with the
value that we have given in the label_dict.

.. code:: ipython3

    detailed_contributions.head()


.. parsed-literal::

    .. table:: 
    
        +--------+------+------+------+--------+--------+---------+--------+--------+------+
        | ypred  |proba |Pclass| Sex  |  Age   | SibSp  |  Parch  |  Fare  |Embarked|Title |
        +========+======+======+======+========+========+=========+========+========+======+
        |Survived|0.7624|0.1097|0.1586|-0.02100|0.007639|-0.003601|-0.09370| 0.04038|0.1928|
        +--------+------+------+------+--------+--------+---------+--------+--------+------+


Summarize explanability of the predictions
------------------------------------------

-  You can use the summarize method to summarize your local
   explainability
-  This summary can be configured with the method modify_mask in order
   for you to have the explainability that satisfy your operational
   needs
-  You can also specify : >- a postprocessing when you initialize your
   SmartPredictor to apply a wording to several values of your dataset.
   >- a label_dict to rename your label in classification problems
   (during the initialisation of your SmartPredictor). >- a
   features_dict to rename your features.

Here, we chose to use modify_mask method to only get the 3 most
contributives features in our explanability.

.. code:: ipython3

    predictor_load.modify_mask(max_contrib=3)

.. code:: ipython3

    explanation = predictor_load.summarize()

-  You can notice in the summarize that the dictionnary of mapping given
   to the SmartExplainer Object allow us to rename the ‘Title’ feature
   into ‘Title of passenger’.
-  Also, we can see that the value of this features has been worded
   correctly has we configured it : First class became 1st class.
-  Our explanability is focused on the 3 most contributive features.

.. code:: ipython3

    explanation.head()


.. parsed-literal::

    .. table:: 
    
        +--------+------+------------------+-------+--------------+---------+-------+--------------+------------+---------+--------------+
        | ypred  |proba |    feature_1     |value_1|contribution_1|feature_2|value_2|contribution_2| feature_3  | value_3 |contribution_3|
        +========+======+==================+=======+==============+=========+=======+==============+============+=========+==============+
        |Survived|0.7624|Title of passenger|Miss   |        0.1928|Sex      |female |        0.1586|Ticket class|1st class|        0.1097|
        +--------+------+------------------+-------+--------------+---------+-------+--------------+------------+---------+--------------+


Configure your summary easily
-----------------------------

If contributions wanted are the ones associated to the class 0 (More useful in multiclass classification)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Then, you can easily change the ypred or the x given to the add_input to
make new prediction and summary of your explanability

You can specify an ypred to get explanability from the label that you
prefer to predict instead.

.. code:: ipython3

    predictor_load.add_input(x=person_x, ypred=pd.DataFrame({0}))

.. code:: ipython3

    predictor_load.modify_mask(max_contrib=3)

.. code:: ipython3

    explanation = predictor_load.summarize()

Here, we changed the ypred from label predicted 1 to 0 which allow us to
automatically get the explanability of features that are associated to
the right label predicted.

.. code:: ipython3

    explanation.head()


.. parsed-literal::

    .. table:: 
    
        +------------+------+------------------+-------+--------------+---------+-------+--------------+------------+---------+--------------+
        |     0      |proba |    feature_1     |value_1|contribution_1|feature_2|value_2|contribution_2| feature_3  | value_3 |contribution_3|
        +============+======+==================+=======+==============+=========+=======+==============+============+=========+==============+
        |Not Survived|0.2376|Title of passenger|Miss   |       -0.1928|Sex      |female |       -0.1586|Ticket class|1st class|       -0.1097|
        +------------+------+------------------+-------+--------------+---------+-------+--------------+------------+---------+--------------+


If users don’t want one feature and want only positive contributions to restituate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  The modify_mask method allows us to configure the explanability to
   satisfy our needs in opeartional process.
-  Here, we can choose to hide some features from our explanability and
   only get the one which has positive contributions.

.. code:: ipython3

    predictor_load.modify_mask(features_to_hide=["Sex"], positive=True)

.. code:: ipython3

    explanation = predictor_load.summarize()

.. code:: ipython3

    explanation.head()


.. parsed-literal::

    .. table:: 
    
        +------------+------+--------------+-------+--------------+---------+-------+--------------+----------------------------------+-------+--------------+
        |     0      |proba |  feature_1   |value_1|contribution_1|feature_2|value_2|contribution_2|            feature_3             |value_3|contribution_3|
        +============+======+==============+=======+==============+=========+=======+==============+==================================+=======+==============+
        |Not Survived|0.2376|Passenger fare|   7.25|       0.09370|Age      |     36|       0.02100|Relatives like children or parents|      0|      0.003601|
        +------------+------+--------------+-------+--------------+---------+-------+--------------+----------------------------------+-------+--------------+


If users want to restituate only contributions with a minimum of impact
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here, we chose to only show the features which has a contribution
greater than 0.05.

.. code:: ipython3

    predictor_load.modify_mask(threshold=0.05)

.. code:: ipython3

    explanation = predictor_load.summarize()

.. code:: ipython3

    explanation.head()


.. parsed-literal::

    .. table:: 
    
        +------------+------+--------------+-------+--------------+
        |     0      |proba |  feature_1   |value_1|contribution_1|
        +============+======+==============+=======+==============+
        |Not Survived|0.2376|Passenger fare|   7.25|       0.09370|
        +------------+------+--------------+-------+--------------+

