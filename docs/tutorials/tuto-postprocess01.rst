How to use postprocessing parameter in compile method
=====================================================

Compile method is a method that creates the explainer you need for your
model. This compile method has many parameters, and among those is
``postprocessing`` parameter, that will be explained in this tutorial.
This parameter allows to **modify** the dataset with several techniques,
for a better visualization. This tutorial presents the different way you
can modify data, and the right syntax to do it.

Contents: - Loading dataset and fitting a model.

-  Creating our SmartExplainer and compiling it without postprocessing.

-  New SmartExplainer with postprocessing parameter.

Data from Kaggle: `Titanic <https://www.kaggle.com/c/titanic/data>`__

.. code:: ipython3

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble.forest import RandomForestClassifier

Building Supervized Model
-------------------------

First step : Importing our dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    from shapash.data.data_loader import data_loading
    titanic_df, titanic_dict = data_loading('titanic')
    y_df=titanic_df['Survived']
    X_df=titanic_df[titanic_df.columns.difference(['Survived'])]

.. code:: ipython3

    titanic_df.head()

.. table::

    +--------+-----------+---------------------------------------------+------+---+-----+-----+-----+-----------+-----+
    |Survived|  Pclass   |                    Name                     | Sex  |Age|SibSp|Parch|Fare | Embarked  |Title|
    +========+===========+=============================================+======+===+=====+=====+=====+===========+=====+
    |       0|Third class|Braund Owen Harris                           |male  | 22|    1|    0| 7.25|Southampton|Mr   |
    +--------+-----------+---------------------------------------------+------+---+-----+-----+-----+-----------+-----+
    |       1|First class|Cumings John Bradley (Florence Briggs Thayer)|female| 38|    1|    0|71.28|Cherbourg  |Mrs  |
    +--------+-----------+---------------------------------------------+------+---+-----+-----+-----+-----------+-----+
    |       1|Third class|Heikkinen Laina                              |female| 26|    0|    0| 7.92|Southampton|Miss |
    +--------+-----------+---------------------------------------------+------+---+-----+-----+-----+-----------+-----+
    |       1|First class|Futrelle Jacques Heath (Lily May Peel)       |female| 35|    1|    0|53.10|Southampton|Mrs  |
    +--------+-----------+---------------------------------------------+------+---+-----+-----+-----+-----------+-----+
    |       0|Third class|Allen William Henry                          |male  | 35|    0|    0| 8.05|Southampton|Mr   |
    +--------+-----------+---------------------------------------------+------+---+-----+-----+-----+-----------+-----+


Second step : Encode our categorical variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    from category_encoders import OrdinalEncoder
    
    categorical_features = [col for col in X_df.columns if X_df[col].dtype == 'object']
    
    encoder = OrdinalEncoder(
        cols=categorical_features,
        handle_unknown='ignore',
        return_df=True).fit(X_df)
    
    X_df = encoder.transform(X_df)

Third step : Train/test split and fitting our model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    Xtrain, Xtest, ytrain, ytest = train_test_split(X_df, y_df, train_size=0.75, random_state=1)

.. code:: ipython3

    classifier = RandomForestClassifier(n_estimators=200).fit(Xtrain, ytrain)

.. code:: ipython3

    y_pred = pd.DataFrame(classifier.predict(Xtest), columns=['pred'], index=Xtest.index) # Predictions

Fourth step : Declaring our Explainer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    from shapash.explainer.smart_explainer import SmartExplainer

.. code:: ipython3

    xpl = SmartExplainer(features_dict=titanic_dict) # Optional parameter, dict specifies label for features name 

Compiling without postprocessing parameter
------------------------------------------

After declaring our explainer, we need to compile it on our model and
data in order to have information.

.. code:: ipython3

    xpl.compile(
        x=Xtest,
        model=classifier,
        preprocessing=encoder, # Optional: compile step can use inverse_transform method
        y_pred=y_pred # Optional
    )


.. parsed-literal::

    Backend: Shap TreeExplainer


We can now use our explainer to understand model predictions, through
plots or data. We also can find our original dataset, before
preprocessing.

.. code:: ipython3

    xpl.x_init.head()



.. table::

    +------------+---------------------------------------------+------+----+-----+-----+-----+-----------+-----+
    |   Pclass   |                    Name                     | Sex  |Age |SibSp|Parch|Fare | Embarked  |Title|
    +============+=============================================+======+====+=====+=====+=====+===========+=====+
    |First class |Swift Frederick Joel (Margaret Welles Barron)|female|48.0|    0|    0|25.93|Southampton|Mrs  |
    +------------+---------------------------------------------+------+----+-----+-----+-----+-----------+-----+
    |Third class |Nenkoff Christo                              |male  |29.5|    0|    0| 7.90|Southampton|Mr   |
    +------------+---------------------------------------------+------+----+-----+-----+-----+-----------+-----+
    |Second class|Ilett Bertha                                 |female|17.0|    0|    0|10.50|Southampton|Miss |
    +------------+---------------------------------------------+------+----+-----+-----+-----+-----------+-----+
    |Third class |Peters Katie                                 |female|29.5|    0|    0| 8.14|Queenstown |Miss |
    +------------+---------------------------------------------+------+----+-----+-----+-----+-----------+-----+
    |Second class|Hart Eva Miriam                              |female| 7.0|    0|    2|26.25|Southampton|Miss |
    +------------+---------------------------------------------+------+----+-----+-----+-----+-----------+-----+



All the analysis you can do is in this tutorial :
`Tutorial <https://github.com/MAIF/shapash/blob/master/tutorial/tutorial02-Shapash-overview-in-Jupyter.ipynb>`__

Compiling with postprocessing parameter
---------------------------------------

Nevertheless, here we want to add postprocessing to our data to
understand them better, and to have a better **explicability**.

The syntax for the **postprocessing parameter** is as follow :

.. code:: python

   postprocess = {
       'name_of_the_feature': {'type': 'type_of_modification', 'rule': 'rule_to_apply'},
       'second_name_of_features': {'type': 'type_of_modification', 'rule': 'rule_to_apply'},
       ...
   }

You have five different types of modifications :

-  

   1) **prefix** : If you want to modify the beginning of the data. The
      syntax is

.. code:: python

   {'features_name': {'type': 'prefix',
                        'rule': 'Example : '}
   }

-  

   2) **suffix** : If you want to add something at the end of some
      features, the syntax is similar :

.. code:: python

   {'features_name': {'type': 'suffix',
                        'rule': ' is an example'}
   }

-  

   3) **transcoding** : This is a mapping function which modifies
      categorical variables. The syntax is :

.. code:: python

   {'features_name': {'type': 'transcoding',  
                        'rule': {'old_name1': 'new_name1',
                                 'old_name2': 'new_name2',
                                 ...
                                }
                       }
   }

If you don’t map all possible values, those values won’t be modified.

-  

   4) **regex** : If you want to modify strings, you can do it by
      regular expressions like this:

.. code:: python

   {'features_name': {'type': 'regex', 
                        'rule': {'in': '^M',
                                 'out': 'm'
                                }
                       }
   }

-  

   5) **case** : If you want to change the case of a certain features,
      you can or change everything in lowercase with
      ``'rule': 'lower'``, or change in uppercase with
      ``'rule': 'upper'``. The syntax is :

.. code:: python

   {'features_name': {'type': 'case', 
                        'rule': 'upper'}

Of course, you don’t have to modify all features. Let’s give an example.

.. code:: ipython3

    postprocess = {
        'Age': {'type': 'suffix', 
                'rule': ' years old' # Adding 'years old' at the end
               }, 
        'Sex': {'type': 'transcoding', 
                'rule': {'male': 'Man',
                         'female': 'Woman'}
               },
        'Pclass': {'type': 'regex', 
                   'rule': {'in': ' class$', 
                            'out': ''} # Deleting 'class' word at the end
                  },
        'Fare': {'type': 'prefix', 
                 'rule': '$' # Adding $ at the beginning
                }, 
        'Embarked': {'type': 'case', 
                     'rule': 'upper'
                    }
    }

You can now add this postprocess dict in parameter :

.. code:: ipython3

    xpl_postprocess = SmartExplainer(features_dict=titanic_dict) # New explainer

.. code:: ipython3

    xpl_postprocess.compile(
        x=Xtest,
        model=classifier,
        preprocessing=encoder, # Optional: compile step can use inverse_transform method
        y_pred=y_pred, # Optional
        postprocessing=postprocess
    )


.. parsed-literal::

    Backend: Shap TreeExplainer


You can now visualize your dataset, which is modified.

.. code:: ipython3

    xpl_postprocess.x_init.head()



.. table::

    +------+---------------------------------------------+-----+--------------+-----+-----+------+-----------+-----+
    |Pclass|                    Name                     | Sex |     Age      |SibSp|Parch| Fare | Embarked  |Title|
    +======+=============================================+=====+==============+=====+=====+======+===========+=====+
    |First |Swift Frederick Joel (Margaret Welles Barron)|Woman|48.0 years old|    0|    0|$25.93|SOUTHAMPTON|Mrs  |
    +------+---------------------------------------------+-----+--------------+-----+-----+------+-----------+-----+
    |Third |Nenkoff Christo                              |Man  |29.5 years old|    0|    0|$7.9  |SOUTHAMPTON|Mr   |
    +------+---------------------------------------------+-----+--------------+-----+-----+------+-----------+-----+
    |Second|Ilett Bertha                                 |Woman|17.0 years old|    0|    0|$10.5 |SOUTHAMPTON|Miss |
    +------+---------------------------------------------+-----+--------------+-----+-----+------+-----------+-----+
    |Third |Peters Katie                                 |Woman|29.5 years old|    0|    0|$8.14 |QUEENSTOWN |Miss |
    +------+---------------------------------------------+-----+--------------+-----+-----+------+-----------+-----+
    |Second|Hart Eva Miriam                              |Woman|7.0 years old |    0|    2|$26.25|SOUTHAMPTON|Miss |
    +------+---------------------------------------------+-----+--------------+-----+-----+------+-----------+-----+



All the plots are also modified with the postprocessing modifications.

Application with to_pandas method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main purpose of postprocessing modifications is a better
understanding of the data, especially when the features names are not
specified, such as in to_pandas() method, which orders the features
depending on their importance.

.. code:: ipython3

    xpl_postprocess.to_pandas().head()


.. parsed-literal::

    to_pandas params: {'features_to_hide': None, 'threshold': None, 'positive': None, 'max_contrib': 20}



.. table::

    +-+------------------+-------+--------------+------------------+-------+--------------+-------------------+----------+--------------+----------------+---------------------------------------------+--------------+---------------------------------+-------+--------------+-------------------+---------------+--------------+----------------------------------+---------------+--------------+----------------------------------+------------+--------------+----------------------------------+--------------+--------------+
    |0|    feature_1     |value_1|contribution_1|    feature_2     |value_2|contribution_2|     feature_3     | value_3  |contribution_3|   feature_4    |                   value_4                   |contribution_4|            feature_5            |value_5|contribution_5|     feature_6     |    value_6    |contribution_6|            feature_7             |    value_7    |contribution_7|            feature_8             |  value_8   |contribution_8|            feature_9             |   value_9    |contribution_9|
    +=+==================+=======+==============+==================+=======+==============+===================+==========+==============+================+=============================================+==============+=================================+=======+==============+===================+===============+==============+==================================+===============+==============+==================================+============+==============+==================================+==============+==============+
    |1|Title of passenger|Mrs    |       0.15923|Sex               |Woman  |       0.14733|Ticket class       |First     |       0.10928|Name, First name|Swift Frederick Joel (Margaret Welles Barron)|      -0.10006|Passenger fare                   |$25.93 |      -0.06283|Age                |48.0 years old |      -0.05674|Relatives such as brother or wife |              0|       0.02809|Port of embarkation               |SOUTHAMPTON |     -0.012219|Relatives like children or parents|             0|     -0.011041|
    +-+------------------+-------+--------------+------------------+-------+--------------+-------------------+----------+--------------+----------------+---------------------------------------------+--------------+---------------------------------+-------+--------------+-------------------+---------------+--------------+----------------------------------+---------------+--------------+----------------------------------+------------+--------------+----------------------------------+--------------+--------------+
    |0|Title of passenger|Mr     |       0.09118|Passenger fare    |$7.9   |       0.07093|Sex                |Man       |       0.06937|Age             |29.5 years old                               |       0.06333|Ticket class                     |Third  |       0.04726|Port of embarkation|SOUTHAMPTON    |       0.01630|Name, First name                  |Nenkoff Christo|       0.01246|Relatives such as brother or wife |           0|     -0.005863|Relatives like children or parents|             0|      0.003994|
    +-+------------------+-------+--------------+------------------+-------+--------------+-------------------+----------+--------------+----------------+---------------------------------------------+--------------+---------------------------------+-------+--------------+-------------------+---------------+--------------+----------------------------------+---------------+--------------+----------------------------------+------------+--------------+----------------------------------+--------------+--------------+
    |1|Sex               |Woman  |       0.18265|Title of passenger|Miss   |       0.17916|Ticket class       |Second    |       0.09173|Name, First name|Ilett Bertha                                 |      -0.05084|Passenger fare                   |$10.5  |      -0.02845|Port of embarkation|SOUTHAMPTON    |      -0.02613|Relatives such as brother or wife |              0|       0.02270|Relatives like children or parents|           0|     -0.010478|Age                               |17.0 years old|      0.000712|
    +-+------------------+-------+--------------+------------------+-------+--------------+-------------------+----------+--------------+----------------+---------------------------------------------+--------------+---------------------------------+-------+--------------+-------------------+---------------+--------------+----------------------------------+---------------+--------------+----------------------------------+------------+--------------+----------------------------------+--------------+--------------+
    |1|Title of passenger|Miss   |       0.21210|Sex               |Woman  |       0.15721|Port of embarkation|QUEENSTOWN|       0.11570|Ticket class    |Third                                        |      -0.08853|Relatives such as brother or wife|      0|       0.03107|Passenger fare     |$8.14          |      -0.02470|Relatives like children or parents|              0|       0.01950|Name, First name                  |Peters Katie|      0.016750|Age                               |29.5 years old|      0.011931|
    +-+------------------+-------+--------------+------------------+-------+--------------+-------------------+----------+--------------+----------------+---------------------------------------------+--------------+---------------------------------+-------+--------------+-------------------+---------------+--------------+----------------------------------+---------------+--------------+----------------------------------+------------+--------------+----------------------------------+--------------+--------------+
    |1|Title of passenger|Miss   |       0.20752|Ticket class      |Second |       0.15710|Sex                |Woman     |       0.10215|Age             |7.0 years old                                |       0.06272|Relatives such as brother or wife|      0|       0.03309|Name, First name   |Hart Eva Miriam|       0.01338|Port of embarkation               |SOUTHAMPTON    |      -0.01070|Passenger fare                    |$26.25      |      0.010183|Relatives like children or parents|             2|      0.005580|
    +-+------------------+-------+--------------+------------------+-------+--------------+-------------------+----------+--------------+----------------+---------------------------------------------+--------------+---------------------------------+-------+--------------+-------------------+---------------+--------------+----------------------------------+---------------+--------------+----------------------------------+------------+--------------+----------------------------------+--------------+--------------+

