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
          <th>Survived</th>
          <th>Pclass</th>
          <th>Name</th>
          <th>Sex</th>
          <th>Age</th>
          <th>SibSp</th>
          <th>Parch</th>
          <th>Fare</th>
          <th>Embarked</th>
          <th>Title</th>
        </tr>
        <tr>
          <th>PassengerId</th>
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
          <td>0</td>
          <td>Third class</td>
          <td>Braund Owen Harris</td>
          <td>male</td>
          <td>22.0</td>
          <td>1</td>
          <td>0</td>
          <td>7.25</td>
          <td>Southampton</td>
          <td>Mr</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1</td>
          <td>First class</td>
          <td>Cumings John Bradley (Florence Briggs Thayer)</td>
          <td>female</td>
          <td>38.0</td>
          <td>1</td>
          <td>0</td>
          <td>71.28</td>
          <td>Cherbourg</td>
          <td>Mrs</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1</td>
          <td>Third class</td>
          <td>Heikkinen Laina</td>
          <td>female</td>
          <td>26.0</td>
          <td>0</td>
          <td>0</td>
          <td>7.92</td>
          <td>Southampton</td>
          <td>Miss</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1</td>
          <td>First class</td>
          <td>Futrelle Jacques Heath (Lily May Peel)</td>
          <td>female</td>
          <td>35.0</td>
          <td>1</td>
          <td>0</td>
          <td>53.10</td>
          <td>Southampton</td>
          <td>Mrs</td>
        </tr>
        <tr>
          <th>5</th>
          <td>0</td>
          <td>Third class</td>
          <td>Allen William Henry</td>
          <td>male</td>
          <td>35.0</td>
          <td>0</td>
          <td>0</td>
          <td>8.05</td>
          <td>Southampton</td>
          <td>Mr</td>
        </tr>
      </tbody>
    </table>
    </div>



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

    xpl.x_pred




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
          <th>Age</th>
          <th>Embarked</th>
          <th>Fare</th>
          <th>Name</th>
          <th>Parch</th>
          <th>Pclass</th>
          <th>Sex</th>
          <th>SibSp</th>
          <th>Title</th>
        </tr>
        <tr>
          <th>PassengerId</th>
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
          <th>863</th>
          <td>48.0</td>
          <td>Southampton</td>
          <td>25.93</td>
          <td>Swift Frederick Joel (Margaret Welles Barron)</td>
          <td>0</td>
          <td>First class</td>
          <td>female</td>
          <td>0</td>
          <td>Mrs</td>
        </tr>
        <tr>
          <th>224</th>
          <td>29.5</td>
          <td>Southampton</td>
          <td>7.90</td>
          <td>Nenkoff Christo</td>
          <td>0</td>
          <td>Third class</td>
          <td>male</td>
          <td>0</td>
          <td>Mr</td>
        </tr>
        <tr>
          <th>85</th>
          <td>17.0</td>
          <td>Southampton</td>
          <td>10.50</td>
          <td>Ilett Bertha</td>
          <td>0</td>
          <td>Second class</td>
          <td>female</td>
          <td>0</td>
          <td>Miss</td>
        </tr>
        <tr>
          <th>681</th>
          <td>29.5</td>
          <td>Queenstown</td>
          <td>8.14</td>
          <td>Peters Katie</td>
          <td>0</td>
          <td>Third class</td>
          <td>female</td>
          <td>0</td>
          <td>Miss</td>
        </tr>
        <tr>
          <th>536</th>
          <td>7.0</td>
          <td>Southampton</td>
          <td>26.25</td>
          <td>Hart Eva Miriam</td>
          <td>2</td>
          <td>Second class</td>
          <td>female</td>
          <td>0</td>
          <td>Miss</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>507</th>
          <td>33.0</td>
          <td>Southampton</td>
          <td>26.00</td>
          <td>Quick Frederick Charles (Jane Richards)</td>
          <td>2</td>
          <td>Second class</td>
          <td>female</td>
          <td>0</td>
          <td>Mrs</td>
        </tr>
        <tr>
          <th>468</th>
          <td>56.0</td>
          <td>Southampton</td>
          <td>26.55</td>
          <td>Smart John Montgomery</td>
          <td>0</td>
          <td>First class</td>
          <td>male</td>
          <td>0</td>
          <td>Mr</td>
        </tr>
        <tr>
          <th>741</th>
          <td>29.5</td>
          <td>Southampton</td>
          <td>30.00</td>
          <td>Hawksford Walter James</td>
          <td>0</td>
          <td>First class</td>
          <td>male</td>
          <td>0</td>
          <td>Mr</td>
        </tr>
        <tr>
          <th>355</th>
          <td>29.5</td>
          <td>Cherbourg</td>
          <td>7.22</td>
          <td>Yousif Wazli</td>
          <td>0</td>
          <td>Third class</td>
          <td>male</td>
          <td>0</td>
          <td>Mr</td>
        </tr>
        <tr>
          <th>450</th>
          <td>52.0</td>
          <td>Southampton</td>
          <td>30.50</td>
          <td>Peuchen Arthur Godfrey</td>
          <td>0</td>
          <td>First class</td>
          <td>male</td>
          <td>0</td>
          <td>Major</td>
        </tr>
      </tbody>
    </table>
    <p>223 rows × 9 columns</p>
    </div>



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

    xpl_postprocess.x_pred




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
          <th>Age</th>
          <th>Embarked</th>
          <th>Fare</th>
          <th>Name</th>
          <th>Parch</th>
          <th>Pclass</th>
          <th>Sex</th>
          <th>SibSp</th>
          <th>Title</th>
        </tr>
        <tr>
          <th>PassengerId</th>
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
          <th>863</th>
          <td>48.0 years old</td>
          <td>SOUTHAMPTON</td>
          <td>$25.93</td>
          <td>Swift Frederick Joel (Margaret Welles Barron)</td>
          <td>0</td>
          <td>First</td>
          <td>Woman</td>
          <td>0</td>
          <td>Mrs</td>
        </tr>
        <tr>
          <th>224</th>
          <td>29.5 years old</td>
          <td>SOUTHAMPTON</td>
          <td>$7.9</td>
          <td>Nenkoff Christo</td>
          <td>0</td>
          <td>Third</td>
          <td>Man</td>
          <td>0</td>
          <td>Mr</td>
        </tr>
        <tr>
          <th>85</th>
          <td>17.0 years old</td>
          <td>SOUTHAMPTON</td>
          <td>$10.5</td>
          <td>Ilett Bertha</td>
          <td>0</td>
          <td>Second</td>
          <td>Woman</td>
          <td>0</td>
          <td>Miss</td>
        </tr>
        <tr>
          <th>681</th>
          <td>29.5 years old</td>
          <td>QUEENSTOWN</td>
          <td>$8.14</td>
          <td>Peters Katie</td>
          <td>0</td>
          <td>Third</td>
          <td>Woman</td>
          <td>0</td>
          <td>Miss</td>
        </tr>
        <tr>
          <th>536</th>
          <td>7.0 years old</td>
          <td>SOUTHAMPTON</td>
          <td>$26.25</td>
          <td>Hart Eva Miriam</td>
          <td>2</td>
          <td>Second</td>
          <td>Woman</td>
          <td>0</td>
          <td>Miss</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>507</th>
          <td>33.0 years old</td>
          <td>SOUTHAMPTON</td>
          <td>$26.0</td>
          <td>Quick Frederick Charles (Jane Richards)</td>
          <td>2</td>
          <td>Second</td>
          <td>Woman</td>
          <td>0</td>
          <td>Mrs</td>
        </tr>
        <tr>
          <th>468</th>
          <td>56.0 years old</td>
          <td>SOUTHAMPTON</td>
          <td>$26.55</td>
          <td>Smart John Montgomery</td>
          <td>0</td>
          <td>First</td>
          <td>Man</td>
          <td>0</td>
          <td>Mr</td>
        </tr>
        <tr>
          <th>741</th>
          <td>29.5 years old</td>
          <td>SOUTHAMPTON</td>
          <td>$30.0</td>
          <td>Hawksford Walter James</td>
          <td>0</td>
          <td>First</td>
          <td>Man</td>
          <td>0</td>
          <td>Mr</td>
        </tr>
        <tr>
          <th>355</th>
          <td>29.5 years old</td>
          <td>CHERBOURG</td>
          <td>$7.22</td>
          <td>Yousif Wazli</td>
          <td>0</td>
          <td>Third</td>
          <td>Man</td>
          <td>0</td>
          <td>Mr</td>
        </tr>
        <tr>
          <th>450</th>
          <td>52.0 years old</td>
          <td>SOUTHAMPTON</td>
          <td>$30.5</td>
          <td>Peuchen Arthur Godfrey</td>
          <td>0</td>
          <td>First</td>
          <td>Man</td>
          <td>0</td>
          <td>Major</td>
        </tr>
      </tbody>
    </table>
    <p>223 rows × 9 columns</p>
    </div>



All the plots are also modified with the postprocessing modifications.

Application with to_pandas method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main purpose of postprocessing modifications is a better
understanding of the data, especially when the features names are not
specified, such as in to_pandas() method, which orders the features
depending on their importance.

.. code:: ipython3

    xpl_postprocess.to_pandas()


.. parsed-literal::

    to_pandas params: {'features_to_hide': None, 'threshold': None, 'positive': None, 'max_contrib': 20}




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
          <th>contribution_6</th>
          <th>feature_7</th>
          <th>value_7</th>
          <th>contribution_7</th>
          <th>feature_8</th>
          <th>value_8</th>
          <th>contribution_8</th>
          <th>feature_9</th>
          <th>value_9</th>
          <th>contribution_9</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>863</th>
          <td>1</td>
          <td>Title of passenger</td>
          <td>Mrs</td>
          <td>0.163479</td>
          <td>Sex</td>
          <td>Woman</td>
          <td>0.154309</td>
          <td>Ticket class</td>
          <td>First</td>
          <td>0.130221</td>
          <td>...</td>
          <td>0.0406219</td>
          <td>Name, First name</td>
          <td>Swift Frederick Joel (Margaret Welles Barron)</td>
          <td>-0.0381955</td>
          <td>Port of embarkation</td>
          <td>SOUTHAMPTON</td>
          <td>-0.0147327</td>
          <td>Relatives like children or parents</td>
          <td>0</td>
          <td>-0.00538103</td>
        </tr>
        <tr>
          <th>224</th>
          <td>0</td>
          <td>Title of passenger</td>
          <td>Mr</td>
          <td>0.094038</td>
          <td>Sex</td>
          <td>Man</td>
          <td>0.0696282</td>
          <td>Age</td>
          <td>29.5 years old</td>
          <td>0.0658556</td>
          <td>...</td>
          <td>0.0151605</td>
          <td>Relatives such as brother or wife</td>
          <td>0</td>
          <td>-0.00855039</td>
          <td>Relatives like children or parents</td>
          <td>0</td>
          <td>0.00124433</td>
          <td>Name, First name</td>
          <td>Nenkoff Christo</td>
          <td>-0.000577095</td>
        </tr>
        <tr>
          <th>85</th>
          <td>1</td>
          <td>Title of passenger</td>
          <td>Miss</td>
          <td>0.190529</td>
          <td>Sex</td>
          <td>Woman</td>
          <td>0.135507</td>
          <td>Ticket class</td>
          <td>Second</td>
          <td>0.0809714</td>
          <td>...</td>
          <td>-0.025286</td>
          <td>Relatives like children or parents</td>
          <td>0</td>
          <td>-0.0238222</td>
          <td>Relatives such as brother or wife</td>
          <td>0</td>
          <td>0.0209045</td>
          <td>Age</td>
          <td>17.0 years old</td>
          <td>-0.00702283</td>
        </tr>
        <tr>
          <th>681</th>
          <td>1</td>
          <td>Title of passenger</td>
          <td>Miss</td>
          <td>0.237477</td>
          <td>Port of embarkation</td>
          <td>QUEENSTOWN</td>
          <td>0.143451</td>
          <td>Sex</td>
          <td>Woman</td>
          <td>0.127931</td>
          <td>...</td>
          <td>0.0243567</td>
          <td>Relatives like children or parents</td>
          <td>0</td>
          <td>0.0165205</td>
          <td>Passenger fare</td>
          <td>$8.14</td>
          <td>-0.0109633</td>
          <td>Age</td>
          <td>29.5 years old</td>
          <td>0.00327866</td>
        </tr>
        <tr>
          <th>536</th>
          <td>1</td>
          <td>Title of passenger</td>
          <td>Miss</td>
          <td>0.210166</td>
          <td>Ticket class</td>
          <td>Second</td>
          <td>0.168247</td>
          <td>Sex</td>
          <td>Woman</td>
          <td>0.0876445</td>
          <td>...</td>
          <td>0.0147503</td>
          <td>Relatives like children or parents</td>
          <td>2</td>
          <td>0.0125069</td>
          <td>Port of embarkation</td>
          <td>SOUTHAMPTON</td>
          <td>-0.0119119</td>
          <td>Name, First name</td>
          <td>Hart Eva Miriam</td>
          <td>0.00654165</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>507</th>
          <td>1</td>
          <td>Title of passenger</td>
          <td>Mrs</td>
          <td>0.215332</td>
          <td>Sex</td>
          <td>Woman</td>
          <td>0.194419</td>
          <td>Ticket class</td>
          <td>Second</td>
          <td>0.166437</td>
          <td>...</td>
          <td>-0.0079185</td>
          <td>Relatives like children or parents</td>
          <td>2</td>
          <td>0.00407485</td>
          <td>Age</td>
          <td>33.0 years old</td>
          <td>-0.00263589</td>
          <td>Name, First name</td>
          <td>Quick Frederick Charles (Jane Richards)</td>
          <td>0.00162901</td>
        </tr>
        <tr>
          <th>468</th>
          <td>0</td>
          <td>Sex</td>
          <td>Man</td>
          <td>0.100602</td>
          <td>Passenger fare</td>
          <td>$26.55</td>
          <td>-0.099794</td>
          <td>Title of passenger</td>
          <td>Mr</td>
          <td>0.0967768</td>
          <td>...</td>
          <td>0.0243706</td>
          <td>Port of embarkation</td>
          <td>SOUTHAMPTON</td>
          <td>0.0124424</td>
          <td>Relatives such as brother or wife</td>
          <td>0</td>
          <td>-0.0108301</td>
          <td>Relatives like children or parents</td>
          <td>0</td>
          <td>-0.00332632</td>
        </tr>
        <tr>
          <th>741</th>
          <td>0</td>
          <td>Title of passenger</td>
          <td>Mr</td>
          <td>0.131861</td>
          <td>Sex</td>
          <td>Man</td>
          <td>0.110845</td>
          <td>Age</td>
          <td>29.5 years old</td>
          <td>0.104878</td>
          <td>...</td>
          <td>0.0339308</td>
          <td>Relatives such as brother or wife</td>
          <td>0</td>
          <td>-0.00715564</td>
          <td>Name, First name</td>
          <td>Hawksford Walter James</td>
          <td>0.00165882</td>
          <td>Relatives like children or parents</td>
          <td>0</td>
          <td>-0.00137946</td>
        </tr>
        <tr>
          <th>355</th>
          <td>0</td>
          <td>Title of passenger</td>
          <td>Mr</td>
          <td>0.12679</td>
          <td>Sex</td>
          <td>Man</td>
          <td>0.0933251</td>
          <td>Age</td>
          <td>29.5 years old</td>
          <td>0.0717939</td>
          <td>...</td>
          <td>-0.0271103</td>
          <td>Name, First name</td>
          <td>Yousif Wazli</td>
          <td>0.0163174</td>
          <td>Relatives such as brother or wife</td>
          <td>0</td>
          <td>-0.0108501</td>
          <td>Relatives like children or parents</td>
          <td>0</td>
          <td>-0.000543508</td>
        </tr>
        <tr>
          <th>450</th>
          <td>0</td>
          <td>Sex</td>
          <td>Man</td>
          <td>0.13572</td>
          <td>Title of passenger</td>
          <td>Major</td>
          <td>-0.0723023</td>
          <td>Age</td>
          <td>52.0 years old</td>
          <td>0.0690373</td>
          <td>...</td>
          <td>0.027384</td>
          <td>Relatives such as brother or wife</td>
          <td>0</td>
          <td>-0.0134144</td>
          <td>Relatives like children or parents</td>
          <td>0</td>
          <td>0.00256623</td>
          <td>Name, First name</td>
          <td>Peuchen Arthur Godfrey</td>
          <td>0.00229483</td>
        </tr>
      </tbody>
    </table>
    <p>223 rows × 28 columns</p>
    </div>


