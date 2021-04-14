Shapash Report
==============

   The Shapash Report feature allows data scientists to deliver to
   anyone who is interested in their project **a document that freezes
   different aspects of their work as a basis of an audit report**. This
   document can be easily shared across teams and does not require
   anything else than a working internet connexion.

| The shapash ``generate_report`` method allows to generate a report of
  your project.
| The result is a standalone HTML file that does not require any
  external dependency or server to work.
| The only requirement for the document to display properly is an active
  internet connexion.

The report contains the following information :

1. General information about the project
2. Description of the dataset used
3. Documentation about data preparation and feature engineering
4. Details about your model used (library, parameters…)
5. Exploration of the data with a focus on the difference between train and test sets
6. Global explainability of the model
7. Model performance

   The first three points are generated using a YML file that the user
   should fill. An example is available
   `here <https://github.com/MAIF/shapash/blob/master/tutorial/report/utils/project_info.yml>`__.

This tutorial presents an example of how one can generate the Shapash
Report.

Content:

- Set up an example project
- Create and fill your project information that will be displayed in the report
- Generate the base Shapash Report
- *Go further*: Generate a custom report

Data from Kaggle `House
Prices <https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data>`__

   Note : you may need to download the HTML report locally and open it
   in your browser otherwise it may not show properly.

.. code:: ipython3

    import pandas as pd
    from category_encoders import OrdinalEncoder
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

Building Supervized Model
-------------------------

.. code:: ipython3

    from shapash.data.data_loader import data_loading
    house_df, house_dict = data_loading('house_prices')
    y_df=house_df['SalePrice']
    X_df=house_df[house_df.columns.difference(['SalePrice'])]

.. code:: ipython3

    from category_encoders import OrdinalEncoder
    
    categorical_features = [col for col in X_df.columns if X_df[col].dtype == 'object']
    
    encoder = OrdinalEncoder(
        cols=categorical_features,
        handle_unknown='ignore',
        return_df=True).fit(X_df)
    
    X_df = encoder.transform(X_df)

.. code:: ipython3

    Xtrain, Xtest, ytrain, ytest = train_test_split(X_df, y_df, train_size=0.75, random_state=1)

.. code:: ipython3

    regressor = RandomForestRegressor(n_estimators=50).fit(Xtrain, ytrain)

.. code:: ipython3

    y_pred = pd.DataFrame(regressor.predict(Xtest),columns=['pred'], index=Xtest.index)

Fill your project information
-----------------------------

**The next step is to create a YML file containing information about
your project.**

| We will use the example file available
  `here <https://github.com/MAIF/shapash/blob/master/tutorial/report/utils/project_info.yml>`__.
| **You are welcome to use this file as a template for your own
  report.**

We display the information contained in the YML file below :

.. code:: ipython3

    import yaml
    
    with open(r'utils/project_info.yml') as file:
        project_info = yaml.full_load(file)
    
    print(yaml.dump(project_info, sort_keys=False))

--------------

**If you want to create your own custom file :**

| The keys of the YML file are the titles of the different sections in
  the report.
| The YML file must then respect the following format:

.. code:: yaml

   Title of section 1:  
       property1 name: property1 value 
       property2 name: property2 value 
       ...
   Title of section 2:  
       property1 name: property1 value 
       ...

..

   Note that the **date** can be computed automatically using the *auto*
   property value (see example above)

Generate your report
--------------------

Declare and compile SmartExplainer object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    from shapash.explainer.smart_explainer import SmartExplainer

.. code:: ipython3

    xpl = SmartExplainer(features_dict=house_dict) # optional parameter, specifies label for features name 

.. code:: ipython3

    xpl.compile(
        x=Xtest,
        model=regressor,
        preprocessing=encoder, # Optional: compile step can use inverse_transform method
        y_pred=y_pred # Optional
    )

At this step the model can be checked and inspected using different
methods of the SmartExplainer object we just created.

Please refer to the other tutorials for more information.

Generate the base Shapash Report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next we can generate the report using the ``generate_report`` method of
our SmartExplainer object.

We need to pass ``x_train``, ``y_train`` and ``y_test`` parameters in
order to explore the data used when training the model.

Please refer to the documentation for a full description of the
parameters.

.. code:: ipython3

    xpl.generate_report(
        output_file='output/report.html', 
        project_info_file='utils/project_info.yml',
        x_train=Xtrain,
        y_train=ytrain,
        y_test=ytest,
        title_story="House prices report",
        title_description="""This document is a data science report of the kaggle house prices tutorial project. 
            It was generated using the Shapash library.""",
        metrics=[
            {
                'path': 'sklearn.metrics.mean_absolute_error',
                'name': 'Mean absolute error', 
            },
            {
                'path': 'sklearn.metrics.mean_squared_error',
                'name': 'Mean squared error',
            }
        ]
    )

Note: Sometimes the jupyter kernel used when generating the report is not the right one.
If this happen you should consider using the ``kernel_name`` parameter to indicate what kernel to use.

Customize your own report
-------------------------

Now let’s customize our report by adding some new sections.

To do so : - First, **copy the base report notebook** you can find
`here <https://github.com/MAIF/shapash/blob/master/shapash/report/base_report.ipynb>`__.
This is the notebook that is used to generate the shapash report. It is
executed and then converted to an HTML file. Only the output of each
cell is kept and the code is deleted. - Then, delete or add cells
depending on what you want to change. - Finally, add the parameter
``notebook_path="path/to/your/custom/report.ipynb"`` in the
``generate_report`` method.

   **Tip** : You can use the ``working_dir`` parameter to easily work
   inside your custom notebook before using the ``generate_report``
   method. This way you can load the parameters used inside the notebook
   by papermill. Replace the ``dir_path`` inside your custom notebook
   with your own ``working_dir`` where are saved the different instances
   used.

For our simple example, we created `this
notebook <https://github.com/MAIF/shapash/blob/master/tutorial/report/utils/custom_report.ipynb>`__.
- We removed the multivariate analysis using the
``report.display_dataset_analysis(multivariate_analysis=False)`` (see
notebook utils/custom_report.ipynb for more information) - It includes
new sections **Relashionship with target variable** and **Relashionship
between training variables** in which we included new simple graphs for
this example. - We also added new cells at the end of the **metrics**
section.

Next, we use this notebook to generate our new custom report :

.. code:: ipython3

    xpl.generate_report(
        output_file='output/custom_report.html', 
        project_info_file='utils/project_info.yml',
        x_train=Xtrain,
        y_train=ytrain,
        y_test=ytest,
        title_story="House prices report",
        title_description="""This document is a data science report of the kaggle house prices tutorial project. 
            It was generated using the Shapash library.""",
        metrics=[
            {
                'path': 'sklearn.metrics.mean_absolute_error',
                'name': 'Mean absolute error', 
            },
            {
                'path': 'sklearn.metrics.mean_squared_error',
                'name': 'Mean squared error',
            }
        ],
        working_dir='working',
        notebook_path="utils/custom_report.ipynb"
    )

