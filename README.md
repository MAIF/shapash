# Shapash

![Actions Status](https://github.com/MAIF/shapash/workflows/Build%20%26%20Test/badge.svg)
![PyPI](https://img.shields.io/pypi/v/shapash)
![Python Version](https://img.shields.io/pypi/pyversions/shapash)
![PyPI - License](https://img.shields.io/pypi/l/shapash)
[![documentation badge](https://readthedocs.org/projects/shapash/badge/?version=latest)](https://readthedocs.org/projects/shapash/)
<img align="right" src="https://raw.githubusercontent.com/MAIF/shapash/master/docs/_static/shapash-resize.png" width="300" title="shapash-logo">

<br/>
<br/>

**Shapash** is a Python library which aims to make machine learning interpretable and understandable by everyone.
It provides several types of visualization that display explicit labels that everyone can understand. 

<br/>

Data Scientists can more easily understand their models and share their results. End users can understand the decision proposed by a model using a summary of the most influential criteria.

<br/> <br/>

**The objectives of shapash:**
- To display clear and understandable results: Plots and outputs use **explicit labels** for each feature and its values:
<p align="center">
  <img src="https://raw.githubusercontent.com/MAIF/shapash/master/docs/_static/shapash-contribution_plot-example.png" width="700" title="contrib">
</p>

- To allow Data Scientists to quickly understand their models by using a **webapp** to easily navigate between global and local explainability, and understand how the different features contribute: [Live Demo Shapash-Monitor](https://shapash-demo.ossbymaif.fr/)

<a href="https://shapash-demo.ossbymaif.fr/">
  <p align="center">
    <img src="https://raw.githubusercontent.com/MAIF/shapash/master/docs/_static/shapash-webapp-demo.gif" width="800" title="contrib">
  </p>
</a>

- To **Summarize and export** the local explanation:
<p align="center">
  <img src="https://raw.githubusercontent.com/MAIF/shapash/master/docs/_static/shapash-to_pandas-example.png" width="700" title="contrib">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/MAIF/shapash/master/docs/_static/shapash-local_plot-example.png" width="700" title="contrib">
</p>

  > **Shapash** proposes a short and clear local explanation. It allows each user, whatever their Data backround, to understand a local prediction of a supervised model thanks to an summarized and explicit explanation
  

- To discuss results: **Shapash** allows Data Scientists to easily share and discuss their results with non-Data users

- Use Shapash to deploy interpretability part of your project: From model training to deployment (API or Batch Mode)

**How does shapash work?** <br />

**Shapash** is an overlay package for libraries dedicated to the interpretability of models. It uses Shap or Lime backend
to compute contributions.
**Shapash** builds on the different steps necessary to build a machine learning model to make the results understandable

<p align="center">
  <img src="https://raw.githubusercontent.com/MAIF/shapash/master/docs/_static/shapash-diagram.png" width="900" title="diagram">
</p>


# User Manual

**Shapash** works for Regression, Binary Classification or Multiclass problem. <br />
It is compatible with many models: *Catboost*, *Xgboost*, *LightGBM*, *Sklearn Ensemble*, *Linear models*, *SVM*. <br />
Shapash can use category-encoders object, sklearn ColumnTransformer or simply features dictionary. <br />
- Category_encoder: *OneHotEncoder*, *OrdinalEncoder*, *BaseNEncoder*, *BinaryEncoder*, *TargetEncoder*
- Sklearn ColumnTransformer: *OneHotEncoder*, *OrdinalEncoder*, *StandardScaler*, *QuantileTransformer*, *PowerTransformer*


You can use Shap or Lime, compute your own local contributions and use **Shapash** to display plot or summarize it. <br />

Using **Shapash** is simple and requires only a few lines of code. <br />
Most parameters are optional, you can displays plots effortlessly. <br />

But you can also tune plots and outputs, specifying labels dict, features dict, encoders, predictions, ... : 
The more you specify parameters, options, dictionaries and more the outputs will be understandable


## Installation

```
pip install shapash
```

## Getting Started: 3 minutes to Shapash

The 4 steps to display results:

- Step 1: Declare SmartExplainer Object
  > You can declare features dict here to specify the labels to display

```
from shapash.explainer.smart_explainer import SmartExplainer
xpl = SmartExplainer(features_dict=house_dict) # optional parameter
```

- Step 2: Compile Model, Dataset, Encoders, ...
  > There are 2 mandatory parameters in compile method: Model and Dataset
 
```
xpl.compile(
    x=Xtest,
    model=regressor,
    preprocessing=encoder, # Optional: compile step can use inverse_transform method
    y_pred=y_pred, # Optional
    postprocessing=postprocess # Optional: see tutorial postprocessing
)
```  

- Step 3: Display output
  > There are several outputs and plots available. for example, you can launch the web app:

```
app = xpl.run_app()
``` 

[Live Demo Shapash-Monitor](https://shapash-demo.ossbymaif.fr/)

- Step 4: From training to deployment : SmartPredictor Object
  > Shapash provides a SmartPredictor object to deploy the summary of local explanation for the operational needs.
  It is an object dedicated to deployment, lighter than SmartExplainer with additional consistency checks.
  SmartPredictor can be used with an API or in batch mode. It provides predictions, detailed or summarized local 
  explainability using appropriate wording.
  
```
predictor = xpl.to_smartpredictor()
```
See the tutorial part to know how to use the SmartPredictor object


## Tutorials
This github repository offers a lot of tutorials to allow you to start more concretely in the use of Shapash.

### More Precise Overview
- [Launch the webapp with a concrete use case](tutorial/tutorial01-Shapash-Overview-Launch-WebApp.ipynb)
- [Jupyter Overviews - The main outputs and methods available with the SmartExplainer object](tutorial/tutorial02-Shapash-overview-in-Jupyter.ipynb)
- [Shapash in production: From model training to deployment (API or Batch Mode)](tutorial/tutorial03-Shapash-overview-model-in-production.ipynb)

### More details about charts and plots
- [**Shapash** Features Importance](tutorial/plot/tuto-plot03-features-importance.ipynb)
- [Contribution plot to understand how one feature affects a prediction](tutorial/plot/tuto-plot02-contribution_plot.ipynb)
- [Summarize, display and export local contribution using filter and local_plot method](tutorial/plot/tuto-plot01-local_plot-and-to_pandas.ipynb)
- [Contributions Comparing plot to understand why predictions on several individuals are different](tutorial/plot/tuto-plot04-compare_plot.ipynb)

### The different ways to use Encoders and Dictionaries
- [Use Category_Encoder & inverse transformation](tutorial/encoder/tuto-encoder01-using-category_encoder.ipynb)
- [Use ColumnTransformers](tutorial/encoder/tuto-encoder02-using-columntransformer.ipynb)
- [Use Simple Python Dictionnaries](tutorial/encoder/tuto-encoder03-using-dict.ipynb)

### Better displaying data with postprocessing
- [Using postprocessing parameter in compile method](tutorial/postprocess/tuto-postprocess.ipynb)

### How to use shapash with Shap or Lime compute
- [Compute Shapley Contributions using **Shap**](tutorial/explainer/tuto-expl01-Shapash-Viz-using-Shap-contributions.ipynb)
- [Use **Lime** to compute local explanation, Sumarize-it with **Shapash**](tutorial/explainer/tuto-expl02-Shapash-Viz-using-Lime-contributions.ipynb)

### Deploy local explainability in production
- [Deploy local explainability in production_with_SmartPredictor](tutorial/predictor/tuto-smartpredictor-introduction-to-SmartPredictor.ipynb)