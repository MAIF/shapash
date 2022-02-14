<p align="center">
<img src="https://raw.githubusercontent.com/MAIF/shapash/master/docs/_static/shapash-resize.png" width="300" title="shapash-logo">
</p>


<p align="center">
  <!-- Tests -->
  <a href="https://github.com/MAIF/shapash/workflows/Build%20%26%20Test/badge.svg">
    <img src="https://github.com/MAIF/shapash/workflows/Build%20%26%20Test/badge.svg" alt="tests">
  </a>
  <!-- PyPi -->
  <a href="https://img.shields.io/pypi/v/shapash">
    <img src="https://img.shields.io/pypi/v/shapash" alt="pypi">
  </a>
  <!-- Downloads -->
  <a href="https://static.pepy.tech/personalized-badge/shapash?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads">
    <img src="https://static.pepy.tech/personalized-badge/shapash?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads" alt="downloads">
  </a>
  <!-- Python Version -->
  <a href="https://img.shields.io/pypi/pyversions/shapash">
    <img src="https://img.shields.io/pypi/pyversions/shapash" alt="pyversion">
  </a>
  <!-- License -->
  <a href="https://img.shields.io/pypi/l/shapash">
    <img src="https://img.shields.io/pypi/l/shapash" alt="license">
  </a>
  <!-- Doc -->
  <a href="https://shapash.readthedocs.io/en/latest/">
    <img src="https://readthedocs.org/projects/shapash/badge/?version=latest" alt="doc">
  </a>
</p>

## 🎉 What's new ?

| Version       | New Feature                                                                           | Description                                                                                                                            | Tutorial |
|:-------------:|:-------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|:--------:|
| 1.6.x         |  Explainability Quality Metrics <br> [article](https://towardsdatascience.com/building-confidence-on-explainability-methods-66b9ee575514)                                                                   | To help increase confidence in explainability methods, you can evaluate the relevance of your explainability using 3 metrics: **Stability**, **Consistency** and **Compacity**                   |  [<img src="https://raw.githubusercontent.com/MAIF/shapash/master/docs/_static/quality-metrics.png" width="50" title="quality-metrics">](https://github.com/MAIF/shapash/blob/master/tutorial/explainability_quality/tuto-quality01-Builing-confidence-explainability.ipynb)    |
| 1.5.x         |  ACV Backend <br>                                                                     | A new way of estimating Shapley values using ACV. [More info about ACV here](https://towardsdatascience.com/the-right-way-to-compute-your-shapley-values-cfea30509254).                   |  [<img src="https://raw.githubusercontent.com/MAIF/shapash/master/docs/_static/wheel.png" width="50" title="wheel-acv-backend">](tutorial/explainer/tuto-expl03-Shapash-acv-backend.ipynb)    |
| 1.4.x         |  Groups of features <br> [demo](https://shapash-demo2.ossbymaif.fr/)                  | You can now regroup features that share common properties together. <br>This option can be useful if your model has a lot of features. |  [<img src="https://raw.githubusercontent.com/MAIF/shapash/master/docs/_static/groups_features.gif" width="120" title="groups-features">](https://github.com/MAIF/shapash/blob/master/tutorial/common/tuto-common01-groups_of_features.ipynb)    | 
| 1.3.x         |  Shapash Report <br> [demo](https://shapash.readthedocs.io/en/latest/report.html)     | A standalone HTML report that constitutes a basis of an audit document.                                                                |  [<img src="https://raw.githubusercontent.com/MAIF/shapash/master/docs/_static/report-icon.png" width="50" title="shapash-report">](https://github.com/MAIF/shapash/blob/master/tutorial/report/tuto-shapash-report01.ipynb)    | 


## 🔍 Overview

**Shapash** is a Python library which aims to make machine learning interpretable and understandable by everyone.
It provides several types of visualization that display explicit labels that everyone can understand. 

Data Scientists can understand their models easily and share their results. End users can understand the decision proposed by a model using a summary of the most influential criteria.

Shapash also contributes to data science auditing by displaying usefull information about any model and data in a unique report. 

- Readthedocs: [![documentation badge](https://readthedocs.org/projects/shapash/badge/?version=latest)](https://shapash.readthedocs.io/en/latest/)
- [Presentation video for french speakers](https://www.youtube.com/watch?v=r1R_A9B9apk)
- Medium:
  - [Understand your model with Shapash - Towards AI](https://pub.towardsai.net/shapash-making-ml-models-understandable-by-everyone-8f96ad469eb3) 
  - [Model auditability - Towards DS](https://towardsdatascience.com/shapash-1-3-2-announcing-new-features-for-more-auditable-ai-64a6db71c919)
  - [Group of features - Towards AI](https://pub.towardsai.net/machine-learning-6011d5d9a444)
  - [Building confidence on explainability - Towards DS](https://towardsdatascience.com/building-confidence-on-explainability-methods-66b9ee575514)

## 🤝 Contributors

<div align="left">
  <div style="display: flex; align-items: flex-start;">
    <img align=middle src="https://github.com/MAIF/shapash/blob/master/docs/_static/logo_maif.png" width="18%"/>
    <img align=middle src="https://github.com/MAIF/shapash/blob/master/docs/_static/logo_quantmetry.png" width="18%" />
    <img align=middle src="https://github.com/MAIF/shapash/blob/master/docs/_static/logo_societe_generale.png" width="18%" /> 
    <img align=middle src="https://github.com/MAIF/shapash/blob/master/docs/_static/logo_groupe_vyv.png" width="18%" /> 
  </div>
</div>


## 🏆 Awards

<a href="https://raw.githubusercontent.com/MAIF/shapash/master/docs/_static/awards-argus-or.png">
  <img align="left" src="https://raw.githubusercontent.com/MAIF/shapash/master/docs/_static/awards-argus-or.png" width="180" />
</a>

<a href="https://www.kdnuggets.com/2021/04/shapash-machine-learning-models-understandable.html">
  <img src="https://www.kdnuggets.com/images/tkb-2104-g.png?raw=true" width="65" />
</a>  


## 🔥 Features

- Display clear and understandable results: plots and outputs use **explicit labels** for each feature and its values

<p align="center">
  <img align="left" src="https://github.com/MAIF/shapash/blob/master/docs/_static/shapash-grid-images-02.png?raw=true" width="28%"/>
  <img src="https://github.com/MAIF/shapash/blob/master/docs/_static/shapash-grid-images-06.png?raw=true" width="28%" />
  <img align="right" src="https://github.com/MAIF/shapash/blob/master/docs/_static/shapash-grid-images-04.png?raw=true" width="28%" /> 
</p>

<p align="center">
  <img align="left" src="https://github.com/MAIF/shapash/blob/master/docs/_static/shapash-grid-images-01.png?raw=true" width="28%" />
  <img src="https://github.com/MAIF/shapash/blob/master/docs/_static/shapash-resize.png?raw=true" width="18%" />
  <img align="right" src="https://github.com/MAIF/shapash/blob/master/docs/_static/shapash-grid-images-13.png?raw=true" width="28%" /> 
</p>

<p align="center">
  <img align="left" src="https://github.com/MAIF/shapash/blob/master/docs/_static/shapash-grid-images-12.png?raw=true" width="33%" />
  <img src="https://github.com/MAIF/shapash/blob/master/docs/_static/shapash-grid-images-03.png?raw=true" width="28%" />
  <img align="right" src="https://github.com/MAIF/shapash/blob/master/docs/_static/shapash-grid-images-10.png?raw=true" width="25%" /> 
</p>


- Allow Data Scientists to quickly understand their models by using a **webapp** to easily navigate between global and local explainability, and understand how the different features contribute: [Live Demo Shapash-Monitor](https://shapash-demo.ossbymaif.fr/)

<a href="https://shapash-demo.ossbymaif.fr/">
  <p align="center">
    <img src="https://raw.githubusercontent.com/MAIF/shapash/master/docs/_static/shapash-webapp-demo.gif" width="800" title="contrib">
  </p>
</a>

- **Summarize and export** the local explanation
> **Shapash** proposes a short and clear local explanation. It allows each user, whatever their Data background, to understand a local prediction of a supervised model thanks to a summarized and explicit explanation


- **Evaluate** the quality of your explainability using different metrics

- Easily share and discuss results with non-Data users

- Deploy interpretability part of your project: From model training to deployment (API or Batch Mode)

- Contribute to the **auditability of your model** by generating a **standalone HTML report** of your projects. [Report Example](https://shapash.readthedocs.io/en/latest/report.html) 
>We hope that this report will bring a valuable support to auditing models and data related to a better AI governance. 
Data Scientists can now deliver to anyone who is interested in their project **a document that freezes different aspects of their work as a basis of an audit report**. 
This document can be easily shared across teams (internal audit, DPO, risk, compliance...).

<a href="https://shapash.readthedocs.io/en/latest/report.html">
  <p align="center">
    <img src="https://raw.githubusercontent.com/MAIF/shapash/master/docs/_static/shapash-report-demo.gif" width="800" title="report-demo">
  </p>
</a>

## ⚙️ How Shapash works 
**Shapash** is an overlay package for libraries dedicated to the interpretability of models. It uses Shap or Lime backend
to compute contributions.
**Shapash** builds on the different steps necessary to build a machine learning model to make the results understandable

<p align="center">
  <img src="https://raw.githubusercontent.com/MAIF/shapash/master/docs/_static/shapash-diagram.png" width="700" title="diagram">
</p>


## 🛠 Installation

Shapash is intended to work with Python versions 3.6 to 3.9. Installation can be done with pip:

```
pip install shapash
```

In order to generate the Shapash Report some extra requirements are needed.
You can install these using the following command :  
```
pip install shapash[report]
```

If you encounter **compatibility issues** you may check the corresponding section in the Shapash documentation [here](https://shapash.readthedocs.io/en/latest/installation-instructions/index.html).

## 🕐 Quickstart

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

- Step 4: Generate the Shapash Report
  > This step allows to generate a standalone html report of your project using the different splits
  of your dataset and also the metrics you used:

```
xpl.generate_report(
    output_file='path/to/output/report.html',
    project_info_file='path/to/project_info.yml',
    x_train=Xtrain,
    y_train=ytrain,
    y_test=ytest,
    title_story="House prices report",
    title_description="""This document is a data science report of the kaggle house prices tutorial project.
        It was generated using the Shapash library.""",
    metrics=[{‘name’: ‘MSE’, ‘path’: ‘sklearn.metrics.mean_squared_error’}]
)
```

[Report Example](https://shapash.readthedocs.io/en/latest/report.html)

- Step 5: From training to deployment : SmartPredictor Object
  > Shapash provides a SmartPredictor object to deploy the summary of local explanation for the operational needs.
  It is an object dedicated to deployment, lighter than SmartExplainer with additional consistency checks.
  SmartPredictor can be used with an API or in batch mode. It provides predictions, detailed or summarized local 
  explainability using appropriate wording.
  
```
predictor = xpl.to_smartpredictor()
```
See the tutorial part to know how to use the SmartPredictor object

## 📖  Tutorials
This github repository offers a lot of tutorials to allow you to start more concretely in the use of Shapash.

### More Precise Overview
- [Launch the webapp with a concrete use case](tutorial/tutorial01-Shapash-Overview-Launch-WebApp.ipynb)
- [Jupyter Overviews - The main outputs and methods available with the SmartExplainer object](tutorial/tutorial02-Shapash-overview-in-Jupyter.ipynb)
- [Shapash in production: From model training to deployment (API or Batch Mode)](tutorial/tutorial03-Shapash-overview-model-in-production.ipynb)
- [Use groups of features](tutorial/common/tuto-common01-groups_of_features.ipynb)

### More details about charts and plots
- [**Shapash** Features Importance](tutorial/plot/tuto-plot03-features-importance.ipynb)
- [Contribution plot to understand how one feature affects a prediction](tutorial/plot/tuto-plot02-contribution_plot.ipynb)
- [Summarize, display and export local contribution using filter and local_plot method](tutorial/plot/tuto-plot01-local_plot-and-to_pandas.ipynb)
- [Contributions Comparing plot to understand why predictions on several individuals are different](tutorial/plot/tuto-plot04-compare_plot.ipynb)
- [Visualize interactions between couple of variables](tutorial/plot/tuto-plot05-interactions-plot.ipynb)

### The different ways to use Encoders and Dictionaries
- [Use Category_Encoder & inverse transformation](tutorial/encoder/tuto-encoder01-using-category_encoder.ipynb)
- [Use ColumnTransformers](tutorial/encoder/tuto-encoder02-using-columntransformer.ipynb)
- [Use Simple Python Dictionnaries](tutorial/encoder/tuto-encoder03-using-dict.ipynb)

### Better displaying data with postprocessing
- [Using postprocessing parameter in compile method](tutorial/postprocess/tuto-postprocess01.ipynb)

### How to use shapash with Shap, Lime or ACV
- [Compute Shapley Contributions using **Shap**](tutorial/explainer/tuto-expl01-Shapash-Viz-using-Shap-contributions.ipynb)
- [Use **Lime** to compute local explanation, Summarize-it with **Shapash**](tutorial/explainer/tuto-expl02-Shapash-Viz-using-Lime-contributions.ipynb)
- [Use **ACV backend** to compute Active Shapley Values and SDP global importance](tutorial/explainer/tuto-expl03-Shapash-acv-backend.ipynb)
- [Compile faster Lime and consistency of contributions](tutorial/explainer/tuto-expl04-Shapash-compute-Lime-faster.ipynb)

### Evaluate the quality of your explainability
- [Building confidence on explainability methods using **Stability**, **Consistency** and **Compacity** metrics](tutorial/explainability_quality/tuto-quality01-Builing-confidence-explainability.ipynb)

### Generate the Shapash Report
- [Generate a standalone HTML report of your project with generate_report](tutorial/report/tuto-shapash-report01.ipynb)

### Deploy local explainability in production
- [Deploy local explainability in production_with_SmartPredictor](tutorial/predictor/tuto-smartpredictor-introduction-to-SmartPredictor.ipynb)

