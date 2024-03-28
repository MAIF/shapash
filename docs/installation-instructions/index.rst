Installation instructions
=========================

Installing
----------

**Shapash** is intended to work with Python versions 3.9 to 3.12. Installation can be done with pip:

.. code:: bash

    pip install shapash


In order to generate the **Shapash Report** you may need to install specific libraries.
You can install these using the following command :

.. code:: bash

    pip install shapash[report]

Jupyter
-------

If you are using Jupyter and you want to show inline graph, you’ll have
few steps more.

In most cases, installing the Python ipywidgets package will also
automatically configure classic Jupyter Notebook and JupyterLab 3.0 to
display ipywidgets. With pip, do:

.. code:: bash

    pip install ipywidgets

or with conda, do:

.. code:: bash

    conda install -c conda-forge ipywidgets

(source :
`ipywidgets <https://ipywidgets.readthedocs.io/en/stable/user_install.html>`__)

Jupyter Notebook
~~~~~~~~~~~~~~~~

Most of the time, installing ipywidgets automatically configures Jupyter
Notebook to use widgets. The ipywidgets package does this by depending
on the widgetsnbextension package, which configures the classic Jupyter
Notebook to display and use widgets. If you have an old version of
Jupyter Notebook installed, you may need to manually enable the
ipywidgets notebook extension with:

.. code:: bash

    jupyter nbextension enable --py widgetsnbextension

When using virtualenv and working in an activated virtual environment,
the –sys-prefix option may be required to enable the extension and keep
the environment isolated (i.e. jupyter nbextension enable –py
widgetsnbextension –sys-prefix).

If your Jupyter Notebook and the IPython kernel are installed in
different environments (for example, separate environments are providing
different Python kernels), then the installation requires two steps:

1. Install the widgetsnbextension package in the environment containing
   the Jupyter Notebook server.
2. Install ipywidgets in each kernel’s environment that will use
   ipywidgets.

For example, if using conda environments, with Jupyter Notebook
installed on the base environment and the kernel installed in an
environment called py36, the commands are:

.. code:: bash

    conda install -n base -c conda-forge widgetsnbextension
    conda install -n py36 -c conda-forge ipywidgets

(source :
`ipywidgets <https://ipywidgets.readthedocs.io/en/stable/user_install.html>`__)

Jupyter Lab 1 or 2
~~~~~~~~~~~~~~~~~~

To install the JupyterLab extension into JupyterLab 1 or 2, you also
need to run the command below in a terminal which requires that you have
nodejs installed.

For example, if using conda environments, you can install nodejs with:

.. code:: bash

    conda install -c conda-forge nodejs

Then you can install the labextension:

.. code:: bash

    jupyter labextension install @jupyter-widgets/jupyterlab-manager

This command defaults to installing the latest version of the ipywidgets
JupyterLab extension. Depending on the version of JupyterLab you have
installed, you may need to install an older version.

If you install this extension while JupyterLab is running, you will need
to refresh the page or restart JupyterLab before the changes take
effect.

Note: A clean reinstall of the JupyterLab extension can be done by first
running the jupyter lab clean command which will remove the staging and
static directories from the lab directory. The location of the lab
directory can be queried by executing the command jupyter lab path in
your terminal.

If you have an error message like this one:

.. code:: bash

    jupyter labextension install @jupyter-widgets/jupyterlab-manager
    An error occured.
    ValueError: Please install nodejs >=10.0.0 before continuing. nodejs may be installed using conda or directly from the nodejs website.
    See the log file for details:  /tmp/jupyterlab-debug-y_g3xxpq.log

Please try to install nodejs with this command line:

.. code:: bash

    conda install -c conda-forge/label/gcc7 nodejs

(source :
`ipywidgets <https://ipywidgets.readthedocs.io/en/stable/user_install.html>`__)

Then you’ll have to install plotly in the jupyterlab environment:

.. code:: bash

    conda install -c plotly plotly
    jupyter labextension install jupyterlab-plotly

(source :
`ipywidgets <https://ipywidgets.readthedocs.io/en/stable/user_install.html>`__)

Jupyter Lab 3
~~~~~~~~~~~~~

The ipywidgets package does this by depending on the jupyterlab_widgets
package, version 1.0, which configures JupyterLab 3 to display and use
widgets.

If your JupyterLab and the IPython kernel are installed in different
environments (for example, separate environments are providing different
Python kernels), then the installation requires two steps:

1. Install the jupyterlab_widgets package (version 1.0 or later) in the
   environment containing JupyterLab.
2. Install ipywidgets in each kernel’s environment that will use
   ipywidgets.

For example, if using conda environments, with JupyterLab installed on
the base environment and the kernel installed in an environment called
py36, the commands are:

.. code:: bash

    conda install -n base -c conda-forge jupyterlab_widgets
    conda install -n py36 -c conda-forge ipywidgets

(source :
`ipywidgets <https://ipywidgets.readthedocs.io/en/stable/user_install.html>`__)

Then you’ll have to install plotly in the jupyterlab environment:

.. code:: bash

    conda install -c plotly plotly
    jupyter labextension install jupyterlab-plotly

If you have an error message like this one:

.. code:: bash

    jupyter labextension install jupyterlab-plotly
    An error occured.
    ValueError: Please install nodejs >=10.0.0 before continuing. nodejs may be installed using conda or directly from the nodejs website.
    See the log file for details:  /tmp/jupyterlab-debug-y_g3xxpq.log

(source :
`plotly <https://plotly.com/python/getting-started/#jupyter-notebook-support>`__)

Check your installation
-----------------------

To test if ipywidgets works:

.. code:: ipython3

    from __future__ import print_function
    from ipywidgets import interact, interactive, fixed, interact_manual
    import ipywidgets as widgets
    
    def f(x):
        return x
    
    interact(f, x=10);

.. image:: widget.png


To test if plotly works:

.. code:: ipython3

    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=[2, 1, 4, 3]))
    fig.add_trace(go.Bar(y=[1, 4, 3, 2]))
    fig.update_layout(title = 'Hello Figure')
    fig.show("jupyterlab")

.. image:: plotly.png


Compatibility issues
--------------------

When using Shapash, you may encounter some compatibility issues related to your environment and the libraries' versions used.
The extras requirements of Shapash allow you to update your requirements to a compatible version.

For example, if you get an error related to the *xgboost* library, you can use the following command to update it to a working version :

.. code:: bash

    pip install shapash[xgboost]

The full list of extras requirements is listed below (replace xgboost with the corresponding library on the command above) :

* xgboost
* lightgbm
* catboost
* scikit-learn
* category_encoders

