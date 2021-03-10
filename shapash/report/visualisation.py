from IPython.display import display, Markdown, Latex, HTML
import matplotlib.pyplot as plt

import pandas as pd


def print_md(text: str):
    """
    Renders markdown text.
    """
    display(Markdown(text))


def print_latex(text: str):
    """
    Renders Latex text.
    """
    display(Latex(text))


def print_html(text: str):
    """
    Renders HTML text.
    """
    display(HTML(text))


def print_css_table():
    print_html("""
    <style type="text/css">
        table.greyGridTable {
            border: solid 1px #DDEEEE;
            border-collapse: collapse;
            border-spacing: 0;
            margin-left: 5px;
        }
        table.greyGridTable tbody td {
              border: solid 1px #DEDDEE;
              color: #333;
              padding: 5px;
              margin-left: 10px;
              margin-right: 10px;
              text-align: center;
              text-shadow: 1px 1px 1px #fff;
        }
        table.greyGridTable thead th {
            border: solid 1px #DEDDEE;
            color: #808080;
            padding: 5px;
            text-align: center;
            text-shadow: 1px 1px 1px #fff;
        }
    </style>
    """)


def convert_fig_to_html(fig):
    """ Convert Matplotlib figure 'fig' into a <img> tag for HTML use using base64 encoding. """
    import io
    import base64
    s = io.BytesIO()
    fig.savefig(s, format='png', bbox_inches="tight")
    plt.close()
    s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    return '<img align="left" src="data:image/png;base64,%s">' % s


def print_df_and_image(df: pd.DataFrame, fig):
    print_html(f"""
    <div class="row-fluid">
      <div class="col-sm-6">{df.to_html(classes="greyGridTable")}</div>
      <div class="col-sm-6">{convert_fig_to_html(fig)}</div>
    </div>
    """)


def print_figure(fig):
    print_html(convert_fig_to_html(fig))

