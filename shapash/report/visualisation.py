import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import HTML, Latex, Markdown, display


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


def print_css_style():
    """Print the CSS"""
    print_html(
        """
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

    .dropdown-feature {
            position: absolute;
            display: inline-block;
        }

    .scrollable-menu {
        height: auto;
        max-height: 180px;
        overflow-x: hidden;
    }

    .scrollable-menu::-webkit-scrollbar {
        -webkit-appearance: none;
        width: 4px;
    }
    .scrollable-menu::-webkit-scrollbar-thumb {
        border-radius: 3px;
        background-color: lightgray;
        -webkit-box-shadow: 0 0 1px rgba(255,255,255,.75);
    }

    .scrollable-menu > li a {
        display: block;
        text-align: left;
        text-decoration: none !important;
        padding: 0px 0px 0px 0px;
        overflow: hidden;
    }

        .scrollable-menu > li{
          list-style:none;
    }
    </style>
    """
    )


def print_javascript_misc():
    """Print the JS"""
    print_html(
        """
    <script>
    function showBlock(groupId, blockId) {
      var elms = document.querySelectorAll('*[id^="' + groupId + '"]');
      for (i=0;i<elms.length;i++) {
        elms[i].style.display = "none";
      }
      var x = document.getElementById(blockId);
      if (x.style.display === "none") {
        x.style.display = "block";
      } else {
        x.style.display = "none";
      }

    var x = document.getElementById(blockId + '-2');
    if (x != null) {
      if (x.style.display === "none") {
        x.style.display = "block";
      } else {
        x.style.display = "none";
      }}
    }
    </script>
    """
    )


def convert_fig_to_html(fig):
    """Convert Matplotlib figure 'fig' into a <img> tag for HTML use using base64 encoding."""
    import base64
    import io

    s = io.BytesIO()
    fig.savefig(s, format="png", bbox_inches="tight")
    plt.close()
    s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    return '<img align="left" src="data:image/png;base64,%s">' % s


def html_str_df_and_image(df: pd.DataFrame, fig: plt.Figure) -> str:
    """Convert dataframe to HTML display"""
    return f"""
    <div class="row-fluid" style="margin-top:5px;">
      <div class="col-sm-6">{df.to_html(classes="greyGridTable")}</div>
      <div class="col-sm-6">{convert_fig_to_html(fig)}</div>
    </div>
    """


def print_figure(fig):
    """Print a figure as HTML"""
    print_html(convert_fig_to_html(fig))
