from IPython.display import display, Markdown, Latex


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
