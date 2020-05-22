"""
Utils is a group of function for the library
"""
import socket
import math

def get_host_name():
    """
    Get the url of the current host
    Returns
    -------
    String
        host name
    """
    return socket.gethostname()


def inclusion(first_x, second_x):
    """
    Check if a list is included in another.

    Parameters
    ----------
    first_x : list
        List to evaluate.
    second_x : list
        Reference list to compare with.

    Returns
    -------
    bool
        True if first_x is contained in second_x.
    """
    return all(elem in second_x for elem in first_x)


def within_dict(list_param, dict_param):
    """
    Check if a list is included in either dict keys or dict values.

    Parameters
    ----------
    list_param : list
        List to evaluate.
    dict_param : dict
        Reference dictionary to compare.
    """
    return inclusion(list_param, dict_param.keys()) or inclusion(list_param, dict_param.values())


def is_nested_list(object_param):
    """
    Check if object is a nested list or not.

    Parameters
    ----------
    object_param : object
        Any object to check.

    Returns
    -------
    Bool
        True if the object is a nested list, False otherwise.
    """
    return any(isinstance(elem, list) for elem in object_param)

def add_line_break(text, nbchar, maxlen=150):
    """
    adding line break in string if necessary

    Parameters
    ----------
    text : string
        string to check in order to add line break
    nbchar : int
        number of characters before line break
    maxlen : int
        number of characters before truncation

    Returns
    -------
    string
        original text + line break
    """
    if isinstance(text,str):
        length = 0
        tot_length = 0
        input_word = text.split()
        final_sep = []
        for w in input_word[:-1]:
            length = length + len(w)
            tot_length = tot_length + len(w)
            if tot_length <= maxlen:
                if length >= nbchar:
                    length = 0
                    final_sep.append('<br />')
                else:
                    final_sep.append(' ')
        if len(final_sep) == len(input_word) - 1:
            last_char=''
        else :
            last_char=('...')

        new_string = "".join(sum(zip(input_word, final_sep+['']), ())[:-1]) + last_char
        return new_string
    else:
        return text

def truncate_str(text, maxlen= 40):
    """
    truncate a string

    Parameters
    ----------
    text : string
        string to check in order to add line break
    maxlen : int
        number of characters before truncation

    Returns
    -------
    string
        truncated text
    """
    if isinstance(text, str) and len(text) > maxlen:
        tot_length = 0
        input_words = text.split()
        output_words = []
        for w in input_words[:-1]:
            tot_length = tot_length + len(w)
            if tot_length <= maxlen:
                output_words.append(w)

        text = " ".join(output_words)
        if len(input_words) > len(output_words):
            text = text + '...'
    return text

def compute_digit_number(value):
    """
    return int, number of digits to display

    Parameters
    ----------
    value : float
        can be the gap between percentiles

    Returns
    -------
    int
        number of digits
    """
    first_nz = int(math.log10(abs(value)))
    digit = abs(min(3,first_nz) - 3)
    return digit

def add_text(text_list,sep):
    """
    return int, number of digits to display

    Parameters
    ----------
    text_list : list
        list of text elements to concat
    sep: str
        separatator

    Returns
    -------
    int
        number of digits
    """
    clean_list = [x for x in text_list if x not in ['', None] ]
    return sep.join(clean_list)