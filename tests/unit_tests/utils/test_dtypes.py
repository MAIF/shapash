import pandas as pd

from shapash.utils.dtypes import is_text_like, text_like_columns


def test_is_text_like_object_string_is_true_in_strict_mode():
    s = pd.Series(["a", None, "b"], dtype=object)

    assert is_text_like(s, strict_object=True) is True


def test_is_text_like_object_mixed_is_false_in_strict_mode():
    s = pd.Series([["a"], {"b": 1}], dtype=object)

    assert is_text_like(s, strict_object=True) is False


def test_is_text_like_object_mixed_is_true_in_permissive_mode():
    s = pd.Series([["a"], {"b": 1}], dtype=object)

    assert is_text_like(s, strict_object=False) is True


def test_text_like_columns_strict_and_permissive_modes():
    df = pd.DataFrame(
        {
            "txt": pd.Series(["x", None], dtype=object),
            "mixed": pd.Series([["a"], {"b": 1}], dtype=object),
            "cat": pd.Series(["a", "b"], dtype="category"),
            "num": [1, 2],
        }
    )

    assert text_like_columns(df, strict_object=True) == ["txt", "cat"]
    assert text_like_columns(df, strict_object=False) == ["txt", "mixed", "cat"]
