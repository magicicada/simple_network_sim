import pandas as pd
import pytest

from . import comix_downsampler


def test_collapse_empty_names():
    with pytest.raises(ValueError):
        comix_downsampler.collapse_columns(
            comix_downsampler.ContactsTable(pd.DataFrame({"a": [1, 2], "b": [4, 5]}, index=list("ab"))), [], "a'",
        )


def test_collapse_column_mismatch():
    with pytest.raises(ValueError):
        comix_downsampler.collapse_columns(
            comix_downsampler.ContactsTable(pd.DataFrame({"a": [1, 2], "b": [4, 5]}, index=list("ab"))),
            ["a", "b", "c"],
            "a'",
        )


def test_collapse_index_column_mismatch():
    with pytest.raises(ValueError):
        comix_downsampler.collapse_columns(
            comix_downsampler.ContactsTable(pd.DataFrame({"a": [1, 2], "b": [4, 5]}, index=list("ac"))),
            ["a", "b"],
            "a'",
        )


def test_collapse_left():
    df = comix_downsampler.ContactsTable(
        pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, index=list("abc"))
    )

    collapsed = comix_downsampler.collapse_columns(df, ["a", "b"], "a'")

    pd.testing.assert_frame_equal(collapsed, pd.DataFrame({"a'": [12, 9], "c": [15, 9]}, index=["a'", "c"]))


def test_collapse_right():
    df = comix_downsampler.ContactsTable(
        pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, index=list("abc"))
    )

    collapsed = comix_downsampler.collapse_columns(df, ["b", "c"], "c'")

    pd.testing.assert_frame_equal(collapsed, pd.DataFrame({"a": [1, 5], "c'": [11, 28]}, index=["a", "c'"]))


def test_split_17_year_olds():
    pop = pd.Series(range(1, 14), index=range(5, 18))
    contacts = pd.DataFrame({"[5,18)": [10, 0], "[18,30)": [12, 17]}, index=["[5,18)", "[18,30)"])

    df = comix_downsampler.split_17_years_old(comix_downsampler.ContactsTable(contacts), pop)

    expected = pd.DataFrame(
        {"[5,17)": [8.28947, 0, 0], "17": [1.41794, 0.29259, 0], "[18,30)": [9.94737, 2.05263, 17.0]},
        index=["[5,17)", "17", "[18,30)"],
    )
    pd.testing.assert_frame_equal(df, expected)


def test_split_collapse_sanity_check():
    pop = pd.Series(range(1, 14), index=range(5, 18))
    original = pd.DataFrame({"[5,18)": [10.0, 0.0], "[18,30)": [12.0, 17.0]}, index=["[5,18)", "[18,30)"])

    new_contacts = comix_downsampler.split_17_years_old(comix_downsampler.ContactsTable(original), pop)
    contacts = comix_downsampler.collapse_columns(new_contacts, ["[5,17)", "17"], "[5,18)")

    pd.testing.assert_frame_equal(contacts, original)


def test_flatten():
    df = pd.DataFrame({"[5,18)": [2.7, 0.89], "[18,30)": [1.19, 0.52]}, index=["[5,18)", "[18,30)"])
    flattened = comix_downsampler._flatten(df)

    expected = pd.DataFrame(
        [
            {"source": "[5,18)", "target": "[5,18)", "mixing": 2.7},
            {"source": "[5,18)", "target": "[18,30)", "mixing": 1.19},
            {"source": "[18,30)", "target": "[5,18)", "mixing": 0.89},
            {"source": "[18,30)", "target": "[18,30)", "mixing": 0.52},
        ]
    )
    pd.testing.assert_frame_equal(flattened, expected)
