import pandas as pd

from scripts.preprocess_emails import clean_text, preprocess_df


def test_clean_text_basic():
    s = "Free entry in 2 a wkly comp to win $$$"
    out = clean_text(s)
    assert "free" in out
    assert "$" not in out


def test_preprocess_df_creates_text_clean():
    df = pd.DataFrame({"label": ["ham", "spam"], "msg": ["Hello there!", "WIN cash now!!!"]})
    out = preprocess_df(df, "msg")
    assert "text_clean" in out.columns
    assert out.loc[0, "text_clean"] == "hello there"
