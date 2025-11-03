import os
import pandas as pd

from scripts.train_spam_classifier import train_and_eval


def test_train_smoke(tmp_path):
    # create small dataset with enough samples per class for stratified split
    df = pd.DataFrame({
        "label": ["ham", "spam"] * 5,
        "text_clean": [
            "hello world",
            "win cash now",
            "how are you",
            "free prize",
            "good morning",
            "claim your prize",
            "see you soon",
            "lottery winner",
            "howdy friend",
            "free voucher"
        ]
    })
    in_path = tmp_path / "small.csv"
    out_dir = tmp_path / "models"
    df.to_csv(in_path, index=False)
    report = train_and_eval(df["text_clean"], df["label"], str(out_dir))
    # expect at least accuracy key
    assert isinstance(report, dict)
    assert os.path.exists(str(out_dir / "model.joblib"))
