"""Loader for the real CICIDS-2017 CSV files.

The real dataset is distributed by the Canadian Institute for
Cybersecurity (UNB) and consists of eight daily CSV files containing
CICFlowMeter v3 output for traffic captured between July 3 and 7,
2017.  This loader is a thin wrapper around `pandas.read_csv` that
normalises column names, casts numeric types, removes infinities,
and harmonises the Label column with the strings used by the
synthetic generator so the rest of the pipeline is data-source
agnostic.

The loader is purely additive: it does *not* attempt to download the
data on its behalf.  Place the daily CSV files under a directory of
your choice and pass that directory to `load_directory`.
"""

from __future__ import annotations

import glob
import os
from typing import Iterable, List

import numpy as np
import pandas as pd


# UNB CSVs sometimes ship with a leading space in column names; we
# normalise them.  We also rename the label column to the singular
# "Label" used elsewhere in the package.
_RENAME_MAP = {
    "Label": "Label",
    " Label": "Label",
}


def _read_one(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=_RENAME_MAP)
    # Replace inf/-inf with NaN then drop
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def load_directory(directory: str, glob_pattern: str = "*.csv") -> pd.DataFrame:
    """Load every CSV under `directory` matching `glob_pattern`."""
    paths = sorted(glob.glob(os.path.join(directory, glob_pattern)))
    if not paths:
        raise FileNotFoundError(f"No CSV files found under {directory}")
    frames: List[pd.DataFrame] = []
    for p in paths:
        frames.append(_read_one(p))
    df = pd.concat(frames, ignore_index=True)
    df = _harmonise_labels(df)
    if "Timestamp" not in df.columns:
        df["Timestamp"] = pd.to_datetime("2017-07-03 09:00:00") + pd.to_timedelta(np.arange(len(df)), unit="s")
    if "FlowID" not in df.columns:
        df.insert(0, "FlowID", np.arange(len(df), dtype=np.int64))
    return df


def _harmonise_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map vendor-specific label strings to the strings expected by SHIELD-AI."""
    mapping = {
        "BENIGN": "BENIGN",
        "DoS Hulk": "DoS Hulk",
        "DoS GoldenEye": "DoS GoldenEye",
        "DoS Slowhttptest": "DoS Slowhttptest",
        "DoS slowloris": "DoS slowloris",
        "DDoS": "DDoS",
        "PortScan": "PortScan",
        "FTP-Patator": "FTP-Patator",
        "SSH-Patator": "SSH-Patator",
        "Bot": "Bot",
        "Heartbleed": "Heartbleed",
        "Infiltration": "Infiltration",
        "Web Attack – Brute Force": "Web Attack - Brute Force",
        "Web Attack – XSS": "Web Attack - XSS",
        "Web Attack – Sql Injection": "Web Attack - SQL Injection",
        "Web Attack \xc3\xa2\xc2\x80\xc2\x93 Brute Force": "Web Attack - Brute Force",
        "Web Attack \xc3\xa2\xc2\x80\xc2\x93 XSS": "Web Attack - XSS",
        "Web Attack \xc3\xa2\xc2\x80\xc2\x93 Sql Injection": "Web Attack - SQL Injection",
    }
    df["Label"] = df["Label"].astype(str).str.strip().map(lambda v: mapping.get(v, v))
    return df
