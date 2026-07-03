"""Dataset loading and canonical-schema harmonization for the age classifier.

Design decision (see project notes):
    - TRAIN on FairFace only. FairFace is race-balanced by construction, so it
      stays our clean, balanced anchor for a fairness-critical task.
    - TEST on UTKFace and Adience as *held-out* datasets the model never saw.
      Strong cross-dataset generalization is a better responsible-AI story than
      simply training on more (imbalanced) rows.

Every dataset is translated into ONE canonical schema so the rest of the code
can treat them uniformly. We keep the raw labels alongside the canonical ones
(`original_*` columns) plus `dataset_source` and `license`, so every row is
fully traceable — this is the provenance/lineage idea from data management.

Canonical columns produced by every loader:
    filepath        absolute path to the image on disk
    age_bin         one of AGE_BINS (9 FairFace bins)
    gender          "Male" | "Female"
    race_fine       one of RACE_FINE (7 FairFace races) or "Unknown"
    race_coarse     one of RACE_COARSE — the 5-way scheme ALL datasets share;
                    use THIS for cross-dataset fairness comparison
    dataset_source  "fairface" | "utkface" | "adience"
    split_hint      dataset's own split, e.g. "train"/"val" (FairFace only)
    original_age    raw age label as it appeared in the source
    original_gender raw gender label
    original_race   raw race label ("" for Adience — no race label exists)
    license         data-use terms (matters for the regulatory section)

NOTE on lossiness (document this in the notebook, do NOT hide it):
    - UTKFace has 5 race classes; FairFace has 7. UTKFace "Asian" cannot be
      split into East/Southeast Asian, and "Others" bundles several groups.
      So UTKFace race is only reliable at the COARSE level. race_fine for
      UTKFace is a best-effort approximation and is flagged as such.
    - Adience has NO race label at all -> race_fine/race_coarse = "Unknown".
      Adience rows must be EXCLUDED from any race-stratified fairness metric.
    - Age bins do not line up perfectly across datasets; Adience ranges that
      straddle two FairFace bins are mapped by midpoint (see ADIENCE_AGE_MAP).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration — where the raw data lives.
# Override with the FACE_AGE_DATA_ROOT environment variable, or edit DATA_ROOT.
#
# Expected layout under DATA_ROOT:
#   data/
#     fairface/
#       fairface_label_train.csv
#       fairface_label_val.csv
#       train/ ...   val/ ...          (image folders referenced by the csv)
#     utkface/
#       UTKFace/ ...                    (jpgs named  age_gender_race_date.jpg)
#     adience/
#       fold_0_data.txt ... fold_4_data.txt
#       faces/ <user_id>/ ...           (aligned face images)
# --------------------------------------------------------------------------- #
DATA_ROOT = Path(os.environ.get(
    "FACE_AGE_DATA_ROOT",
    Path(__file__).resolve().parent.parent / "data",
))

# --------------------------------------------------------------------------- #
# Canonical vocabulary.
# --------------------------------------------------------------------------- #
AGE_BINS = ["0-2", "3-9", "10-19", "20-29", "30-39",
            "40-49", "50-59", "60-69", "70+"]

GENDERS = ["Male", "Female"]

# Fine race scheme = FairFace's 7 groups (+ "Unknown" for label-less rows).
RACE_FINE = ["White", "Black", "Latino_Hispanic", "East Asian",
             "Southeast Asian", "Indian", "Middle Eastern", "Unknown"]

# Coarse race scheme = the greatest common denominator across all 3 datasets.
# This is what cross-dataset fairness comparisons should use.
RACE_COARSE = ["White", "Black", "Asian", "Indian", "Other", "Unknown"]

# FairFace fine race -> coarse race.
_FAIRFACE_FINE_TO_COARSE = {
    "White": "White",
    "Black": "Black",
    "East Asian": "Asian",
    "Southeast Asian": "Asian",
    "Indian": "Indian",
    "Latino_Hispanic": "Other",
    "Middle Eastern": "Other",
}

LICENSES = {
    "fairface": "CC BY 4.0 (research use)",
    "utkface": "Non-commercial research use only",
    "adience": "Research use (Adience benchmark terms)",
}


# UTKFace filename: age_gender_race_datetime.jpg
#   ^(\d{1,3})  age    = 1-3 digits
#   ([01])      gender = 0 (Male) or 1 (Female) only
#   ([0-4])     race   = 0-4 only (White/Black/Asian/Indian/Others)
# Encoding the valid ranges in the pattern means bad values never match.
UTK_PATTERN = re.compile(r"^(\d{1,3})_([01])_([0-4])_")


# --------------------------------------------------------------------------- #
# Small mapping helpers.
# --------------------------------------------------------------------------- #
def age_int_to_bin(age: int) -> str | None:
    """Map an integer age to a canonical FairFace age bin."""
    try:
        age = int(age)
    except (TypeError, ValueError):
        return None
    if age < 0 or age > 120:
        return None
    if age <= 2:
        return "0-2"
    if age <= 9:
        return "3-9"
    if age <= 19:
        return "10-19"
    if age <= 29:
        return "20-29"
    if age <= 39:
        return "30-39"
    if age <= 49:
        return "40-49"
    if age <= 59:
        return "50-59"
    if age <= 69:
        return "60-69"
    return "70+"


def fairface_age_to_bin(raw: str) -> str | None:
    """FairFace already uses these bins; only 'more than 70' needs renaming."""
    raw = str(raw).strip()
    if raw in ("more than 70", "70+", "70 plus"):
        return "70+"
    return raw if raw in AGE_BINS else None


# Adience labels age as a range string. Several ranges straddle two FairFace
# bins; we resolve those by midpoint and flag the approximation here so it is
# explicit rather than silent.  (approx=True -> note it in the model card)
ADIENCE_AGE_MAP = {
    "(0, 2)": ("0-2", False),
    "(4, 6)": ("3-9", False),
    "(8, 12)": ("10-19", True),    # straddles 3-9 / 10-19; midpoint 10
    "(8, 23)": ("10-19", True),    # wide; midpoint ~15
    "(15, 20)": ("10-19", True),   # straddles 10-19 / 20-29; midpoint 17.5
    "(25, 32)": ("20-29", True),   # straddles 20-29 / 30-39; midpoint 28.5
    "(27, 32)": ("20-29", True),
    "(38, 43)": ("40-49", True),   # straddles 30-39 / 40-49; midpoint 40.5
    "(38, 42)": ("40-49", True),
    "(38, 48)": ("40-49", True),
    "(48, 53)": ("50-59", True),   # straddles 40-49 / 50-59; midpoint 50.5
    "(60, 100)": ("70+", True),    # very wide; midpoint -> 70+
}


def adience_age_to_bin(raw: str) -> tuple[str | None, bool]:
    """Map an Adience age label to (canonical_bin, is_approximate)."""
    raw = str(raw).strip()
    if raw in ADIENCE_AGE_MAP:
        return ADIENCE_AGE_MAP[raw]
    # Some rows carry a bare integer instead of a range.
    b = age_int_to_bin(raw)
    return (b, False) if b else (None, False)


def _empty_frame() -> pd.DataFrame:
    """A zero-row DataFrame with the canonical columns, for graceful failures."""
    cols = ["filepath", "age_bin", "gender", "race_fine", "race_coarse",
            "dataset_source", "split_hint", "original_age", "original_gender",
            "original_race", "license"]
    return pd.DataFrame(columns=cols)


# --------------------------------------------------------------------------- #
# Loaders — one per dataset. Each returns a canonical DataFrame (possibly empty
# if the data is not present yet). None of them raise on missing data; they log
# and return an empty frame so the module is runnable before you download.
# --------------------------------------------------------------------------- #
def load_fairface(root: Path | None = None) -> pd.DataFrame:
    """Load FairFace (the training anchor) into the canonical schema."""
    root = Path(root) if root else DATA_ROOT / "fairface"
    if not root.exists():
        logger.warning("FairFace not found at %s — skipping.", root)
        return _empty_frame()

    frames = []
    for split, csv_name in [("train", "fairface_label_train.csv"),
                            ("val", "fairface_label_val.csv")]:
        csv_path = root / csv_name
        if not csv_path.exists():
            logger.warning("FairFace csv missing: %s", csv_path)
            continue
        df = pd.read_csv(csv_path)
        # FairFace csv columns: file, age, gender, race, service_test
        out = pd.DataFrame()
        out["filepath"] = df["file"].apply(lambda f: str((root / f).resolve()))
        out["age_bin"] = df["age"].apply(fairface_age_to_bin)
        out["gender"] = df["gender"].apply(
            lambda g: g if g in GENDERS else None)
        out["race_fine"] = df["race"].apply(
            lambda r: r if r in RACE_FINE else None)
        out["race_coarse"] = out["race_fine"].map(_FAIRFACE_FINE_TO_COARSE)
        out["dataset_source"] = "fairface"
        out["split_hint"] = split
        out["original_age"] = df["age"].astype(str)
        out["original_gender"] = df["gender"].astype(str)
        out["original_race"] = df["race"].astype(str)
        out["license"] = LICENSES["fairface"]
        frames.append(out)

    if not frames:
        return _empty_frame()
    df = pd.concat(frames, ignore_index=True)
    return _clean(df, "fairface")


def load_utkface(root: Path | None = None) -> pd.DataFrame:
    """Load UTKFace (held-out test) into the canonical schema.

    Filenames encode labels as:  age_gender_race_datetime.jpg
        gender: 0=Male, 1=Female
        race:   0=White 1=Black 2=Asian 3=Indian 4=Others
    Malformed / truncated filenames are skipped and counted.
    """
    root = Path(root) if root else DATA_ROOT / "utkface"
    if not root.exists():
        logger.warning("UTKFace not found at %s — skipping.", root)
        return _empty_frame()

    # UTKFace ships under a few possible subfolder names; search them all.
    jpgs = list(root.rglob("*.jpg"))
    if not jpgs:
        logger.warning("UTKFace: no .jpg files under %s", root)
        return _empty_frame()

    gender_map = {"0": "Male", "1": "Female"}
    # (race_fine approximation, race_coarse) — see module docstring on lossiness.
    race_map = {
        "0": ("White", "White"),
        "1": ("Black", "Black"),
        "2": ("East Asian", "Asian"),   # fine label is APPROXIMATE
        "3": ("Indian", "Indian"),
        "4": ("Unknown", "Other"),      # "Others" -> cannot pin a fine race
    }

    rows, skipped = [], 0
    for p in jpgs:
        # Filename encodes labels as  age_gender_race_datetime.jpg
        # The pattern validates the shape while extracting: age is 1-3 digits,
        # gender is only 0/1, race is only 0-4 — anything else fails to match
        # and is skipped (this also catches UTKFace's known malformed names,
        # e.g. files missing the race field).
        m = UTK_PATTERN.match(p.stem)
        if not m:
            skipped += 1
            continue
        raw_age, raw_gender, raw_race = m.groups()
        age_bin = age_int_to_bin(raw_age)
        gender = gender_map.get(raw_gender)
        race_fine, race_coarse = race_map.get(raw_race, (None, None))
        if age_bin is None or gender is None or race_coarse is None:
            skipped += 1
            continue
        rows.append({
            "filepath": str(p.resolve()),
            "age_bin": age_bin,
            "gender": gender,
            "race_fine": race_fine,
            "race_coarse": race_coarse,
            "dataset_source": "utkface",
            "split_hint": "test",
            "original_age": raw_age,
            "original_gender": raw_gender,
            "original_race": raw_race,
            "license": LICENSES["utkface"],
        })
    if skipped:
        logger.info("UTKFace: skipped %d files with malformed labels.", skipped)
    if not rows:
        return _empty_frame()
    return _clean(pd.DataFrame(rows), "utkface")


def load_adience(root: Path | None = None) -> pd.DataFrame:
    """Load Adience (held-out test) into the canonical schema.

    Adience has age + gender but NO race, so race_fine/race_coarse = "Unknown"
    and these rows must be excluded from race-stratified fairness metrics.
    Labels live in tab-separated fold files (fold_0_data.txt ...).
    """
    root = Path(root) if root else DATA_ROOT / "adience"
    if not root.exists():
        logger.warning("Adience not found at %s — skipping.", root)
        return _empty_frame()

    fold_files = sorted(root.glob("fold_*_data.txt"))
    if not fold_files:
        logger.warning("Adience: no fold_*_data.txt files under %s", root)
        return _empty_frame()

    gender_map = {"m": "Male", "f": "Female"}
    faces_dir = root / "faces"

    rows, skipped, missing_img = [], 0, 0
    for ff in fold_files:
        df = pd.read_csv(ff, sep="\t")
        for _, r in df.iterrows():
            age_bin, _approx = adience_age_to_bin(r.get("age"))
            gender = gender_map.get(str(r.get("gender")).strip().lower())
            if age_bin is None or gender is None:
                skipped += 1
                continue
            # Aligned face filename convention used by the Adience release.
            user_id = str(r.get("user_id"))
            face_id = str(r.get("face_id"))
            orig = str(r.get("original_image"))
            aligned_name = f"coarse_tilt_aligned_face.{face_id}.{orig}"
            img_path = faces_dir / user_id / aligned_name
            if not img_path.exists():
                missing_img += 1
                continue
            rows.append({
                "filepath": str(img_path.resolve()),
                "age_bin": age_bin,
                "gender": gender,
                "race_fine": "Unknown",
                "race_coarse": "Unknown",
                "dataset_source": "adience",
                "split_hint": "test",
                "original_age": str(r.get("age")),
                "original_gender": str(r.get("gender")),
                "original_race": "",  # no race label exists in Adience
                "license": LICENSES["adience"],
            })
    if skipped:
        logger.info("Adience: skipped %d rows (unusable age/gender).", skipped)
    if missing_img:
        logger.info("Adience: %d labelled rows had no matching image on disk.",
                    missing_img)
    if not rows:
        return _empty_frame()
    return _clean(pd.DataFrame(rows), "adience")


# --------------------------------------------------------------------------- #
# Shared cleanup + convenience entry points.
# --------------------------------------------------------------------------- #
def _clean(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Drop rows with an unresolved canonical label and log how many."""
    before = len(df)
    df = df.dropna(subset=["age_bin", "gender", "race_coarse"]).reset_index(
        drop=True)
    dropped = before - len(df)
    if dropped:
        logger.info("%s: dropped %d rows with unmapped labels.", source, dropped)
    logger.info("%s: %d usable rows.", source, len(df))
    return df


def get_train_test(data_root: Path | None = None):
    """Return (train_df, test_df) per the project design.

    train_df = FairFace.
    test_df  = UTKFace + Adience, kept together but tagged by dataset_source
               so you can slice per-source at evaluation time.
    """
    global DATA_ROOT
    if data_root is not None:
        DATA_ROOT = Path(data_root)

    train_df = load_fairface()
    utk = load_utkface()
    adi = load_adience()
    test_df = pd.concat([utk, adi], ignore_index=True) if len(utk) or len(adi) \
        else _empty_frame()
    return train_df, test_df


def summarize(df: pd.DataFrame, by: str = "race_coarse") -> pd.DataFrame:
    """Quick EDA: cross-tab of age_bin vs a protected attribute.

    Rows with an unknown protected attribute (e.g. Adience race) are kept
    visible as their own column so the gap is obvious, not hidden.
    """
    if df.empty:
        print("(empty frame — no data loaded yet)")
        return df
    print(f"total rows: {len(df):,}")
    print(f"by dataset_source:\n{df['dataset_source'].value_counts()}\n")
    ct = pd.crosstab(df["age_bin"], df[by]).reindex(AGE_BINS)
    print(f"age_bin x {by}:\n{ct}\n")
    return ct


if __name__ == "__main__":
    # Runnable smoke test: reports what is present under DATA_ROOT.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(f"DATA_ROOT = {DATA_ROOT}\n")
    train_df, test_df = get_train_test()
    print("\n=== TRAIN (FairFace) ===")
    summarize(train_df, by="race_coarse")
    print("=== TEST (UTKFace + Adience) ===")
    summarize(test_df, by="dataset_source")
