import os
import re
import gc
import math
import warnings
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

from rapidfuzz.fuzz import (
    ratio,
    WRatio,
    token_sort_ratio,
    token_set_ratio,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from xgboost import XGBClassifier

from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42

# Maximum candidates retained for one S1 entity
MAX_CANDIDATES = 80

# Number of negative examples sampled per positive
NEGATIVE_RATIO = 3

# Final prediction threshold
DEFAULT_THRESHOLD = 0.60


# ============================================================
# FILE LOADING
# ============================================================

def load_train():
    print("\nLoading training data...")

    s1 = pd.read_csv(
        os.path.join(TRAIN_DIR, "train_source1.tsv"),
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    s2 = pd.read_csv(
        os.path.join(TRAIN_DIR, "train_source2.tsv"),
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    s3 = pd.read_csv(
        os.path.join(TRAIN_DIR, "train_source3.tsv"),
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    gt = pd.read_csv(
        os.path.join(TRAIN_DIR, "train_ground_truth.tsv"),
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    return s1, s2, s3, gt


def load_test():
    print("\nLoading test data...")

    s1 = pd.read_csv(
        os.path.join(TEST_DIR, "test_source1.tsv"),
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    s2 = pd.read_csv(
        os.path.join(TEST_DIR, "test_source2.tsv"),
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    s3 = pd.read_csv(
        os.path.join(TEST_DIR, "test_source3.tsv"),
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    return s1, s2, s3


# ============================================================
# TEXT NORMALIZATION
# ============================================================

ABBREVIATIONS = {
    "corporation": "corp",
    "company": "co",
    "limited": "ltd",
    "private": "pvt",
    "incorporated": "inc",
    "road": "rd",
    "street": "st",
    "avenue": "ave",
    "boulevard": "blvd",
    "highway": "hwy",
    "drive": "dr",
    "lane": "ln",
    "apartment": "apt",
    "building": "bldg",
}


def normalize_text(text):
    """
    Generic normalization.

    Important:
    - lowercases
    - removes punctuation
    - normalizes whitespace
    - expands selected common abbreviations
    """

    if text is None:
        return ""

    text = str(text).lower().strip()

    if not text:
        return ""

    # Unicode normalization
    import unicodedata

    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    # Replace & with and
    text = text.replace("&", " and ")

    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Whitespace
    text = re.sub(r"\s+", " ", text).strip()

    words = []

    for word in text.split():
        words.append(ABBREVIATIONS.get(word, word))

    return " ".join(words)


def normalize_name(text):
    return normalize_text(text)


def normalize_address(text):
    return normalize_text(text)


def add_normalized_columns(df):
    df = df.copy()

    df["name_norm"] = df["business_name"].map(normalize_name)
    df["address_norm"] = df["business_address"].map(normalize_address)

    df["country_norm"] = (
        df["country"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    return df


# ============================================================
# BLOCKING KEYS
# ============================================================

def first_chars(s, n):
    if not s:
        return ""
    return s[:n]


def name_tokens(s):
    if not s:
        return []
    return s.split()


def get_digits(s):
    return "".join(re.findall(r"\d+", s))


def get_postal_code(s):
    """
    Extract useful numeric blocks from address.

    We don't assume US/India/France formats.
    """

    nums = re.findall(r"\b\d{4,10}\b", s)

    if nums:
        return nums[-1]

    return ""


def blocking_keys(row):
    """
    Generate several independent blocking keys.

    Multiple blocks are used because no single key should be trusted.
    """

    name = row["name_norm"]
    address = row["address_norm"]
    country = row["country_norm"]

    keys = []

    # Exact normalized name
    if name:
        keys.append(("name_exact", country, name))

    # Name prefixes
    if len(name) >= 3:
        keys.append(("name_prefix3", country, name[:3]))

    if len(name) >= 5:
        keys.append(("name_prefix5", country, name[:5]))

    # First / second name token
    tokens = name.split()

    if tokens:
        keys.append(("name_token0", country, tokens[0]))

    if len(tokens) >= 2:
        keys.append(("name_token01", country, tokens[0] + "|" + tokens[1]))

    # Address exact
    if address:
        keys.append(("addr_exact", country, address))

    # Address prefix
    if len(address) >= 5:
        keys.append(("addr_prefix5", country, address[:5]))

    # Postal/numeric block
    postal = get_postal_code(address)

    if postal:
        keys.append(("postal", country, postal))

    # Combined name + address prefix
    if name and address:
        keys.append(
            (
                "name_addr",
                country,
                name[:5] + "|" + address[:8]
            )
        )

    return keys


# ============================================================
# BUILD BLOCK INDEX
# ============================================================

def build_block_index(df):
    """
    Maps blocking keys -> record indices.

    Uses source-local indices.
    """

    index = defaultdict(list)

    for idx, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Building blocking index"
    ):
        for key in blocking_keys(row):
            index[key].append(idx)

    return index


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_candidates_for_row(
    row,
    source_df,
    block_index,
    max_candidates=MAX_CANDIDATES
):

    candidate_indices = set()

    keys = blocking_keys(row)

    for key in keys:

        ids = block_index.get(key, [])

        if len(ids) <= max_candidates:

            candidate_indices.update(ids)

        else:

            # Don't explode on extremely common blocks.
            # Prefer candidates that share stronger keys.
            candidate_indices.update(ids[:max_candidates])

    # If nothing found, use country-level limited candidates.
    if not candidate_indices:

        country = row["country_norm"]

        same_country = source_df.index[
            source_df["country_norm"] == country
        ].tolist()

        if len(same_country) <= max_candidates:
            candidate_indices.update(same_country)
        else:
            candidate_indices.update(
                same_country[:max_candidates]
            )

    # Limit
    if len(candidate_indices) > max_candidates:

        candidate_indices = set(
            list(candidate_indices)[:max_candidates]
        )

    return candidate_indices


def generate_candidates(
    s1,
    s2,
    s3,
    save_path=None
):

    print("\nBuilding blocking indexes...")

    index2 = build_block_index(s2)
    index3 = build_block_index(s3)

    candidate_rows = []

    for _, row in tqdm(
        s1.iterrows(),
        total=len(s1),
        desc="Generating candidates"
    ):

        c2 = generate_candidates_for_row(
            row,
            s2,
            index2
        )

        c3 = generate_candidates_for_row(
            row,
            s3,
            index3
        )

        ids = []

        for idx in c2:
            ids.append(s2.loc[idx, "entity_id"])

        for idx in c3:
            ids.append(s3.loc[idx, "entity_id"])

        ids = list(dict.fromkeys(ids))

        candidate_rows.append({
            "source1_entity_id": row["entity_id"],
            "candidate_entity_ids": ",".join(ids)
        })

    result = pd.DataFrame(candidate_rows)

    if save_path:
        result.to_csv(
            save_path,
            sep="\t",
            index=False
        )

    return result, index2, index3


# ============================================================
# FAST LOOKUPS
# ============================================================

def create_lookup(df):

    return df.set_index("entity_id").to_dict("index")


# ============================================================
# STRING FEATURES
# ============================================================

def safe_ratio(a, b):

    if not a or not b:
        return 0.0

    return ratio(a, b) / 100.0


def safe_wratio(a, b):

    if not a or not b:
        return 0.0

    return WRatio(a, b) / 100.0


def safe_token_sort(a, b):

    if not a or not b:
        return 0.0

    return token_sort_ratio(a, b) / 100.0


def safe_token_set(a, b):

    if not a or not b:
        return 0.0

    return token_set_ratio(a, b) / 100.0


def jaccard_tokens(a, b):

    if not a or not b:
        return 0.0

    A = set(a.split())
    B = set(b.split())

    if not A or not B:
        return 0.0

    return len(A & B) / len(A | B)


def containment(a, b):

    if not a or not b:
        return 0.0

    A = set(a.split())
    B = set(b.split())

    if not A or not B:
        return 0.0

    return max(
        len(A & B) / len(A),
        len(A & B) / len(B)
    )


def digit_similarity(a, b):

    da = get_digits(a)
    db = get_digits(b)

    if not da or not db:
        return 0.0

    if da == db:
        return 1.0

    if da in db or db in da:
        return 0.7

    return 0.0


# ============================================================
# FEATURE EXTRACTION
# ============================================================

FEATURE_COLUMNS = [
    "name_ratio",
    "name_wratio",
    "name_token_sort",
    "name_token_set",
    "name_jaccard",
    "name_containment",

    "address_ratio",
    "address_wratio",
    "address_token_sort",
    "address_token_set",
    "address_jaccard",
    "address_containment",

    "digit_similarity",

    "country_match",

    "name_exact",
    "address_exact",

    "name_length_diff",
    "address_length_diff",
]


def make_features(a, b):

    name_a = a["name_norm"]
    name_b = b["name_norm"]

    addr_a = a["address_norm"]
    addr_b = b["address_norm"]

    country_a = a["country_norm"]
    country_b = b["country_norm"]

    features = {

        "name_ratio":
            safe_ratio(name_a, name_b),

        "name_wratio":
            safe_wratio(name_a, name_b),

        "name_token_sort":
            safe_token_sort(name_a, name_b),

        "name_token_set":
            safe_token_set(name_a, name_b),

        "name_jaccard":
            jaccard_tokens(name_a, name_b),

        "name_containment":
            containment(name_a, name_b),

        "address_ratio":
            safe_ratio(addr_a, addr_b),

        "address_wratio":
            safe_wratio(addr_a, addr_b),

        "address_token_sort":
            safe_token_sort(addr_a, addr_b),

        "address_token_set":
            safe_token_set(addr_a, addr_b),

        "address_jaccard":
            jaccard_tokens(addr_a, addr_b),

        "address_containment":
            containment(addr_a, addr_b),

        "digit_similarity":
            digit_similarity(addr_a, addr_b),

        "country_match":
            float(country_a == country_b),

        "name_exact":
            float(
                bool(name_a) and
                name_a == name_b
            ),

        "address_exact":
            float(
                bool(addr_a) and
                addr_a == addr_b
            ),

        "name_length_diff":
            abs(len(name_a) - len(name_b)),

        "address_length_diff":
            abs(len(addr_a) - len(addr_b)),
    }

    return features


# ============================================================
# GROUND TRUTH
# ============================================================

def build_ground_truth(gt):

    truth = {}

    for _, row in gt.iterrows():

        s1_id = row["source1_entity_id"]

        value = row["matched_entity_ids"]

        if not value:
            truth[s1_id] = set()

        else:
            truth[s1_id] = set(
                x.strip()
                for x in str(value).split(",")
                if x.strip()
            )

    return truth


# ============================================================
# TRAINING DATA CREATION
# ============================================================

def create_training_pairs(
    s1,
    s2,
    s3,
    candidates,
    gt
):

    print("\nCreating training pairs...")

    s2_lookup = create_lookup(s2)
    s3_lookup = create_lookup(s3)

    truth = build_ground_truth(gt)

    X_rows = []
    y_rows = []

    rng = np.random.default_rng(RANDOM_STATE)

    for _, crow in tqdm(
        candidates.iterrows(),
        total=len(candidates),
        desc="Creating training features"
    ):

        s1_id = crow["source1_entity_id"]

        if not crow["candidate_entity_ids"]:
            continue

        source1_record = s1[
            s1["entity_id"] == s1_id
        ]

        if source1_record.empty:
            continue

        source1_record = source1_record.iloc[0]

        candidate_ids = crow[
            "candidate_entity_ids"
        ].split(",")

        true_ids = truth.get(s1_id, set())

        positives = []
        negatives = []

        for cid in candidate_ids:

            if cid in true_ids:
                positives.append(cid)
            else:
                negatives.append(cid)

        # Add positives
        for cid in positives:

            if cid.startswith("S2-"):
                record = s2_lookup.get(cid)
            else:
                record = s3_lookup.get(cid)

            if record is None:
                continue

            f = make_features(
                source1_record,
                record
            )

            X_rows.append(f)
            y_rows.append(1)

        # Sample negatives
        if negatives:

            n_neg = min(
                len(negatives),
                max(
                    3,
                    len(positives) * NEGATIVE_RATIO
                )
            )

            if len(negatives) > n_neg:

                negatives = rng.choice(
                    negatives,
                    size=n_neg,
                    replace=False
                )

            for cid in negatives:

                if cid.startswith("S2-"):
                    record = s2_lookup.get(cid)
                else:
                    record = s3_lookup.get(cid)

                if record is None:
                    continue

                f = make_features(
                    source1_record,
                    record
                )

                X_rows.append(f)
                y_rows.append(0)

    X = pd.DataFrame(
        X_rows,
        columns=FEATURE_COLUMNS
    )

    y = np.array(y_rows)

    print("\nTraining examples:", len(X))
    print("Positive:", int(y.sum()))
    print("Negative:", int((y == 0).sum()))

    return X, y


# ============================================================
# F0.5
# ============================================================

def f05(precision, recall):

    denominator = (
        0.25 * precision +
        recall
    )

    if denominator == 0:
        return 0.0

    return (
        1.25 *
        precision *
        recall /
        denominator
    )


def macro_f05(
    predictions,
    truth,
    s1_ids
):

    scores = []

    for s1_id, pred in zip(
        s1_ids,
        predictions
    ):

        pred = set(pred)
        actual = truth.get(s1_id, set())

        if not pred and not actual:
            scores.append(1.0)
            continue

        if not pred and actual:
            scores.append(0.0)
            continue

        if pred and not actual:
            scores.append(0.0)
            continue

        tp = len(pred & actual)

        precision = tp / len(pred)
        recall = tp / len(actual)

        scores.append(
            f05(precision, recall)
        )

    return float(np.mean(scores))


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(X, y):

    print("\nTraining XGBoost model...")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = XGBClassifier(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    model_path = os.path.join(
        MODEL_DIR,
        "entity_resolution_xgb.json"
    )

    model.save_model(model_path)

    print("Model saved:", model_path)

    return model


# ============================================================
# SCORE CANDIDATES
# ============================================================

def score_candidates(
    s1,
    s2,
    s3,
    candidates,
    model
):

    s2_lookup = create_lookup(s2)
    s3_lookup = create_lookup(s3)

    all_predictions = {}

    for _, row in tqdm(
        candidates.iterrows(),
        total=len(candidates),
        desc="Scoring candidates"
    ):

        s1_id = row["source1_entity_id"]

        if not row["candidate_entity_ids"]:
            all_predictions[s1_id] = []
            continue

        source1_record = s1[
            s1["entity_id"] == s1_id
        ]

        if source1_record.empty:
            all_predictions[s1_id] = []
            continue

        source1_record = source1_record.iloc[0]

        candidate_ids = row[
            "candidate_entity_ids"
        ].split(",")

        feature_rows = []
        valid_ids = []

        for cid in candidate_ids:

            if cid.startswith("S2-"):
                record = s2_lookup.get(cid)
            else:
                record = s3_lookup.get(cid)

            if record is None:
                continue

            feature_rows.append(
                make_features(
                    source1_record,
                    record
                )
            )

            valid_ids.append(cid)

        if not feature_rows:
            all_predictions[s1_id] = []
            continue

        X = pd.DataFrame(
            feature_rows,
            columns=FEATURE_COLUMNS
        )

        probabilities = model.predict_proba(X)[:, 1]

        scored = list(
            zip(
                valid_ids,
                probabilities
            )
        )

        scored.sort(
            key=lambda x: x[1],
            reverse=True
        )

        # Conservative selection.
        selected = [
            cid
            for cid, probability in scored
            if probability >= DEFAULT_THRESHOLD
        ]

        all_predictions[s1_id] = selected

    return all_predictions


# ============================================================
# WRITE OUTPUT
# ============================================================

def write_matching_results(
    s1,
    predictions
):

    rows = []

    for s1_id in s1["entity_id"]:

        matches = predictions.get(
            s1_id,
            []
        )

        rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(matches)
        })

    result = pd.DataFrame(rows)

    path = os.path.join(
        OUTPUT_DIR,
        "matching_results.tsv"
    )

    result.to_csv(
        path,
        sep="\t",
        index=False
    )

    print("\nSaved:", path)

    return result


def write_candidate_pairs(candidates):

    path = os.path.join(
        OUTPUT_DIR,
        "candidate_pairs.tsv"
    )

    candidates.to_csv(
        path,
        sep="\t",
        index=False
    )

    print("Saved:", path)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026")
    print("BUSINESS ENTITY RESOLUTION")
    print("=" * 70)

    # --------------------------------------------------------
    # TRAIN DATA
    # --------------------------------------------------------

    train_s1, train_s2, train_s3, gt = load_train()

    print("\nTRAIN SHAPES")

    print("S1:", train_s1.shape)
    print("S2:", train_s2.shape)
    print("S3:", train_s3.shape)
    print("GT:", gt.shape)

    # Normalize
    print("\nNormalizing training data...")

    train_s1 = add_normalized_columns(train_s1)
    train_s2 = add_normalized_columns(train_s2)
    train_s3 = add_normalized_columns(train_s3)

    # --------------------------------------------------------
    # TRAIN CANDIDATES
    # --------------------------------------------------------

    train_candidate_path = os.path.join(
        OUTPUT_DIR,
        "train_candidate_pairs.tsv"
    )

    train_candidates, _, _ = generate_candidates(
        train_s1,
        train_s2,
        train_s3,
        save_path=train_candidate_path
    )

    # --------------------------------------------------------
    # TRAIN FEATURES
    # --------------------------------------------------------

    X, y = create_training_pairs(
        train_s1,
        train_s2,
        train_s3,
        train_candidates,
        gt
    )

    if len(X) == 0:
        raise RuntimeError(
            "No training pairs were generated. "
            "Blocking strategy needs adjustment."
        )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = train_model(X, y)

    # --------------------------------------------------------
    # TEST DATA
    # --------------------------------------------------------

    test_s1, test_s2, test_s3 = load_test()

    print("\nTEST SHAPES")

    print("S1:", test_s1.shape)
    print("S2:", test_s2.shape)
    print("S3:", test_s3.shape)

    print("\nNormalizing test data...")

    test_s1 = add_normalized_columns(test_s1)
    test_s2 = add_normalized_columns(test_s2)
    test_s3 = add_normalized_columns(test_s3)

    # --------------------------------------------------------
    # TEST CANDIDATES
    # --------------------------------------------------------

    candidate_path = os.path.join(
        OUTPUT_DIR,
        "candidate_pairs.tsv"
    )

    test_candidates, _, _ = generate_candidates(
        test_s1,
        test_s2,
        test_s3,
        save_path=candidate_path
    )

    # --------------------------------------------------------
    # FINAL PREDICTION
    # --------------------------------------------------------

    predictions = score_candidates(
        test_s1,
        test_s2,
        test_s3,
        test_candidates,
        model
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    write_matching_results(
        test_s1,
        predictions
    )

    write_candidate_pairs(
        test_candidates
    )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print("\nFinal files:")

    print(
        os.path.join(
            OUTPUT_DIR,
            "matching_results.tsv"
        )
    )

    print(
        os.path.join(
            OUTPUT_DIR,
            "candidate_pairs.tsv"
        )
    )


if __name__ == "__main__":
    main()