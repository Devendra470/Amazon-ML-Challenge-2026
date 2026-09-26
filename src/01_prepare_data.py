import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_tsv(path):
    print(f"\nLoading:")
    print(path)

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    return df


def save_parquet(df, filename):
    path = os.path.join(
        PROCESSED_DIR,
        filename
    )

    df.to_parquet(
        path,
        index=False
    )

    print("Saved:", path)
    print("Size:", round(os.path.getsize(path) / (1024**2), 2), "MB")


print("=" * 70)
print("AMAZON ML CHALLENGE 2026")
print("STEP 1 — DATA PREPARATION")
print("=" * 70)


# ------------------------------------------------------------
# TRAINING DATA
# ------------------------------------------------------------

print("\nTRAIN SOURCE 1")

train_s1 = load_tsv(
    os.path.join(
        TRAIN_DIR,
        "train_source1.tsv"
    )
)

print("\nTRAIN SOURCE 2")

train_s2 = load_tsv(
    os.path.join(
        TRAIN_DIR,
        "train_source2.tsv"
    )
)

print("\nTRAIN SOURCE 3")

train_s3 = load_tsv(
    os.path.join(
        TRAIN_DIR,
        "train_source3.tsv"
    )
)

print("\nGROUND TRUTH")

ground_truth = load_tsv(
    os.path.join(
        TRAIN_DIR,
        "train_ground_truth.tsv"
    )
)


# ------------------------------------------------------------
# TEST DATA
# ------------------------------------------------------------

print("\nTEST SOURCE 1")

test_s1 = load_tsv(
    os.path.join(
        TEST_DIR,
        "test_source1.tsv"
    )
)

print("\nTEST SOURCE 2")

test_s2 = load_tsv(
    os.path.join(
        TEST_DIR,
        "test_source2.tsv"
    )
)

print("\nTEST SOURCE 3")

test_s3 = load_tsv(
    os.path.join(
        TEST_DIR,
        "test_source3.tsv"
    )
)


# ------------------------------------------------------------
# VALIDATE COLUMNS
# ------------------------------------------------------------

required_columns = [
    "entity_id",
    "business_name",
    "business_address",
    "country"
]

for name, df in [
    ("train_s1", train_s1),
    ("train_s2", train_s2),
    ("train_s3", train_s3),
    ("test_s1", test_s1),
    ("test_s2", test_s2),
    ("test_s3", test_s3),
]:

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing columns: {missing}"
        )

    print(f"✓ {name} columns OK")


# ------------------------------------------------------------
# CHECK DUPLICATES
# ------------------------------------------------------------

print("\nChecking entity IDs...")

for name, df in [
    ("train_s1", train_s1),
    ("train_s2", train_s2),
    ("train_s3", train_s3),
]:

    duplicate_count = df["entity_id"].duplicated().sum()

    print(
        f"{name}: "
        f"{duplicate_count:,} duplicate entity IDs"
    )


# ------------------------------------------------------------
# SAVE AS PARQUET
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SAVING PARQUET FILES")
print("=" * 70)

save_parquet(
    train_s1,
    "train_s1.parquet"
)

save_parquet(
    train_s2,
    "train_s2.parquet"
)

save_parquet(
    train_s3,
    "train_s3.parquet"
)

save_parquet(
    ground_truth,
    "ground_truth.parquet"
)

save_parquet(
    test_s1,
    "test_s1.parquet"
)

save_parquet(
    test_s2,
    "test_s2.parquet"
)

save_parquet(
    test_s3,
    "test_s3.parquet"
)


print("\n" + "=" * 70)
print("STEP 1 COMPLETE")
print("=" * 70)

print("\nProcessed files are in:")
print(PROCESSED_DIR)