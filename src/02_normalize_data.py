import os
import gc
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "processed"
)


# ============================================================
# FAST NORMALIZATION
# ============================================================

ABBREVIATIONS = {
    r"\bstreet\b": "st",
    r"\broad\b": "rd",
    r"\bavenue\b": "ave",
    r"\bboulevard\b": "blvd",
    r"\bdrive\b": "dr",
    r"\blane\b": "ln",
    r"\bhighway\b": "hwy",
    r"\bbuilding\b": "bldg",
    r"\bfloor\b": "fl",
    r"\bapartment\b": "apt",
    r"\bsuite\b": "ste",
    r"\bcorporation\b": "corp",
    r"\bcompany\b": "co",
    r"\blimited\b": "ltd",
    r"\bincorporated\b": "inc",
    r"\binternational\b": "intl",
}


def normalize_series(series):

    # IMPORTANT:
    # Force Python string backend.
    # PyArrow regex does not support \uXXXX.
    s = pd.Series(
        series.fillna("").astype(str),
        dtype="string[python]"
    )

    # Unicode normalization
    s = s.str.normalize("NFKD")

    # Remove Unicode combining marks.
    # Python regex supports this.
    s = s.str.replace(
        r"[\u0300-\u036f]",
        "",
        regex=True
    )

    # Lowercase
    s = s.str.lower()

    # Ampersand -> and
    s = s.str.replace(
        "&",
        " and ",
        regex=False
    )

    # Keep letters, numbers and spaces
    s = s.str.replace(
        r"[^a-z0-9\s]",
        " ",
        regex=True
    )

    # Collapse multiple spaces
    s = s.str.replace(
        r"\s+",
        " ",
        regex=True
    ).str.strip()

    # Abbreviations
    for pattern, replacement in ABBREVIATIONS.items():

        s = s.str.replace(
            pattern,
            replacement,
            regex=True
        )

    return s


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(filename):

    input_path = os.path.join(
        PROCESSED_DIR,
        filename
    )

    output_filename = filename.replace(
        ".parquet",
        "_norm.parquet"
    )

    output_path = os.path.join(
        PROCESSED_DIR,
        output_filename
    )

    print("\n" + "=" * 70)
    print("PROCESSING:", filename)
    print("=" * 70)

    print("\nReading:", input_path)

    df = pd.read_parquet(
        input_path
    )

    print(
        "Rows:",
        f"{len(df):,}"
    )

    print(
        "Memory before:",
        round(
            df.memory_usage(deep=True).sum()
            / (1024 ** 2),
            2
        ),
        "MB"
    )

    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

    print("\nNormalizing business names...")

    df["name_norm"] = normalize_series(
        df["business_name"]
    )

    # --------------------------------------------------------
    # ADDRESS
    # --------------------------------------------------------

    print("Normalizing addresses...")

    df["address_norm"] = normalize_series(
        df["business_address"]
    )

    # --------------------------------------------------------
    # COUNTRY
    # --------------------------------------------------------

    print("Normalizing countries...")

    df["country_norm"] = (
        pd.Series(
            df["country"].fillna("").astype(str),
            dtype="string[python]"
        )
        .str.lower()
        .str.strip()
    )

    # --------------------------------------------------------
    # COMPACT NAME
    # --------------------------------------------------------

    print("Creating compact names...")

    df["name_compact"] = (
        df["name_norm"]
        .str.replace(
            " ",
            "",
            regex=False
        )
    )

    # --------------------------------------------------------
    # COMPACT ADDRESS
    # --------------------------------------------------------

    print("Creating compact addresses...")

    df["address_compact"] = (
        df["address_norm"]
        .str.replace(
            " ",
            "",
            regex=False
        )
    )

    # --------------------------------------------------------
    # NAME BLOCKING KEYS
    # --------------------------------------------------------

    print("Creating name blocking keys...")

    df["name_prefix3"] = (
        df["name_compact"]
        .str[:3]
    )

    df["name_prefix5"] = (
        df["name_compact"]
        .str[:5]
    )

    df["name_first"] = (
        df["name_norm"]
        .str.split()
        .str[0]
        .fillna("")
    )

    # --------------------------------------------------------
    # ADDRESS BLOCKING KEYS
    # --------------------------------------------------------

    df["address_prefix8"] = (
        df["address_compact"]
        .str[:8]
    )

    print("Creating numeric address key...")

    df["address_digits"] = (
        df["address_norm"]
        .str.replace(
            r"\D",
            "",
            regex=True
        )
        .str[:6]
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    print("\nSaving:", output_path)

    df.to_parquet(
        output_path,
        index=False
    )

    size_mb = (
        os.path.getsize(output_path)
        / (1024 ** 2)
    )

    print(
        "Saved size:",
        round(size_mb, 2),
        "MB"
    )

    print(
        "Memory after:",
        round(
            df.memory_usage(deep=True).sum()
            / (1024 ** 2),
            2
        ),
        "MB"
    )

    del df
    gc.collect()

    print("✓ COMPLETE")


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("AMAZON ML CHALLENGE 2026")
print("STEP 2 — NORMALIZATION")
print("=" * 70)

files = [
    "train_s1.parquet",
    "train_s2.parquet",
    "train_s3.parquet",
    "test_s1.parquet",
    "test_s2.parquet",
    "test_s3.parquet",
]

for filename in files:
    process_file(filename)


print("\n" + "=" * 70)
print("STEP 2 COMPLETE")
print("=" * 70)

print("\nCreated normalized files:")

for filename in files:

    normalized = filename.replace(
        ".parquet",
        "_norm.parquet"
    )

    print(
        os.path.join(
            PROCESSED_DIR,
            normalized
        )
    )