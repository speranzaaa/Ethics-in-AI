import numpy as np
import pandas as pd

from .config import DATA_DIR

DATE_FORMAT = "%d.%m.%Y %H:%M:%S"

DATE_COLUMNS = [
    "DATAORA_ACCETTAZIONE",
    "DATAORA_TRIAGE",
    "DATAORA_DIMISSIONE",
    "DATA_NASCITA",
]

# NAP pattern definitions (Tier A / B / C)

TIER_A_PATTERNS = [
    r"\bmaltrattament\w*",
    r"\babuso\s+(sessuale|fisico|minorile|sui\s+minori|su\s+minor)",
    r"\bviolenza\s+(sessuale|fisica|domestic\w+)",
    r"\bpercoss\w+",
    r"\btrauma\s+non\s+accidental\w*",
    r"\bpatologia\s+non\s+accidental\w*",
    r"\blesion\w+\s+non\s+accidental\w*",
    r"\bsindrome\s+del\s+bambino\s+scosso",
    r"\bshaken\s+baby", r"\bbattered\s+child",
]

NAP_CONFIRM = r"\bnap\b"
NAP_NEGATE = (
    r"\b(sospett\w+|possibil\w+|esclus\w+|escludere|rule\s*out|screening\s+(per\s+)?)"
    r"(\s+di\s+|\s+la\s+|\s+)?(nap)\b"
    r"|\bnap\s+(sospett|esclus|negativ)"
)

TIER_B_PATTERNS = [
    r"\bsospett\w+\s+(di\s+)?(abus|maltratt|violenz|percoss|trauma\s+non\s+accidental|nap\b)",
    r"\b(abus|maltratt|violenz)\w*\s+sospett",
    r"\bscreening\s+(per\s+)?nap\b",
    r"\bnon\s+(compatibile|congruente|coerente|plausibile)\s+con\s+(la\s+|l['']\\s*)?(anamnesi|meccanismo|dinamica|riferito)",
    r"\bincompatibil\w+\s+con\s+(la\s+|l['']\\s*)?(anamnesi|meccanismo|dinamica)",
    r"\bincongru\w+\s+con\s+(la\s+|l['']\\s*)?(anamnesi|meccanismo|dinamica|riferito)",
    r"\banamnesi\s+(non\s+chiara|confusa|contraddittoria|incongruente)",
    r"\bdinamica\s+(non\s+chiara|poco\s+chiara|non\s+plausibile)",
]

TIER_C_PATTERNS = [
    r"\bservizi\s+social\w+",
    r"\btribunale\s+(per\s+i\s+minoren\w+|dei\s+minor\w*)",
    r"\bprocura\s+(della\s+repubblica\s+)?(per\s+i\s+minoren\w+|dei\s+minor\w*|minor\w*)",
    r"\bsegnala\w*\s+(ai|al|alla)\s+(servizi|autorit|procur|tribunal)",
    r"\btutela\s+(del|della)?\s*minor",
    r"\bdenuncia",
    r"\bcodice\s+rosa",
]

TEXT_COLS = ["DIAGNOSI", "TERAPIA", "DATI_RIFERITI", "ANAMNESI", "NOTE_AGGIUNTIVE"]

ACC_DIM_COLS = [
    "D01_ID_ACCESSO", "D01_ID_PAZIENTE", "D01_DATAORA_ACCETTAZIONE",
    "D30_DESC_GRAVITA", "D01_DATI_RIFERITI",
    "D01_DIAGNOSI_DIMISSIONE", "D01_TERAPIA_DIMISSIONE", "D31_DESC_CAUSALE",
]

TRIAGE_COLS = [
    "D07_ID_ACCESSO",
    "D07_FREQUENZA_CARDIACA",
    "D07_TEMPERATURA", "D07_SATURAZIONE_OSSIGENO",
    "D07_GCS_PEDIATRICO", "D07_GCS_VALORE",
    "D07_GCS_NEONATALE", "D07_PESO_CORPOREO",
]

RENAME_MAP = {
    "D01_ID_ACCESSO":"ID_ACCESSO",
    "D01_ID_PAZIENTE": "ID_PAZIENTE",
    "D01_DATAORA_ACCETTAZIONE": "DATA_ACCETTAZIONE",
    "D30_DESC_GRAVITA": "GRAVITA",
    "D01_DATI_RIFERITI": "DATI_RIFERITI",
    "D01_DIAGNOSI_DIMISSIONE": "DIAGNOSI",
    "D31_DESC_CAUSALE": "CAUSALE",
    "D01_TERAPIA_DIMISSIONE": "TERAPIA",
    "A01_SESSO": "SESSO",
    "A01_DATA_NASCITA": "DATA_NASCITA",
    "D07_FREQUENZA_CARDIACA":"FREQUENZA_CARDIACA",
    "D07_TEMPERATURA":"TEMPERATURA",
    "D07_SATURAZIONE_OSSIGENO": "SATURAZIONE_OSSIGENO",
    "D07_GCS_PEDIATRICO": "GCS_PEDIATRICO",
    "D07_GCS_VALORE": "GCS_VALORE",
    "D07_GCS_NEONATALE": "GCS_NEONATALE",
    "D07_PESO_CORPOREO":"PESO_CORPOREO",
}

GRAVITA_MAP = {"BIANCO": 0.0, "VERDE": 1.0, "GIALLO": 2.0, "ROSSO": 3.0}
BASE_FEATURES = ["codice_gravita", "codice_triage", "age_months",
                 "days_since_last_visit", "num_visits_90d"]


# NAP matching helpers

def any_match(text_low, cols, patterns):
    """True if at least one pattern matches at least one column."""
    regex = "|".join(patterns)
    masks = [text_low[c].str.contains(regex, regex=True, na=False) for c in cols]
    return np.logical_or.reduce(masks)


# KDE window construction

def rolling_prior_90d(group):
    group = group.sort_values("DATAORA_ACCETTAZIONE")
    dt_idx = pd.DatetimeIndex(group["DATAORA_ACCETTAZIONE"])
    ones = pd.Series(1.0, index=dt_idx)
    counts = (
        ones
        .rolling("90D", min_periods=1)
        .sum()
        .sub(1.0)
        .clip(lower=0.0)
    )
    return pd.Series(counts.values, index=group.index, dtype=float)


def build_features(df):
    df = df.copy()

    for col in ("DATAORA_ACCETTAZIONE", "DATA_NASCITA"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    n_before = len(df)
    df = df.dropna(subset=["DATAORA_ACCETTAZIONE"])
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)} rows with NaT in DATAORA_ACCETTAZIONE")

    if "DATA_NASCITA" in df.columns:
        age_days = (df["DATAORA_ACCETTAZIONE"] - df["DATA_NASCITA"]).dt.days
        df["age_months"] = (age_days / 30.44).clip(lower=0.0)
        median_age = df["age_months"].median()
        n_missing = df["age_months"].isna().sum()
        if n_missing:
            print(
                f"{n_missing} patients have missing DATA_NASCITA; imputing age_months "
                f"with median ({median_age})."
            )
        df["age_months"] = df["age_months"].fillna(
            median_age if pd.notna(median_age) else 0.0
        )
    else:
        print("DATA_NASCITA not found; age_months set to 0.0 for all rows.")
        df["age_months"] = 0.0

    df = df.sort_values(["ID_PAZIENTE", "DATAORA_ACCETTAZIONE"]).reset_index(drop=True)

    df["days_since_last_visit"] = (
        df.groupby("ID_PAZIENTE")["DATAORA_ACCETTAZIONE"]
        .diff()
        .dt.days
        .fillna(0.0)
        .clip(lower=0.0)
    )

    df["num_visits_90d"] = pd.concat(
        [rolling_prior_90d(g) for _, g in df.groupby("ID_PAZIENTE", sort=False)]
    ).reindex(df.index)

    print(
        f"build_features complete: {len(df)} rows — "
        "new columns: age_months, days_since_last_visit, num_visits_90d."
    )
    return df


def create_sliding_windows(
    df: pd.DataFrame,
    group_col: str = "ID_PAZIENTE",
    time_col: str = "DATAORA_ACCETTAZIONE",
    window_len: int = 3,
    numeric_only: bool = False,
):
    if window_len < 2:
        raise ValueError("window_len must be >= 2.")
    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found in dataframe.")
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found in dataframe.")

    exclude: set[str] = {group_col, time_col, "ID_ACCESSO"}
    exclude |= {c for c in DATE_COLUMNS if c in df.columns and c != time_col}

    if numeric_only:
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]
    else:
        feature_cols = [c for c in df.columns if c not in exclude]

    rows: list[dict] = []

    for patient_id, group in df.groupby(group_col, sort=False):
        group = group.sort_values(time_col).reset_index(drop=True)

        if len(group) < window_len:
            continue

        for end_idx in range(window_len - 1, len(group)):
            window = group.iloc[end_idx - window_len + 1 : end_idx + 1]
            row: dict = {
                group_col: patient_id,
                time_col: group.at[end_idx, time_col],
            }

            for offset, (_, visit) in enumerate(window.iloc[::-1].iterrows()):
                suffix = "_t" if offset == 0 else f"_t-{offset}"
                for col in feature_cols:
                    row[f"{col}{suffix}"] = visit[col]

            rows.append(row)

    if not rows:
        print(
            f"create_sliding_windows produced 0 windows. \n"
            f"Check that window_len ({window_len}) <= visit counts per patient."
        )
        return pd.DataFrame()

    result = pd.DataFrame(rows).reset_index(drop=True)
    print(
        f"created {len(result)} windows from {result[group_col].nunique()} patients "
        f"({window_len} windows with {len(feature_cols)} features each)."
    )
    return result


def build_window_dataset(df_src):
    """From the visits table build the time windows for KDE training."""
    df = df_src.copy()
    df = df.rename(columns={"DATA_ACCETTAZIONE": "DATAORA_ACCETTAZIONE"})
    df["DATAORA_ACCETTAZIONE"] = pd.to_datetime(df["DATAORA_ACCETTAZIONE"], errors="coerce")

    df["codice_gravita"] = df["GRAVITA"].map(GRAVITA_MAP).fillna(1.0)
    df["codice_triage"]  = df["codice_gravita"]

    df_feat = build_features(df)
    df_slim = df_feat[["ID_PAZIENTE", "DATAORA_ACCETTAZIONE"] + BASE_FEATURES].copy()

    windows = create_sliding_windows(
        df_slim,
        group_col="ID_PAZIENTE",
        time_col="DATAORA_ACCETTAZIONE",
        window_len=3,
        numeric_only=True,
    )
    return windows


# Main preprocessing pipeline

def run_preprocessing(input_path, output_dir=None):
    
    if output_dir is None:
        output_dir = DATA_DIR

    output_dir = str(output_dir)

    # Load all sheets
    sheets = pd.read_excel(input_path, sheet_name=None)
    print({name: df.shape for name, df in sheets.items()})

    # --- Anagrafiche ---
    anagrafiche = (
        sheets["Anagrafiche"][["A01_ID_PERSONA", "A01_SESSO", "A01_DATA_NASCITA"]]
        .drop_duplicates(subset="A01_ID_PERSONA")
        .copy()
    )
    anagrafiche["A01_DATA_NASCITA"] = pd.to_datetime(
        anagrafiche["A01_DATA_NASCITA"], format=DATE_FORMAT, errors="coerce"
    )

    # --- Acc-Dim ---
    acc_dim = sheets["Acc-Dim"][ACC_DIM_COLS].copy()
    acc_dim["D01_DATAORA_ACCETTAZIONE"] = pd.to_datetime(
        acc_dim["D01_DATAORA_ACCETTAZIONE"], format=DATE_FORMAT, errors="coerce"
    )

    # --- Calcolo età ---
    visite = acc_dim.merge(
        anagrafiche, left_on="D01_ID_PAZIENTE", right_on="A01_ID_PERSONA", how="left"
    ).drop(columns="A01_ID_PERSONA")

    adm, dob = visite["D01_DATAORA_ACCETTAZIONE"], visite["A01_DATA_NASCITA"]
    already_had_birthday = (
        (adm.dt.month * 100 + adm.dt.day) >= (dob.dt.month * 100 + dob.dt.day)
    )
    visite["ETA_ALLA_VISITA"] = (
        adm.dt.year - dob.dt.year - (~already_had_birthday).astype("Int64")
    ).astype("Int64")
    visite["ETA_IN_GIORNI_ALLA_VISITA"] = (
        visite["D01_DATAORA_ACCETTAZIONE"] - visite["A01_DATA_NASCITA"]
    ).dt.days.astype("Int64")

    # --- OBI ---
    obi_ids = set(sheets["OBI"]["D08_ID_ACCESSO"])
    visite["OBI"] = visite["D01_ID_ACCESSO"].isin(obi_ids)

    # --- Ricovero ---
    ricovero_ids = set(sheets["Ricovero"]["D36_ID_ACCESSO"])
    visite["RICOVERO"] = visite["D01_ID_ACCESSO"].isin(ricovero_ids)

    # --- Problema principale ---
    pp = sheets["Problema Principale"]
    pp_agg = (
        pp.dropna(subset=["E35_DESCRIZIONE"])
        .groupby("E36_ID_ACCESSO")["E35_DESCRIZIONE"]
        .agg(" | ".join)
        .reset_index()
        .rename(columns={"E35_DESCRIZIONE": "PROBLEMA_PRINCIPALE"})
    )
    visite = visite.merge(
        pp_agg, left_on="D01_ID_ACCESSO", right_on="E36_ID_ACCESSO", how="left"
    ).drop(columns="E36_ID_ACCESSO")
    visite["PROBLEMA_PRINCIPALE"] = visite["PROBLEMA_PRINCIPALE"].fillna("NON SPECIFICATO")

    # --- Triage ---
    triage = (
        sheets["Triage"][TRIAGE_COLS]
        .drop_duplicates(subset="D07_ID_ACCESSO", keep="first")
    )
    visite = visite.merge(
        triage, left_on="D01_ID_ACCESSO", right_on="D07_ID_ACCESSO", how="left"
    ).drop(columns="D07_ID_ACCESSO")

    # --- Dati clinici ---
    dc = sheets["Dati clinici"][["O03_ID_ACCESSO", "O16_DESC_TIPO_DATO", "O03_TESTO"]].copy()
    dc_agg = (
        dc.dropna(subset=["O03_TESTO"])
        .assign(O03_TESTO=lambda d: d["O03_TESTO"].astype(str))
        .groupby(["O03_ID_ACCESSO", "O16_DESC_TIPO_DATO"])["O03_TESTO"]
        .agg(" ".join)
        .reset_index()
    )
    dc_wide = dc_agg.pivot(
        index="O03_ID_ACCESSO", columns="O16_DESC_TIPO_DATO", values="O03_TESTO"
    ).reset_index()
    dc_wide.columns.name = None
    DC_RENAME = {
        "Anamnesi":"ANAMNESI",
        "Esame obiettivo": "NOTE_AGGIUNTIVE",
        "Referto": "REFERTO",
        "Diario clinico": "DIARIO_CLINICO",
    }
    dc_wide = dc_wide.rename(columns=DC_RENAME)
    visite = visite.merge(
        dc_wide, left_on="D01_ID_ACCESSO", right_on="O03_ID_ACCESSO", how="left"
    ).drop(columns=[c for c in ["O03_ID_ACCESSO", "REFERTO", "DIARIO_CLINICO"] if c in dc_wide.columns])

    # --- Outlier and NA filling ---
    n_before = len(visite)
    visite = visite[visite["ETA_ALLA_VISITA"] < 18].reset_index(drop=True)
    print(f"Removed {n_before - len(visite)} adult rows")

    high = visite["D07_TEMPERATURA"] > 50
    visite.loc[high, "D07_TEMPERATURA"] /= 10
    low = visite["D07_TEMPERATURA"] < 30
    visite.loc[low, "D07_TEMPERATURA"] = np.nan

    sp_low = visite["D07_SATURAZIONE_OSSIGENO"] < 70
    visite.loc[sp_low, "D07_SATURAZIONE_OSSIGENO"] = np.nan

    for col in ["D07_FREQUENZA_CARDIACA", "D07_PESO_CORPOREO"]:
        by_age = visite.groupby("ETA_ALLA_VISITA")[col].transform("median")
        overall = visite[col].median()
        visite[col] = visite[col].fillna(by_age).fillna(overall)

    visite["D07_TEMPERATURA"] = visite["D07_TEMPERATURA"].fillna(36.5)
    visite["D07_SATURAZIONE_OSSIGENO"] = visite["D07_SATURAZIONE_OSSIGENO"].fillna(99.0)
    visite["D07_GCS_VALORE"] = visite["D07_GCS_VALORE"].fillna(15.0)

    visite["D07_GCS_PEDIATRICO"] = visite["D07_GCS_PEDIATRICO"].fillna("NO")
    visite["D07_GCS_NEONATALE"]  = visite["D07_GCS_NEONATALE"].fillna("NO")
    visite["D31_DESC_CAUSALE"] = visite["D31_DESC_CAUSALE"].fillna("NON PRESENTE")

    for col in ["ANAMNESI", "NOTE_AGGIUNTIVE", "D01_DIAGNOSI_DIMISSIONE", "D01_TERAPIA_DIMISSIONE"]:
        visite[col] = visite[col].fillna("")

    # --- Column rename ---
    visite.rename(columns=RENAME_MAP, inplace=True)

    # --- NAP labelling ---
    text_low = {c: visite[c].fillna("").astype(str).str.lower() for c in TEXT_COLS}

    explicit_a    = any_match(text_low, ["DIAGNOSI", "TERAPIA"], TIER_A_PATTERNS)
    nap_any       = any_match(text_low, TEXT_COLS, [NAP_CONFIRM])
    nap_negated   = any_match(text_low, TEXT_COLS, [NAP_NEGATE])
    nap_confirmed = nap_any & ~nap_negated
    caus_aggr     = visite["CAUSALE"].fillna("").str.contains("AGGRESSIONE", regex=False)
    tier_a = explicit_a | nap_confirmed | caus_aggr

    tier_b = any_match(text_low, TEXT_COLS, TIER_B_PATTERNS) & ~tier_a
    tier_c = any_match(text_low, TEXT_COLS, TIER_C_PATTERNS) & ~tier_a & ~tier_b

    visite["NAP_LABEL"] = "negativo"
    visite.loc[tier_c, "NAP_LABEL"] = "segnale_sociale"
    visite.loc[tier_b, "NAP_LABEL"] = "sospetto"
    visite.loc[tier_a, "NAP_LABEL"] = "confermato"

    print("Distribuzione etichette:")
    print(visite["NAP_LABEL"].value_counts().to_string())

    # --- Split ---
    for col in visite.select_dtypes(include="object").columns:
        visite[col] = visite[col].where(visite[col].isna(), visite[col].astype(str))

    sicuri   = visite[visite["NAP_LABEL"] == "confermato"].copy()
    sospetti = visite[visite["NAP_LABEL"].isin(["sospetto", "segnale_sociale"])].copy()
    negativi = visite[visite["NAP_LABEL"] == "negativo"].copy()

    sicuri.to_parquet(f"{output_dir}/nap_sicuri.parquet", index=False)
    sospetti.to_parquet(f"{output_dir}/nap_sospetti.parquet", index=False)
    negativi.to_parquet(f"{output_dir}/nap_negativi.parquet", index=False)

    print(f"NAP: {len(sicuri):>6,}  -> nap_sicuri.parquet")
    print(f"Suspect: {len(sospetti):>6,}  -> nap_sospetti.parquet")
    print(f"Negativi: {len(negativi):>6,}  -> nap_negativi.parquet")

    # --- KDE windows ---
    windows_neg = build_window_dataset(negativi)
    windows_neg.to_parquet(f"{output_dir}/nap_negativi_windows.parquet", index=False)
    print(f"Windows salvate: {len(windows_neg)} righe -> nap_negativi_windows.parquet")

    return sicuri, sospetti, negativi, windows_neg
