"""Load the ISS food package into clean tables.

Reads ``iss_menu.csv``, ``usda_vitamins.csv``, ``micronutrient_requirements.csv``,
``shelf_life_decay.csv``, and ``energy_rules.json``. Menu rows are joined to USDA
vitamin estimates on item name. Rows whose ``data_quality_notes`` say the item
is a non-food placeholder or has no packaged mass are dropped. Every other noted
row is kept and flagged.

Run ``python data_loader.py`` to print row counts and missing values.
"""

from __future__ import annotations

import json
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

MENU_FILE = "iss_menu.csv"
VITAMIN_FILE = "usda_vitamins.csv"
REQUIREMENTS_FILE = "micronutrient_requirements.csv"
DECAY_FILE = "shelf_life_decay.csv"
ENERGY_FILE = "energy_rules.json"

MENU_NUMERIC = [
    "mass_g",
    "kcal",
    "protein_g",
    "carb_g",
    "fat_g",
    "sat_fat_g",
    "sodium_mg",
    "potassium_mg",
    "magnesium_mg",
    "iron_mg",
    "zinc_mg",
    "calcium_mg",
    "phosphorus_mg",
    "fiber_g",
    "source_page",
]

MINERAL_COLUMNS = [
    "sodium_mg",
    "potassium_mg",
    "magnesium_mg",
    "iron_mg",
    "zinc_mg",
    "calcium_mg",
    "phosphorus_mg",
]

VITAMIN_NUMERIC = [
    "menu_mass_g",
    "menu_kcal",
    "fdc_id",
    "usda_kcal_per_100g",
    "vit_a_rae_ug_per100g",
    "vit_c_mg_per100g",
    "vit_d_ug_per100g",
    "vit_e_mg_per100g",
    "vit_k_ug_per100g",
    "thiamin_b1_mg_per100g",
    "riboflavin_b2_mg_per100g",
    "niacin_b3_mg_per100g",
    "vit_b6_mg_per100g",
    "vit_b12_ug_per100g",
    "folate_dfe_ug_per100g",
    "serving_equiv_g",
    "vit_a_rae_ug_per_serving_est",
    "vit_c_mg_per_serving_est",
    "vit_d_ug_per_serving_est",
    "vit_e_mg_per_serving_est",
    "vit_k_ug_per_serving_est",
    "thiamin_b1_mg_per_serving_est",
    "riboflavin_b2_mg_per_serving_est",
    "niacin_b3_mg_per_serving_est",
    "vit_b6_mg_per_serving_est",
    "vit_b12_ug_per_serving_est",
    "folate_dfe_ug_per_serving_est",
]

# Copied from the menu file when the USDA row was built. The menu columns win.
VITAMIN_DROP = ["food_type", "menu_mass_g", "menu_kcal"]

REQUIREMENT_NUMERIC = ["rda_or_ai", "ul", "target_max"]

DECAY_NUMERIC = [
    "c_initial",
    "c_1yr",
    "c_3yr",
    "t_end_years",
    "k_per_year",
    "k_1yr_per_year",
    "half_life_years",
    "shelf_life_years_max",
]

@dataclass
class FoodPackage:
    """Clean tables for the ISS standard menu and the rules around it."""

    data_dir: Path
    menu: pd.DataFrame
    vitamins: pd.DataFrame
    requirements: pd.DataFrame
    decay: pd.DataFrame
    energy_rules: dict
    energy_parameters: pd.DataFrame
    eer_equations: pd.DataFrame
    macro_guidelines: pd.DataFrame
    foods: pd.DataFrame
    dropped: pd.DataFrame
    row_counts: pd.DataFrame
    missing_values: pd.DataFrame
    minerals_recoded: int
    summary_text: str


def load_all(data_dir: Path | str | None = None) -> FoodPackage:
    """Load, clean, join, and summarize the food package."""

    root = Path(data_dir) if data_dir is not None else DATA_DIR
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory not found: {root}")

    menu = load_iss_menu(root / MENU_FILE)
    vitamins = load_usda_vitamins(root / VITAMIN_FILE)
    requirements = load_micronutrient_requirements(root / REQUIREMENTS_FILE)
    decay = load_shelf_life_decay(root / DECAY_FILE)
    energy_rules = load_energy_rules(root / ENERGY_FILE)
    energy_parameters = energy_parameters_frame(energy_rules)
    eer_equations = eer_equations_frame(energy_rules)
    macro_guidelines = macro_guidelines_frame(energy_rules)

    menu, minerals_recoded = recode_unreliable_mineral_zeros(menu)
    merged = merge_menu_and_vitamins(menu, vitamins)
    merged = add_atwater_columns(merged)
    merged["quality_flag"] = merged["quality_action"].eq("flag")
    merged = _move_to_end(
        merged,
        [
            "quality_action",
            "quality_flag",
            "quality_reason",
            "atwater_kcal",
            "atwater_gap_pct",
        ],
    )

    foods = (
        merged.loc[merged["quality_action"].ne("drop")]
        .sort_values("item")
        .reset_index(drop=True)
    )
    dropped = (
        merged.loc[merged["quality_action"].eq("drop")]
        .sort_values("item")
        .reset_index(drop=True)
    )

    counted = {
        "iss_menu": (
            menu,
            "All source rows. quality_action is keep, flag, or drop.",
        ),
        "usda_vitamins": (
            vitamins,
            "USDA FoodData Central match, per 100 g and per serving.",
        ),
        "micronutrient_requirements": (
            requirements,
            "NASEM DRI life-stage rows and NASA spaceflight rows.",
        ),
        "shelf_life_decay": (
            decay,
            "Long format. Concentration columns are empty on shelf-life and qualitative rows.",
        ),
        "energy_parameters": (
            energy_parameters,
            "Flattened from the single energy_rules.json object.",
        ),
        "eer_equations": (
            eer_equations,
            "NASA-STD-3001 Table 7.1-1 coefficients. Male intercept is 662.",
        ),
        "macro_guidelines": (
            macro_guidelines,
            "Blank low or high means that side is not bounded in the source.",
        ),
        "foods": (
            foods,
            "Menu joined to vitamins on item. Unusable rows removed.",
        ),
        "dropped": (
            dropped,
            "Placeholder or missing-mass rows removed from the ledger.",
        ),
    }

    row_counts = pd.DataFrame(
        [_count_row(name, frame, detail) for name, (frame, detail) in counted.items()]
    )
    missing_values = missing_values_frame(
        {name: frame for name, (frame, _) in counted.items() if name != "macro_guidelines"}
    )
    package = FoodPackage(
        data_dir=root,
        menu=menu,
        vitamins=vitamins,
        requirements=requirements,
        decay=decay,
        energy_rules=energy_rules,
        energy_parameters=energy_parameters,
        eer_equations=eer_equations,
        macro_guidelines=macro_guidelines,
        foods=foods,
        dropped=dropped,
        row_counts=row_counts,
        missing_values=missing_values,
        minerals_recoded=minerals_recoded,
        summary_text="",
    )
    package.summary_text = format_summary(package)
    return package


def load_iss_menu(path: Path) -> pd.DataFrame:
    frame = _read_csv(path)
    _require_columns(frame, ["item", "data_quality_notes", *MENU_NUMERIC], path)
    frame["item"] = _clean_item(frame["item"], path)
    _numeric_strict(frame, MENU_NUMERIC)
    _blank_text_to_empty(frame, ["data_quality_notes"])
    frame["quality_action"] = frame["data_quality_notes"].map(quality_action)
    frame["quality_reason"] = frame["data_quality_notes"].map(quality_reason)
    return frame


def load_usda_vitamins(path: Path) -> pd.DataFrame:
    frame = _read_csv(path)
    _require_columns(frame, ["item", *VITAMIN_NUMERIC, "match_confidence"], path)
    frame["item"] = _clean_item(frame["item"], path)
    _numeric_strict(frame, VITAMIN_NUMERIC)
    _blank_text_to_empty(frame, ["match_notes"])
    if "fdc_id" in frame.columns:
        frame["fdc_id"] = frame["fdc_id"].astype("Int64")
    return frame


def load_micronutrient_requirements(path: Path) -> pd.DataFrame:
    frame = _read_csv(path)
    _require_columns(
        frame,
        ["nutrient", "unit", "group", "rda_or_ai", "ul", "value_type", "target_max"],
        path,
    )
    _numeric_strict(frame, REQUIREMENT_NUMERIC)
    _blank_text_to_empty(frame, ["notes", "differs_from_dri"])
    for column in ("nutrient", "unit", "group", "value_type"):
        frame[column] = frame[column].astype(str).str.strip()
    return frame


def load_shelf_life_decay(path: Path) -> pd.DataFrame:
    frame = _read_csv(path)
    _require_columns(frame, ["record_type", "vitamin", "category", *DECAY_NUMERIC], path)
    _numeric_strict(frame, DECAY_NUMERIC)
    # Qualitative bounds (>=0.85, stable, undetectable) and multi-temperature
    # notes stay text. Numeric-looking cells are stringified so the column is one type.
    for column in ("fraction_remaining_end", "storage_temp_c"):
        if column in frame.columns:
            frame[column] = frame[column].map(_as_text)
    return frame


def load_energy_rules(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    rules = json.loads(path.read_text(encoding="utf-8"))
    male = rules["eer_equations"]["male_19plus"]
    female = rules["eer_equations"]["female_19plus"]
    for equation in (male, female):
        for key in ("intercept", "age_coef", "mass_coef", "height_coef", "formula"):
            if key not in equation:
                raise KeyError(f"{path} is missing eer coefficient {key}")
    if float(male["intercept"]) != 662:
        raise ValueError(
            "Male EER intercept must be 662 (NASA-STD-3001 Rev E). "
            f"Found {male['intercept']}."
        )
    _ = rules["eva_energy"]["extra_kcal_per_eva_hour"]
    _ = rules["default_daily_energy"]["kcal_per_crewmember_per_day"]
    _ = rules["spaceflight_micronutrient_exception"]["vitamin_d_ug_per_day"]
    return rules


def energy_parameters_frame(rules: dict) -> pd.DataFrame:
    """One row per scalar rule in ``energy_rules.json``."""

    provision = rules["default_daily_energy"]
    activity = rules["eer_equations"]["activity_factor"]
    eva = rules["eva_energy"]
    vitamin_d = rules["spaceflight_micronutrient_exception"]
    history = rules["historical_iss_intake_reference"]
    rows = [
        ("provision", "daily_energy", provision["kcal_per_crewmember_per_day"], "kcal/day", provision["rule"]),
        ("provision", "daily_energy", provision["kj_per_crewmember_per_day"], "kJ/day", provision["reference_population_note"]),
        ("activity_factor", "nominal", activity["nominal"], "1", activity["nominal_label"]),
        ("activity_factor", "allowed_min", activity["allowed_range"][0], "1", activity["range_note"]),
        ("activity_factor", "allowed_max", activity["allowed_range"][1], "1", activity["range_note"]),
        ("eva", "extra_energy", eva["extra_kcal_per_eva_hour"], "kcal/hour", eva["rule"]),
        ("eva", "extra_energy", eva["extra_kj_per_eva_hour"], "kJ/hour", eva["suited_nutrition_rule"]),
        ("vitamin_d", "spaceflight_target", vitamin_d["vitamin_d_ug_per_day"], "ug/day", vitamin_d["rule"]),
        ("historical_iss", "energy", history["energy_kcal_per_day"], "kcal/day", history["note"]),
        ("historical_iss", "protein", history["protein_g_per_day"], "g/day", history["note"]),
        ("historical_iss", "sodium", history["sodium_mg_per_day"], "mg/day", history["note"]),
        ("historical_iss", "calcium", history["calcium_mg_per_day"], "mg/day", history["note"]),
        ("historical_iss", "potassium", history["potassium_mg_per_day"], "mg/day", history["note"]),
        ("historical_iss", "iron", history["iron_mg_per_day"], "mg/day", history["note"]),
        ("historical_iss", "n", history["n"], "crew observations", history["source"]),
        (
            "caveat",
            "male_intercept",
            rules["eer_equations"]["male_19plus"]["intercept"],
            "kcal",
            rules["eer_equations"]["caveat"],
        ),
    ]
    frame = pd.DataFrame(rows, columns=["section", "parameter", "value", "unit", "note"])
    frame["value"] = frame["value"].map(lambda value: value if isinstance(value, str) else f"{value:g}")
    return frame


def eer_equations_frame(rules: dict) -> pd.DataFrame:
    rows = []
    for sex, key in (("M", "male_19plus"), ("F", "female_19plus")):
        equation = rules["eer_equations"][key]
        rows.append(
            {
                "sex": sex,
                "equation": key,
                "intercept": float(equation["intercept"]),
                "age_coef": float(equation["age_coef"]),
                "mass_coef": float(equation["mass_coef"]),
                "height_coef": float(equation["height_coef"]),
                "formula": equation["formula"],
            }
        )
    return pd.DataFrame(rows)


def macro_guidelines_frame(rules: dict) -> pd.DataFrame:
    hidh = rules["macronutrients_nasa_hidh"]
    amdr = rules["macronutrients_dri_amdr_adults"]
    vitamin_d = rules["spaceflight_micronutrient_exception"]["vitamin_d_ug_per_day"]
    rows = [
        ("NASA HIDH", "protein", hidh["protein"]["g_per_kg_per_day"], None, "g/kg/day", hidh["protein"]["text"]),
        ("NASA HIDH", "protein", None, hidh["protein"]["max_pct_energy"], "% energy", hidh["protein"]["text"]),
        ("NASA HIDH", "carbohydrate", hidh["carbohydrate_pct_energy"][0], hidh["carbohydrate_pct_energy"][1], "% energy", ""),
        ("NASA HIDH", "fat", hidh["fat_pct_energy"][0], hidh["fat_pct_energy"][1], "% energy", ""),
        ("NASA HIDH", "saturated fat", None, hidh["saturated_fat_max_pct_energy"], "% energy", ""),
        ("NASA HIDH", "trans fat", None, hidh["trans_fat_max_pct_energy"], "% energy", ""),
        ("NASA HIDH", "cholesterol", None, hidh["cholesterol_max_mg_per_day"], "mg/day", ""),
        ("NASA HIDH", "fiber", hidh["fiber_g_per_4187kJ"][0], hidh["fiber_g_per_4187kJ"][1], "g/1000 kcal", hidh["fiber_note"]),
        ("NASA HIDH", "omega-6", hidh["omega6_g_per_day"], hidh["omega6_g_per_day"], "g/day", ""),
        ("NASA HIDH", "omega-3", hidh["omega3_g_per_day"][0], hidh["omega3_g_per_day"][1], "g/day", ""),
        ("DRI AMDR", "protein", amdr["protein_pct_energy"][0], amdr["protein_pct_energy"][1], "% energy", ""),
        ("DRI AMDR", "carbohydrate", amdr["carbohydrate_pct_energy"][0], amdr["carbohydrate_pct_energy"][1], "% energy", ""),
        ("DRI AMDR", "fat", amdr["fat_pct_energy"][0], amdr["fat_pct_energy"][1], "% energy", ""),
        ("DRI AMDR", "linoleic acid", amdr["linoleic_pct_energy"][0], amdr["linoleic_pct_energy"][1], "% energy", ""),
        ("DRI AMDR", "alpha-linolenic acid", amdr["alpha_linolenic_pct_energy"][0], amdr["alpha_linolenic_pct_energy"][1], "% energy", ""),
        ("DRI AMDR", "protein", amdr["protein_rda_g_per_kg"], amdr["protein_rda_g_per_kg"], "g/kg/day", ""),
        ("NASA-STD-3001", "vitamin D", vitamin_d, vitamin_d, "ug/day", rules["spaceflight_micronutrient_exception"]["rule"]),
    ]
    return pd.DataFrame(rows, columns=["source", "nutrient", "low", "high", "unit", "note"])


def quality_action(note: str) -> str:
    """Classify a ``data_quality_notes`` value.

    Drop placeholders and rows with no packaged mass (Caribbean Chicken, Milk,
    Drinking Water Container). Flag every other non-empty note. Keep the rest.
    """

    text = (note or "").strip().lower()
    if not text:
        return "keep"
    if "placeholder" in text or "mass blank" in text or "listed with no mass" in text:
        return "drop"
    return "flag"


def quality_reason(note: str) -> str:
    """Short label for a quality note. Empty when the source had no note."""

    text = (note or "").strip()
    if not text:
        return ""
    low = text.lower()
    reasons: list[str] = []
    if "placeholder" in low:
        reasons.append("non-food placeholder")
    if "mass blank" in low or "listed with no mass" in low:
        reasons.append("mass missing")
    if "likely missing data, not true zero" in low:
        reasons.append("minerals likely missing")
    if "sat_fat_g" in low:
        reasons.append("saturated fat blank")
    if "kcal check" in low:
        reasons.append("kcal far from 4P+4C+9F")
    if "possibly source typo" in low or "unusually high" in low:
        reasons.append("iron value suspect")
    specific = {"minerals likely missing", "saturated fat blank", "mass missing", "non-food placeholder"}
    if "blank in source:" in low and not specific.intersection(reasons):
        reasons.append("blank cells in source")
    if not reasons:
        reasons.append("see data_quality_notes")
    unique: list[str] = []
    for reason in reasons:
        if reason not in unique:
            unique.append(reason)
    return "; ".join(unique)


def recode_unreliable_mineral_zeros(menu: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Turn mineral zeros into missing where the note says they are not real zeros."""

    frame = menu.copy()
    mask = frame["data_quality_notes"].str.contains(
        "likely missing data, not true zero",
        case=False,
        na=False,
    )
    recoded = 0
    for column in MINERAL_COLUMNS:
        was_zero = mask & frame[column].eq(0)
        recoded += int(was_zero.sum())
        frame.loc[was_zero, column] = pd.NA
    frame[MINERAL_COLUMNS] = frame[MINERAL_COLUMNS].astype("Float64")
    return frame, recoded


def merge_menu_and_vitamins(menu: pd.DataFrame, vitamins: pd.DataFrame) -> pd.DataFrame:
    """Left-join USDA vitamins onto the menu by item name.

    Each kept food then has menu macros and minerals plus vitamin estimates
    per serving (``*_per_serving_est``).
    """

    vitamin_columns = vitamins.drop(columns=[column for column in VITAMIN_DROP if column in vitamins.columns])
    overlap = (set(menu.columns) & set(vitamin_columns.columns)) - {"item"}
    if overlap:
        raise ValueError(f"Menu and vitamin tables share unexpected columns: {sorted(overlap)}")
    return menu.merge(vitamin_columns, on="item", how="left", validate="one_to_one")


def add_atwater_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add derived Atwater energy, 4P + 4C + 9F, and the percent gap vs listed kcal."""

    out = frame.copy()
    out["atwater_kcal"] = 4 * out["protein_g"] + 4 * out["carb_g"] + 9 * out["fat_g"]
    denominator = out["kcal"].where(out["kcal"] > 0)
    out["atwater_gap_pct"] = (out["kcal"] - out["atwater_kcal"]) / denominator * 100
    return out


def missing_values_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = [_missing_frame(name, frame) for name, frame in frames.items()]
    parts = [part for part in parts if not part.empty]
    columns = ["table", "column", "missing", "rows", "missing_pct"]
    if not parts:
        return pd.DataFrame(columns=columns)
    combined = pd.concat(parts, ignore_index=True)
    return combined.sort_values(["table", "missing", "column"], ascending=[True, False, True]).reset_index(drop=True)


def format_summary(package: FoodPackage) -> str:
    """Plain-text row counts, quality actions, and missing values."""

    lines = [
        "ISS food data summary",
        f"Source: {package.data_dir}",
        "",
        "Row counts",
        f"{'table':<32} {'rows':>6} {'cols':>6} {'missing':>9}",
        "-" * 56,
    ]
    for row in package.row_counts.itertuples(index=False):
        lines.append(f"{row.table:<32} {row.rows:6d} {row.columns:6d} {row.missing_cells:9d}")
        if row.detail:
            lines.append(textwrap.fill(row.detail, width=88, initial_indent="    ", subsequent_indent="    "))

    dropped_n = int((package.menu["quality_action"] == "drop").sum())
    flagged_n = int((package.menu["quality_action"] == "flag").sum())
    lines.extend(["", f"Dropped: {dropped_n}", f"Flagged (kept): {flagged_n}", ""])
    lines.append(
        "Dropped rows are non-food placeholders or items with no packaged mass. "
        "Flagged rows stay in the ledger; their notes mark blank or suspect cells."
    )
    lines.append(
        f"Mineral zeros recoded to missing: {package.minerals_recoded} cells "
        f"({', '.join(MINERAL_COLUMNS)}) on rows noted as likely missing data, not true zero."
    )

    lines.extend(["", "Dropped items"])
    if package.dropped.empty:
        lines.append("  (none)")
    for row in package.dropped.itertuples(index=False):
        lines.extend(_item_note_lines(row.item, row.quality_reason, row.data_quality_notes))

    lines.extend(["", "Flagged items"])
    flagged = package.foods.loc[package.foods["quality_flag"]].sort_values("item")
    if flagged.empty:
        lines.append("  (none)")
    for row in flagged.itertuples(index=False):
        lines.extend(_item_note_lines(row.item, row.quality_reason, row.data_quality_notes))

    not_scaled = package.foods.loc[
        package.foods["scaling_basis"].fillna("").str.startswith("not scalable"),
        "item",
    ]
    if not not_scaled.empty:
        names = ", ".join(not_scaled.tolist())
        lines.extend(
            [
                "",
                f"Kept items with a USDA match but no per-serving vitamin estimate (not scalable): {names}.",
            ]
        )

    lines.extend(["", "Food types in the ledger"])
    type_counts = (
        package.foods.groupby(["food_type", "food_type_name"], dropna=False)
        .size()
        .reset_index(name="items")
        .sort_values(["food_type", "food_type_name"])
    )
    for row in type_counts.itertuples(index=False):
        lines.append(f"  {row.food_type:<4} {row.food_type_name:<22} {row.items:4d}")

    confidence = package.foods["match_confidence"].value_counts(dropna=False)
    lines.extend(["", "USDA match confidence (kept foods)"])
    for label, count in confidence.items():
        lines.append(f"  {str(label):<8} {int(count):4d}")

    lines.extend(["", "Missing values (columns with at least one gap)"])
    lines.append(
        "Empty notes, match notes, and differs_from_dri cells are stored as blank text, "
        "so they are not counted here. macro_guidelines leaves low or high blank when "
        "that side is unbounded; those structural blanks are omitted below."
    )
    scanned = [
        "iss_menu",
        "usda_vitamins",
        "micronutrient_requirements",
        "shelf_life_decay",
        "energy_parameters",
        "eer_equations",
        "foods",
        "dropped",
    ]
    for table in scanned:
        part = package.missing_values.loc[package.missing_values["table"].eq(table)]
        if part.empty:
            lines.extend(["", f"{table}: no missing values"])
            continue
        lines.extend(["", table])
        for row in part.itertuples(index=False):
            lines.append(f"  {row.column:<42} {row.missing:5d}  {row.missing_pct:7.1%}")

    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> FoodPackage:
    args = list(sys.argv[1:] if argv is None else argv)
    data_dir = Path(args[0]) if args else DATA_DIR
    package = load_all(data_dir)
    print(package.summary_text)
    return package


def _item_note_lines(item: str, reason: str, note: str) -> list[str]:
    lines = [f"  - {item} — {reason}"]
    if note and note.strip() and note.strip() != reason:
        lines.append(textwrap.fill(note.strip(), width=88, initial_indent="      ", subsequent_indent="      "))
    return lines


def _count_row(name: str, frame: pd.DataFrame, detail: str) -> dict:
    return {
        "table": name,
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "missing_cells": int(frame.isna().sum().sum()),
        "detail": detail,
    }


def _missing_frame(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    missing = frame.isna().sum()
    missing = missing[missing > 0]
    rows = []
    total = len(frame)
    for column, count in missing.items():
        rows.append(
            {
                "table": name,
                "column": str(column),
                "missing": int(count),
                "rows": int(total),
                "missing_pct": (float(count) / total) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8")


def _require_columns(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")


def _clean_item(series: pd.Series, path: Path) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    if cleaned.eq("").any() or series.isna().any() or cleaned.eq("nan").any():
        raise ValueError(f"{path.name} has a blank item name")
    if cleaned.duplicated().any():
        dupes = cleaned.loc[cleaned.duplicated()].unique().tolist()
        raise ValueError(f"{path.name} has duplicate item names: {dupes}")
    return cleaned


def _numeric_strict(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        raw = frame[column]
        converted = pd.to_numeric(raw, errors="coerce")
        original = raw.map(_strip_or_na)
        bad = original.notna() & converted.isna()
        if bad.any():
            sample = original.loc[bad].head(3).tolist()
            raise ValueError(f"Column {column} has non-numeric values: {sample}")
        frame[column] = converted


def _blank_text_to_empty(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column not in frame.columns:
            continue
        frame[column] = frame[column].map(lambda value: "" if pd.isna(value) else str(value).strip())


def _strip_or_na(value):
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    return text if text else pd.NA


def _as_text(value):
    if pd.isna(value):
        return pd.NA
    if isinstance(value, float):
        text = format(value, "g")
    else:
        text = str(value).strip()
    return text if text else pd.NA


def _move_to_end(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    present = [column for column in columns if column in frame.columns]
    rest = [column for column in frame.columns if column not in present]
    return frame[rest + present]


if __name__ == "__main__":
    main()
