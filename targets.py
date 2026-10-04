"""Daily nutrient targets for one crew member on one mission day.

Energy comes from the NASA EER equations in :func:`nutrition.eer_components`
(male intercept 662, nominal activity factor, and the EVA increment stored in
``energy_rules.json``). Macronutrient grams use the HIDH bands in that same
file. Vitamins and minerals start from the DRI life-stage row and are replaced
by a NASA spaceflight row when one applies to this crew member. Vitamin D uses
the flight value in ``spaceflight_micronutrient_exception``.

The menu planner in :func:`nutrition.daily_targets` is a separate table: it
keeps DRI AMDRs and DRI iron. This module is the crew-day dict.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from data_loader import load_all
from nutrition import ENERGY_SHARE, _food_ul, eer_components, life_stage_group

# HIDH lists fiber as grams per 4187 kJ. The energy-rules note equates 4187 kJ to 1000 kcal.
_FIBER_BASIS_KCAL = 1000.0

_AMOUNT_UNITS = {"ug", "mg", "g", "kcal"}

_SEX_CODES = {
    "m": "M",
    "male": "M",
    "f": "F",
    "female": "F",
}

_CACHE: dict[str | None, object] = {}


def daily_targets(
    age: float,
    sex: str,
    weight_kg: float,
    height_m: float,
    eva_hours: float = 0.0,
    activity_factor: float | None = None,
    data_dir: Path | str | None = None,
) -> dict:
    """Return one dict of daily nutrient targets for this crew member.

    Each value is a dict with a ``unit`` and the numeric bound or bounds that
    apply (``minimum``, ``maximum``, or both). Sex is ``M``/``F`` or
    ``male``/``female``. Age is years, height is meters, weight is kilograms.
    ``activity_factor`` defaults to the nominal factor in the energy rules.
    ``eva_hours`` adds the JSON EVA increment on top of estimated energy.

    Raises:
        ValueError: age is under 19, or sex is not recognized. This package
            has no DRI rows below age 19.
    """

    sex_code = _normalize_sex(sex)
    age_years = float(age)
    group = life_stage_group(sex_code, age_years)
    package = _package(data_dir)
    rules = package.energy_rules
    factor = None if activity_factor is None else float(activity_factor)
    components = eer_components(
        sex_code,
        age_years,
        float(weight_kg),
        float(height_m),
        af=factor,
        eva_h=float(eva_hours),
        rules=rules,
    )
    targets = _macro_targets(rules, components.eer_kcal, float(weight_kg))
    targets.update(_micronutrient_targets(package.requirements, rules, sex_code, group))
    return targets


def _macro_targets(rules: dict, kcal: float, weight_kg: float) -> dict:
    hidh = rules["macronutrients_nasa_hidh"]
    protein = hidh["protein"]
    protein_kcal = ENERGY_SHARE["Protein"]
    carb_kcal = ENERGY_SHARE["Carbohydrate"]
    fat_kcal = ENERGY_SHARE["Fat"]
    carb_low, carb_high = hidh["carbohydrate_pct_energy"]
    fat_low, fat_high = hidh["fat_pct_energy"]
    fiber_low, fiber_high = hidh["fiber_g_per_4187kJ"]
    return {
        "energy": _entry("kcal", kcal, kcal),
        "protein": _entry(
            "g",
            float(protein["g_per_kg_per_day"]) * weight_kg,
            _grams_from_energy_pct(protein["max_pct_energy"], kcal, protein_kcal),
        ),
        "carbohydrate": _entry(
            "g",
            _grams_from_energy_pct(carb_low, kcal, carb_kcal),
            _grams_from_energy_pct(carb_high, kcal, carb_kcal),
        ),
        "fat": _entry(
            "g",
            _grams_from_energy_pct(fat_low, kcal, fat_kcal),
            _grams_from_energy_pct(fat_high, kcal, fat_kcal),
        ),
        "saturated_fat": _entry(
            "g",
            maximum=_grams_from_energy_pct(hidh["saturated_fat_max_pct_energy"], kcal, fat_kcal),
        ),
        "fiber": _entry(
            "g",
            float(fiber_low) * kcal / _FIBER_BASIS_KCAL,
            float(fiber_high) * kcal / _FIBER_BASIS_KCAL,
        ),
    }


def _micronutrient_targets(requirements: pd.DataFrame, rules: dict, sex: str, group: str) -> dict:
    targets: dict = {}
    seen: set[str] = set()
    for nutrient in requirements["nutrient"].tolist():
        if nutrient == "Protein" or nutrient in seen:
            continue
        seen.add(nutrient)
        targets[_nutrient_key(nutrient)] = _micro_entry(requirements, rules, nutrient, sex, group)
    return targets


def _micro_entry(
    requirements: pd.DataFrame,
    rules: dict,
    nutrient: str,
    sex: str,
    group: str,
) -> dict:
    """DRI life-stage target, replaced by a NASA spaceflight row when one applies.

    A NASA ``target_max`` is the top of the flight band (sodium 1500–2300,
    iron 8–10). Otherwise a numeric DRI UL is the maximum when it applies to
    food. Supplement, synthetic, and preformed-vitamin-A ULs are left off.
    Vitamin D's minimum is the flight value from the energy rules.
    """

    dri = _dri_row(requirements, nutrient, group)
    nasa = _nasa_row(requirements, nutrient, sex)
    if dri is None and nasa is None:
        raise ValueError(f"No requirement row for {nutrient} / {group}")

    minimum = _finite(dri["rda_or_ai"]) if dri is not None else None
    unit = _amount_unit(dri["unit"]) if dri is not None else None
    maximum = None
    if nasa is not None:
        nasa_min = _finite(nasa["rda_or_ai"])
        if nasa_min is not None:
            minimum = nasa_min
        nasa_unit = _amount_unit(nasa["unit"])
        if nasa_unit is not None:
            unit = nasa_unit
        nasa_max = _finite(nasa["target_max"])
        if nasa_max is not None:
            maximum = nasa_max
    if nutrient == "Vitamin D":
        minimum = float(rules["spaceflight_micronutrient_exception"]["vitamin_d_ug_per_day"])
        unit = unit or "ug"
    if maximum is None and dri is not None:
        maximum, _note = _food_ul(dri)
    if unit is None:
        raise ValueError(f"No amount unit for {nutrient}")
    if minimum is None and maximum is None:
        raise ValueError(f"No numeric target for {nutrient}")
    return _entry(unit, minimum, maximum)


def _dri_row(requirements: pd.DataFrame, nutrient: str, group: str) -> pd.Series | None:
    hit = requirements.loc[requirements["nutrient"].eq(nutrient) & requirements["group"].eq(group)]
    if hit.empty:
        return None
    if len(hit) != 1:
        raise ValueError(f"Expected one DRI row for {nutrient} / {group}, found {len(hit)}")
    return hit.iloc[0]


def _nasa_row(requirements: pd.DataFrame, nutrient: str, sex: str) -> pd.Series | None:
    """Sex-specific NASA row when present, otherwise the all-crew row."""

    sex_group = "nasa_spaceflight male" if sex == "M" else "nasa_spaceflight female"
    rows = requirements.loc[
        requirements["nutrient"].eq(nutrient)
        & requirements["group"].isin([sex_group, "nasa_spaceflight all crew"])
    ]
    if rows.empty:
        return None
    specific = rows.loc[rows["group"].eq(sex_group)]
    chosen = specific if not specific.empty else rows.loc[rows["group"].eq("nasa_spaceflight all crew")]
    minima = [_finite(value) for value in chosen["rda_or_ai"]]
    maxima = [_finite(value) for value in chosen["target_max"]]
    if len(set(minima)) > 1 or len(set(maxima)) > 1:
        raise ValueError(f"Conflicting NASA spaceflight rows for {nutrient}")
    return chosen.iloc[0]


def _grams_from_energy_pct(pct: float, kcal: float, kcal_per_g: float) -> float:
    return float(pct) / 100.0 * float(kcal) / float(kcal_per_g)


def _entry(unit: str, minimum: float | None = None, maximum: float | None = None) -> dict:
    entry: dict = {}
    if minimum is not None:
        entry["minimum"] = float(minimum)
    if maximum is not None:
        entry["maximum"] = float(maximum)
    entry["unit"] = unit
    return entry


def _nutrient_key(nutrient: str) -> str:
    return "_".join(nutrient.strip().lower().replace("-", " ").split())


def _normalize_sex(sex: str) -> str:
    if isinstance(sex, str):
        code = _SEX_CODES.get(sex.strip().lower())
        if code is not None:
            return code
    raise ValueError("sex must be 'M', 'F', 'male', or 'female'")


def _amount_unit(raw: object) -> str | None:
    """Amount unit from a CSV unit cell: strip a trailing ``/d`` and keep ug, mg, or g."""

    if raw is None or pd.isna(raw):
        return None
    text = str(raw).strip()
    if not text or text.lower() == "see notes":
        return None
    lowered = text.lower()
    if lowered.endswith("/day"):
        text = text[: -len("/day")].strip()
    elif lowered.endswith("/d"):
        text = text[: -len("/d")].strip()
    if not text or text.lower() == "see notes":
        return None
    token = text.split()[0].lower().replace("µ", "u")
    if token in _AMOUNT_UNITS:
        return token
    return None


def _finite(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _package(data_dir: Path | str | None):
    key = None if data_dir is None else str(Path(data_dir).expanduser().resolve())
    package = _CACHE.get(key)
    if package is None:
        package = load_all(data_dir)
        _CACHE[key] = package
    return package
