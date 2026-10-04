"""Minimum-mass crew menu as a continuous linear program.

Servings are continuous, from zero up to two per person. The objective is
packaged mass. Nutrient floors and ceilings come from
:func:`targets.daily_targets`, summed across the crew. Energy is allowed to
exceed the estimated energy requirement: that target stores the same number
as both bounds, and hitting every other minimum while minimizing mass is not
an energy-equality problem. Sodium uses the tighter of the summed target
maximum and 2,300 mg per person.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from data_loader import load_all
from targets import daily_targets

# Food columns already used by nutrition.daily_targets / the menu table.
# Keys match targets.daily_targets. Nutrients with no column here (choline,
# copper, and the other DRI rows the menu does not carry) are not constrained.
FOOD_COLUMNS: dict[str, str] = {
    "energy": "kcal",
    "protein": "protein_g",
    "carbohydrate": "carb_g",
    "fat": "fat_g",
    "saturated_fat": "sat_fat_g",
    "fiber": "fiber_g",
    "sodium": "sodium_mg",
    "potassium": "potassium_mg",
    "calcium": "calcium_mg",
    "magnesium": "magnesium_mg",
    "iron": "iron_mg",
    "zinc": "zinc_mg",
    "phosphorus": "phosphorus_mg",
    "vitamin_a": "vit_a_rae_ug_per_serving_est",
    "vitamin_c": "vit_c_mg_per_serving_est",
    "vitamin_d": "vit_d_ug_per_serving_est",
    "vitamin_e": "vit_e_mg_per_serving_est",
    "vitamin_k": "vit_k_ug_per_serving_est",
    "thiamin": "thiamin_b1_mg_per_serving_est",
    "riboflavin": "riboflavin_b2_mg_per_serving_est",
    "niacin": "niacin_b3_mg_per_serving_est",
    "vitamin_b6": "vit_b6_mg_per_serving_est",
    "vitamin_b12": "vit_b12_ug_per_serving_est",
    "folate": "folate_dfe_ug_per_serving_est",
}

SERVINGS_PER_PERSON = 2.0
SODIUM_MG_PER_PERSON = 2300.0
MIN_ALLERGY_CHARS = 3
SERVING_FLOOR = 1e-4
# A minimum slack counts as unmet when it clears either bar.
SLACK_FRACTION = 0.005
SLACK_ABS = 1e-3
# Mass is a tie-break on the slack LP. It stays far below a 0.5% relative slack
# even if every food is taken at the serving cap.
_SLACK_MASS_WEIGHT = 1e-12
_REQUIRED_MEMBER_KEYS = ("age", "sex", "weight_kg", "height_m")


class NutrientConstraint:
    """One summed nutrient bound and the per-serving coefficient row."""

    __slots__ = ("name", "unit", "minimum", "maximum", "coef")

    def __init__(
        self,
        name: str,
        unit: str,
        minimum: float | None,
        maximum: float | None,
        coef: np.ndarray,
    ) -> None:
        self.name = name
        self.unit = unit
        self.minimum = minimum
        self.maximum = maximum
        self.coef = coef


def optimize_menu(
    crew: Sequence[Mapping] | Mapping,
    *,
    foods: pd.DataFrame | None = None,
    allergies: Sequence[str] | str | None = None,
    data_dir: Path | str | None = None,
) -> dict:
    """Minimize packaged grams for one crew day.

    ``crew`` is a list of members (or one member mapping). Each member needs
    ``age``, ``sex``, ``weight_kg``, and ``height_m``. Optional fields are
    ``eva_hours`` (default 0), ``activity_factor``, and ``allergies``.
    ``allergies`` on the call is united with allergies on the members.
    Allergy strings shorter than 3 characters are ignored. A remaining token
    excludes a food when it is a case-insensitive substring of ``item``.

    ``foods`` defaults to :func:`data_loader.load_all` ``.foods``. Rows with
    missing or non-positive ``mass_g`` are skipped. A missing nutrient cell
    contributes 0.

    The return dict has ``feasible``, ``foods`` (``item``, ``servings``,
    ``mass_g`` for servings above 1e-4), ``total_mass_g``, ``nutrients``,
    ``unmet``, and ``message``. When the mass problem is infeasible, slacks
    on the minimums identify the nutrients that cannot be met and those names
    are listed in ``message``.
    """

    members = _normalize_crew(crew)
    summed = _sum_targets(members, data_dir)
    _relax_energy_equality(summed)
    _cap_sodium(summed, len(members))
    tokens = _collect_allergies(members, allergies)
    items, mass, table = _eligible_foods(foods, data_dir, tokens)
    constraints = _constraints(summed, table)
    cap = SERVINGS_PER_PERSON * len(members)
    sodium_cap = _sodium_cap(summed, len(members))
    sodium_coef = _numeric_column(table, FOOD_COLUMNS["sodium"])

    if len(mass) == 0:
        return _result_from_solution(
            False,
            items,
            np.zeros(0),
            mass,
            constraints,
            _unmet_names(constraints, np.zeros(0), None, sodium_coef, cap, sodium_cap),
        )

    solution = _solve_mass(mass, constraints, cap)
    if solution is not None and _solution_meets(solution, constraints):
        return _result_from_solution(True, items, solution, mass, constraints, [])

    slack_x, slack_values = _solve_slacks(mass, constraints, cap)
    if slack_x is None:
        slack_x = np.zeros(len(mass))
        slack_values = None
    unmet = _unmet_names(constraints, slack_x, slack_values, sodium_coef, cap, sodium_cap)
    return _result_from_solution(False, items, slack_x, mass, constraints, unmet)


def _normalize_crew(crew: Sequence[Mapping] | Mapping) -> list[Mapping]:
    if isinstance(crew, Mapping):
        crew = [crew]
    if isinstance(crew, (str, bytes)) or not isinstance(crew, Sequence):
        raise TypeError("crew must be a list of members")
    members = list(crew)
    if not members:
        raise ValueError("crew must include at least one member")
    for member in members:
        if not isinstance(member, Mapping):
            raise TypeError("each crew member must be a mapping")
        missing = [key for key in _REQUIRED_MEMBER_KEYS if key not in member]
        if missing:
            raise ValueError(f"crew member is missing {', '.join(missing)}")
    return members


def _sum_targets(members: Sequence[Mapping], data_dir: Path | str | None) -> dict[str, dict]:
    combined: dict[str, dict] = {}
    for member in members:
        eva = member.get("eva_hours", 0.0)
        if eva is None:
            eva = 0.0
        activity = member.get("activity_factor", None)
        day = daily_targets(
            float(member["age"]),
            str(member["sex"]),
            float(member["weight_kg"]),
            float(member["height_m"]),
            eva_hours=float(eva),
            activity_factor=None if activity is None else float(activity),
            data_dir=data_dir,
        )
        for name, entry in day.items():
            slot = combined.get(name)
            if slot is None:
                slot = {"unit": entry["unit"], "minimum": None, "maximum": None}
                combined[name] = slot
            elif slot["unit"] != entry["unit"]:
                raise ValueError(f"Unit mismatch for {name}: {slot['unit']} vs {entry['unit']}")
            if "minimum" in entry:
                slot["minimum"] = float(entry["minimum"]) + (0.0 if slot["minimum"] is None else slot["minimum"])
            if "maximum" in entry:
                slot["maximum"] = float(entry["maximum"]) + (0.0 if slot["maximum"] is None else slot["maximum"])
    return combined


def _relax_energy_equality(summed: dict[str, dict]) -> None:
    """Drop an energy ceiling that only repeats the EER.

    Minimizing mass subject to every minimum must be allowed to land above
    that single number. A wider stored band, if one existed, would stay.
    """

    energy = summed.get("energy")
    if energy is None:
        return
    low = energy["minimum"]
    high = energy["maximum"]
    if low is None or high is None:
        return
    if math.isclose(low, high, rel_tol=1e-9, abs_tol=1e-6):
        energy["maximum"] = None


def _cap_sodium(summed: dict[str, dict], crew_size: int) -> None:
    """Keep sodium at or under 2,300 mg per person, or tighter when the target is."""

    cap = SODIUM_MG_PER_PERSON * crew_size
    slot = summed.get("sodium")
    if slot is None:
        summed["sodium"] = {"unit": "mg", "minimum": None, "maximum": cap}
        return
    if slot["maximum"] is None:
        slot["maximum"] = cap
    else:
        slot["maximum"] = min(float(slot["maximum"]), cap)


def _sodium_cap(summed: dict[str, dict], crew_size: int) -> float:
    slot = summed.get("sodium")
    if slot is None or slot["maximum"] is None:
        return SODIUM_MG_PER_PERSON * crew_size
    return float(slot["maximum"])


def _collect_allergies(members: Sequence[Mapping], extra: Sequence[str] | str | None) -> list[str]:
    raw: list[object] = []
    for member in members:
        if "allergies" in member:
            raw.append(member.get("allergies"))
        elif "allergy" in member:
            raw.append(member.get("allergy"))
    if extra is not None:
        raw.append(extra)
    tokens: list[str] = []
    seen: set[str] = set()
    for value in raw:
        for token in _allergy_tokens(value):
            if token not in seen:
                seen.add(token)
                tokens.append(token)
    return tokens


def _allergy_tokens(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Sequence[object] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        values = [value]
    tokens: list[str] = []
    for item in values:
        if item is None:
            continue
        token = str(item).strip().casefold()
        if len(token) < MIN_ALLERGY_CHARS:
            continue
        tokens.append(token)
    return tokens


def _eligible_foods(
    foods: pd.DataFrame | None,
    data_dir: Path | str | None,
    tokens: list[str],
) -> tuple[list[str], np.ndarray, pd.DataFrame]:
    frame = foods if foods is not None else load_all(data_dir).foods
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("foods must be a DataFrame")
    if "item" not in frame.columns or "mass_g" not in frame.columns:
        raise ValueError("foods must include item and mass_g columns")
    items = frame["item"].map(_item_name)
    mass = pd.to_numeric(frame["mass_g"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(mass) & (mass > 0) & items.ne("").to_numpy()
    if tokens:
        folded = items.str.casefold()
        blocked = np.zeros(len(frame), dtype=bool)
        for token in tokens:
            blocked |= folded.str.contains(token, regex=False, na=False).to_numpy()
        mask &= ~blocked
    positions = np.flatnonzero(mask)
    kept = frame.iloc[positions].reset_index(drop=True)
    kept_mass = np.ascontiguousarray(mass[positions], dtype=float)
    kept_items = items.iloc[positions].tolist()
    return kept_items, kept_mass, kept


def _item_name(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _constraints(summed: dict[str, dict], foods: pd.DataFrame) -> list[NutrientConstraint]:
    rows: list[NutrientConstraint] = []
    for name, column in FOOD_COLUMNS.items():
        slot = summed.get(name)
        if slot is None:
            continue
        minimum = _finite_or_none(slot.get("minimum"))
        maximum = _finite_or_none(slot.get("maximum"))
        if minimum is None and maximum is None:
            continue
        rows.append(
            NutrientConstraint(
                name,
                str(slot["unit"]),
                minimum,
                maximum,
                _numeric_column(foods, column),
            )
        )
    return rows


def _finite_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _numeric_column(foods: pd.DataFrame, column: str) -> np.ndarray:
    if column not in foods.columns:
        return np.zeros(len(foods), dtype=float)
    return pd.to_numeric(foods[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)


def _solve_mass(mass: np.ndarray, constraints: Sequence[NutrientConstraint], cap: float) -> np.ndarray | None:
    rows, rhs = _bound_rows(constraints, n_slack=0)
    result = _linprog(mass, rows, rhs, (0.0, float(cap)))
    if not result.success or result.x is None:
        return None
    return np.clip(result.x[: len(mass)], 0.0, cap)


def _solve_slacks(
    mass: np.ndarray,
    constraints: Sequence[NutrientConstraint],
    cap: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Minimize the sum of relative minimum-slacks, with upper bounds kept hard.

    A tiny mass term only breaks ties. It is too small to buy a material slack.
    """

    minima = [row for row in constraints if row.minimum is not None]
    n = len(mass)
    n_slack = len(minima)
    if n_slack == 0:
        return np.zeros(n), np.zeros(0)
    cost = np.zeros(n + n_slack)
    cost[:n] = mass * _SLACK_MASS_WEIGHT
    for index, row in enumerate(minima):
        cost[n + index] = 1.0 / max(float(row.minimum), 1e-6)
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for index, row in enumerate(minima):
        coef = np.zeros(n + n_slack)
        coef[:n] = -row.coef
        coef[n + index] = -1.0
        rows.append(coef)
        rhs.append(-float(row.minimum))
    for row in constraints:
        if row.maximum is None:
            continue
        coef = np.zeros(n + n_slack)
        coef[:n] = row.coef
        rows.append(coef)
        rhs.append(float(row.maximum))
    bounds = [(0.0, float(cap))] * n + [(0.0, None)] * n_slack
    result = _linprog(cost, rows, rhs, bounds)
    if not result.success or result.x is None:
        return None, None
    servings = np.clip(result.x[:n], 0.0, cap)
    slacks = np.maximum(result.x[n : n + n_slack], 0.0)
    return servings, slacks


def _bound_rows(
    constraints: Sequence[NutrientConstraint],
    n_slack: int,
) -> tuple[list[np.ndarray], list[float]]:
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for row in constraints:
        width = len(row.coef) + n_slack
        if row.minimum is not None:
            coef = np.zeros(width)
            coef[: len(row.coef)] = -row.coef
            rows.append(coef)
            rhs.append(-float(row.minimum))
        if row.maximum is not None:
            coef = np.zeros(width)
            coef[: len(row.coef)] = row.coef
            rows.append(coef)
            rhs.append(float(row.maximum))
    return rows, rhs


def _linprog(cost: np.ndarray, rows: list[np.ndarray], rhs: list[float], bounds) -> object:
    kwargs = {
        "c": np.asarray(cost, dtype=float),
        "bounds": bounds,
        "method": "highs",
    }
    if rows:
        matrix = np.vstack(rows)
        limits = np.asarray(rhs, dtype=float)
        scale = np.max(np.abs(matrix), axis=1)
        scale = np.where(scale > 0.0, scale, 1.0)
        kwargs["A_ub"] = matrix / scale[:, None]
        kwargs["b_ub"] = limits / scale
    return linprog(**kwargs)


def _solution_meets(servings: np.ndarray, constraints: Sequence[NutrientConstraint]) -> bool:
    for row in constraints:
        amount = float(row.coef @ servings) if len(servings) else 0.0
        if not _is_met(amount, row.minimum, row.maximum):
            return False
    return True


def _is_met(amount: float, minimum: float | None, maximum: float | None) -> bool:
    if minimum is not None and amount < minimum - _met_tolerance(minimum):
        return False
    if maximum is not None and amount > maximum + _met_tolerance(maximum):
        return False
    return True


def _met_tolerance(value: float) -> float:
    return max(1e-4, 1e-6 * abs(value))


def _unmet_names(
    constraints: Sequence[NutrientConstraint],
    servings: np.ndarray,
    slacks: np.ndarray | None,
    sodium: np.ndarray,
    cap: float,
    sodium_cap: float,
) -> list[str]:
    minima = [row for row in constraints if row.minimum is not None]
    slack_by_name: dict[str, float] = {}
    if slacks is not None and len(slacks) == len(minima):
        for row, slack in zip(minima, slacks):
            slack_by_name[row.name] = float(slack)
    unmet: list[str] = []
    for row in constraints:
        if row.minimum is None:
            continue
        amount = float(row.coef @ servings) if len(servings) else 0.0
        shortfall = max(0.0, float(row.minimum) - amount)
        slack = max(shortfall, slack_by_name.get(row.name, 0.0))
        alone = _max_under_sodium(row.coef, sodium, cap, sodium_cap)
        individually_short = alone < float(row.minimum) - max(1e-5, 1e-8 * abs(float(row.minimum)))
        if _slack_is_material(slack, float(row.minimum)) or individually_short:
            unmet.append(row.name)
    return unmet


def _slack_is_material(slack: float, minimum: float) -> bool:
    if slack <= 0.0 or not math.isfinite(slack):
        return False
    if minimum <= 0.0:
        return slack > SLACK_ABS
    return slack > SLACK_FRACTION * abs(minimum) or slack > SLACK_ABS


def _max_under_sodium(coef: np.ndarray, sodium: np.ndarray, cap: float, sodium_cap: float) -> float:
    """Most of one nutrient under the serving cap and the sodium cap.

    One knapsack constraint plus upper bounds is a fractional knapsack: take
    zero-sodium foods first, then the rest in nutrient-per-milligram order.
    """

    if cap <= 0.0 or coef.size == 0:
        return 0.0
    total = 0.0
    ranked: list[tuple[float, float, float]] = []
    for amount, na in zip(coef.tolist(), sodium.tolist()):
        if amount <= 0.0:
            continue
        if na <= 0.0:
            total += amount * cap
            continue
        ranked.append((amount / na, amount, na))
    ranked.sort(key=lambda row: (-row[0], -row[1]))
    remaining = max(0.0, float(sodium_cap))
    for _ratio, amount, na in ranked:
        if remaining <= 1e-12:
            break
        servings = min(cap, remaining / na)
        total += amount * servings
        remaining -= na * servings
    return total


def _result_from_solution(
    feasible: bool,
    items: list[str],
    servings: np.ndarray,
    mass: np.ndarray,
    constraints: Sequence[NutrientConstraint],
    unmet: list[str],
) -> dict:
    foods: list[dict] = []
    for item, taken, pack_mass in zip(items, servings.tolist(), mass.tolist()):
        taken_f = float(taken)
        if taken_f <= SERVING_FLOOR:
            continue
        foods.append(
            {
                "item": item,
                "servings": taken_f,
                "mass_g": taken_f * float(pack_mass),
            }
        )
    foods.sort(key=lambda row: (-row["mass_g"], row["item"]))
    nutrients: dict[str, dict] = {}
    for row in constraints:
        amount = float(row.coef @ servings) if len(servings) else 0.0
        nutrients[row.name] = {
            "amount": amount,
            "unit": row.unit,
            "minimum": None if row.minimum is None else float(row.minimum),
            "maximum": None if row.maximum is None else float(row.maximum),
            "met": _is_met(amount, row.minimum, row.maximum),
        }
    total_mass_g = float(mass @ servings) if len(servings) else 0.0
    if feasible:
        message = f"Minimum-mass menu is {total_mass_g:.1f} g across {len(foods)} foods."
        unmet_out: list[str] = []
    else:
        unmet_out = list(unmet)
        if unmet_out:
            message = "Infeasible. Unmet nutrients: " + ", ".join(unmet_out) + "."
        else:
            message = "Infeasible. The serving cap, sodium cap, and nutrient bounds cannot be met together."
    return {
        "feasible": bool(feasible),
        "foods": foods,
        "total_mass_g": total_mass_g,
        "nutrients": nutrients,
        "unmet": unmet_out,
        "message": message,
    }
