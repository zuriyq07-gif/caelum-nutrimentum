"""Vitamin content over a mission, and a menu replan every few weeks.

Shelf-life rates in ``shelf_life_decay.csv`` are per year. Mission time is
days, so the amount on day ``t`` is::

    C(t) = C0 * exp(-k_per_year * (t_days / 365.25))

which is ``C0 * exp(-k_per_day * t_days)`` with
``k_per_day = k_per_year / 365.25``. The yearly rate is never multiplied by
the raw day count. That is the same closed form as
:func:`nutrition.remaining`, ``C(t) = C0 * exp(-k t)``, with ``t`` in years.

Matching a rate to a food uses vitamin identity and food class. Precedence,
first finite ``k`` wins, is documented on :func:`match_decay_rate`. Rates are
not averaged. A negative published ``k`` is clipped to 0, matching the data
package (fortified foods that rose in the measurements are treated as stable).
Vitamins with no applicable rate stay at ``C0``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from data_loader import load_all
from mission import _crew_for_day
from optimizer import optimize_menu
from targets import daily_targets

DAYS_PER_YEAR = 365.25

# Canonical vitamin key -> per-serving column on the merged menu.
VITAMIN_COLUMNS: dict[str, str] = {
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

# Minerals are decayed only when the shelf-life table has a numeric k.
MINERAL_COLUMNS: dict[str, str] = {
    "calcium": "calcium_mg",
    "potassium": "potassium_mg",
}

NUTRIENT_COLUMNS: dict[str, str] = {**VITAMIN_COLUMNS, **MINERAL_COLUMNS}

VITAMIN_LABELS: dict[str, str] = {
    "vitamin_a": "Vitamin A",
    "vitamin_c": "Vitamin C",
    "vitamin_d": "Vitamin D",
    "vitamin_e": "Vitamin E",
    "vitamin_k": "Vitamin K",
    "thiamin": "Thiamin (B1)",
    "riboflavin": "Riboflavin (B2)",
    "niacin": "Niacin (B3)",
    "vitamin_b6": "Vitamin B6",
    "vitamin_b12": "Vitamin B12",
    "folate": "Folate",
}

_VITAMIN_ALIASES = {
    "vitamin_c": "vitamin_c",
    "vitamin c": "vitamin_c",
    "ascorbic acid": "vitamin_c",
    "vitamin_a": "vitamin_a",
    "vitamin a": "vitamin_a",
    "beta-carotene": "vitamin_a",
    "beta-carotene (provitamin a)": "vitamin_a",
    "vitamin_d": "vitamin_d",
    "vitamin d": "vitamin_d",
    "vitamin_e": "vitamin_e",
    "vitamin e": "vitamin_e",
    "vitamin_k": "vitamin_k",
    "vitamin k": "vitamin_k",
    "vitamin_b1": "thiamin",
    "vitamin b1": "thiamin",
    "thiamin": "thiamin",
    "thiamine": "thiamin",
    "vitamin_b2": "riboflavin",
    "vitamin b2": "riboflavin",
    "riboflavin": "riboflavin",
    "niacin": "niacin",
    "vitamin_b3": "niacin",
    "vitamin b3": "niacin",
    "vitamin_b6": "vitamin_b6",
    "vitamin b6": "vitamin_b6",
    "vitamin_b12": "vitamin_b12",
    "vitamin b12": "vitamin_b12",
    "folate": "folate",
    "folic acid": "folate",
    "calcium": "calcium",
    "potassium": "potassium",
}

_GENERIC_CATEGORIES = {"any food", "any other food", "all foods", "all"}
_SERVING_FLOOR = 1e-4

_POULTRY = {"chicken", "turkey", "poultry", "duck"}
_BEEF = {"beef", "brisket", "steak"}
_FISH = {"fish", "tuna", "salmon", "shrimp", "crawfish", "seafood"}
_PORK = {"pork", "ham", "bacon", "sausage"}
_BREAD = {"bread", "tortilla", "tortillas", "cracker", "crackers", "cookie", "cookies", "biscuit", "biscuits", "muffin", "muffins"}
_FRUIT = {
    "apple",
    "apples",
    "applesauce",
    "pear",
    "pears",
    "peach",
    "peaches",
    "apricot",
    "apricots",
    "berry",
    "berries",
    "strawberry",
    "strawberries",
    "blueberry",
    "blueberries",
    "raspberry",
    "raspberries",
    "cranberry",
    "cranberries",
    "cherry",
    "cherries",
    "grape",
    "grapes",
    "grapefruit",
    "pineapple",
    "mango",
    "banana",
    "bananas",
    "citrus",
    "fruit",
    "orange",
    "oranges",
    "plum",
    "plums",
    "raisin",
    "raisins",
    "cranapple",
}
_VEGETABLE = {
    "broccoli",
    "bean",
    "beans",
    "potato",
    "potatoes",
    "corn",
    "asparagus",
    "pea",
    "peas",
    "carrot",
    "carrots",
    "tomato",
    "tomatoes",
    "vegetable",
    "vegetables",
    "cauliflower",
    "mushroom",
    "mushrooms",
    "spinach",
    "cabbage",
    "eggplant",
    "artichoke",
    "artichokes",
}


@dataclass(frozen=True)
class DecayRate:
    """One vitamin's first-order rate for one food.

    ``level`` is ``decay_item``, ``food_type``, ``decay_group``,
    ``decay_system``, ``fallback_default``, or ``stable`` when nothing matched.
    ``k_per_year`` is 0 when the vitamin is stable or a negative rate was clipped.
    """

    vitamin: str
    k_per_year: float
    level: str
    matched: str
    clipped: bool = False


@dataclass(frozen=True)
class _Row:
    order: int
    category: object
    label: str
    k: float | None
    norm_name: str


@dataclass
class _DecayIndex:
    items: dict[tuple[str, str], list[_Row]]
    food_type: dict[str, list[_Row]]
    group: dict[str, list[_Row]]
    system: dict[str, list[_Row]]
    fallback: dict[str, list[_Row]]


def concentration(c0: float, k_per_year: float, day: float) -> float:
    """Vitamin left on mission day ``day``.

    ``k_per_year`` is the shelf-life table rate. Mission time is days, so the
    exponent is ``-k_per_year * (day / 365.25)``. That equals
    ``C0 * exp(-k_per_day * day)`` with ``k_per_day = k_per_year / 365.25``.
    Do not multiply the yearly rate by the raw day count.

    This is the closed form used by :func:`nutrition.remaining`,
    ``C(t) = C0 * exp(-k * t)``, with ``t`` expressed in years. Day 0 returns
    ``C0``. Pass ``k_per_year`` 0 for a vitamin that has no applicable rate.
    """

    t_years = float(day) / DAYS_PER_YEAR
    return float(c0) * math.exp(-float(k_per_year) * t_years)


def epoch_starts(days: float, replan_every: float = 30) -> list[int]:
    """Epoch starts ``0, step, 2*step, ...`` while ``start < days``.

    A 30-day mission is one epoch at day 0 (days 0–29). A 31-day mission
    adds day 30. A 90-day mission is days 0, 30, and 60.
    """

    mission_days = _positive_number(days, "days")
    step = _positive_number(replan_every, "replan_every")
    starts: list[int] = []
    t = 0.0
    while t < mission_days - 1e-9:
        starts.append(int(round(t)))
        t += step
        if len(starts) > 10000:
            raise ValueError("replan interval is too small for this mission length")
    return starts


def match_decay_rate(
    item: str,
    food_type: str | None,
    vitamin: str,
    decay_table: pd.DataFrame,
) -> DecayRate:
    """Pick one yearly rate for this food and vitamin.

    Precedence, first row with a finite ``k`` wins. Rates are not averaged.

    1. ``decay_item`` — normalized item name equals ``item_or_group``.
       Normalization is casefold, asterisks removed, whitespace collapsed,
       and a trailing ``in pouch`` / ``in pouches`` dropped. Match is exact
       after that, so Applesauce does not claim Rhubarb Applesauce. The
       processing code is not required to agree. A non-numeric item row does
       not block a less specific rate. A negative item rate is clipped to 0
       and does block less specific rates (the food was measured).
    2. Food type — a ``fallback_default`` row whose category names this
       food's class (fruit, vegetable, fortified beverage, meat or poultry
       or fish, chicken, beef, bread or tortilla, fresh food). This is more
       specific than a whole-menu rate. The fruit vitamin C default
       (category ``fruit (any type)``, ``k_per_year`` 0.3005) wins here for
       a fruit that has no item row. Generic categories ``any food`` and
       ``any other food`` are not this level. If the winning category has
       no numeric ``k`` (fresh food), the vitamin stays at ``C0`` and
       shelf-stable rates are not applied.
    3. ``decay_group`` — group category (fruit products, meats, breads,
       chicken products, beef products, or ``all foods`` when no named
       group matches). A published range that is two rows is not averaged;
       the earliest row in the table at the winning specificity is used.
    4. ``decay_system`` — whole-menu rate for that vitamin. Not applied to
       beverages (food type ``B``); those system rows excluded beverages.
    5. ``fallback_default`` generic — ``any other food``, else ``any food``.
    6. Otherwise ``k = 0`` (stable).

    Food class comes from the menu food-type code and the item name. Beverages
    (code ``B``) stay beverages even when the name contains a fruit.
    Fresh food (code ``FF``) does not inherit shelf-stable classes. A meat,
    poultry, or fish name does not also take a fruit or vegetable rate.
    Bread and bakery rates apply to natural-form breads, tortillas, crackers,
    cookies, and muffins, not to every dish with the word bread.
    Only vitamin columns are decayed, plus calcium and potassium when the
    table actually has a ``k`` for them. Macros, mass, and kcal are never
    given a rate by this function.
    """

    return _match(_index(decay_table), item, food_type, vitamin)


def decayed_foods(foods: pd.DataFrame, day: float, decay_table: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``foods`` with per-serving vitamin columns set to ``C(day)``.

    Day 0 returns ``C0`` (an unmodified copy). The input frame is not mutated.
    Vitamin columns, and calcium or potassium when the decay table has a rate,
    are replaced. Mass, kcal, macros, and minerals without a rate are copied
    through. Blank cells stay blank. See :func:`match_decay_rate` for which
    ``k`` is used. The exponent uses :func:`concentration`: yearly ``k`` times
    ``day / 365.25``, not yearly ``k`` times the raw day count.
    """

    if not isinstance(foods, pd.DataFrame):
        raise TypeError("foods must be a DataFrame")
    if not isinstance(decay_table, pd.DataFrame):
        raise TypeError("decay_table must be a DataFrame")
    day_value = float(day)
    if isinstance(day, bool) or not math.isfinite(day_value) or day_value < 0.0:
        raise ValueError(f"day must be >= 0, got {day!r}")
    out = foods.copy()
    if day_value == 0.0 or out.empty:
        return out
    index = _index(decay_table)
    t_years = day_value / DAYS_PER_YEAR
    present = [key for key, column in NUTRIENT_COLUMNS.items() if column in out.columns]
    if not present:
        return out
    rates = np.zeros((len(out), len(present)), dtype=float)
    for row_i, record in enumerate(out.itertuples(index=False)):
        item = _field(record, out.columns, "item")
        food_type = _field(record, out.columns, "food_type")
        for col_i, vitamin in enumerate(present):
            rates[row_i, col_i] = _match(index, item, food_type, vitamin).k_per_year
    factors = np.exp(-rates * t_years)
    for col_i, vitamin in enumerate(present):
        column = NUTRIENT_COLUMNS[vitamin]
        values = pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=float)
        out[column] = values * factors[:, col_i]
    return out


def plan_deltas(
    epochs: Sequence[Mapping],
    mass_g: Mapping[str, float] | None = None,
) -> list[dict]:
    """Serving changes versus the previous feasible epoch of the same kind.

    Infeasible epochs are not a baseline and do not emit rows. Each row has
    ``day``, ``kind``, ``item``, ``previous_servings``, ``new_servings``,
    ``delta``, and ``abs_mass_delta_g`` (``abs(delta) * mass_g`` per serving).
    Sort is descending absolute mass change. Items whose servings did not
    move are omitted. An item that appears or disappears is a change from or
    to zero.
    """

    grams = {} if mass_g is None else mass_g
    grouped: dict[str, list[Mapping]] = {}
    for epoch in epochs:
        kind = str(epoch.get("kind") or "typical")
        grouped.setdefault(kind, []).append(epoch)
    rows: list[dict] = []
    for kind, group in grouped.items():
        ordered = sorted(group, key=lambda epoch: float(epoch.get("day") or 0))
        previous: Mapping[str, float] | None = None
        for epoch in ordered:
            if not epoch.get("feasible"):
                continue
            current = _serving_map(epoch.get("servings"))
            if previous is not None:
                rows.extend(_delta_rows(int(epoch["day"]), kind, previous, current, grams))
            previous = current
    rows.sort(key=lambda row: (-row["abs_mass_delta_g"], -abs(row["delta"]), row["item"], row["kind"], row["day"]))
    return rows


def project_daily(
    days: float,
    foods: pd.DataFrame,
    decay_table: pd.DataFrame,
    locked_servings: Mapping[str, float],
    epoch_plans: Sequence[Mapping],
    targets: Mapping[str, Mapping],
    *,
    kind: str | None = None,
) -> dict:
    """Daily vitamin totals for a locked menu and for epoch menus.

    ``days`` is the mission length. Day 0 through day ``days - 1`` are
    projected. ``locked_servings`` is the day-0 menu (the no-replan path):
    nutrient totals are ``sum(servings * C(day))`` with concentrations
    decaying every day and targets held fixed. A shortfall day is one where
    that total is strictly below the vitamin's minimum, after a 1e-8 relative
    cushion that ignores linear-program roundoff on the bound.

    ``epoch_plans`` hold the servings solved at each epoch start. Inside an
    epoch those servings stay fixed while ``C(day)`` keeps falling, so decay
    between replans shows up. An infeasible epoch contributes no servings
    (totals 0) until the next epoch. Only vitamins that have both a target
    minimum and a column on ``foods`` are projected. Blank concentrations
    count as 0 in the sum, the same way the menu solver treats them.
    """

    n_days = _day_count(days)
    plans = [epoch for epoch in epoch_plans if kind is None or str(epoch.get("kind") or "typical") == kind]
    vitamins = [key for key in VITAMIN_COLUMNS if key in targets and VITAMIN_COLUMNS[key] in foods.columns and _minimum(targets[key]) is not None]
    if n_days == 0 or not vitamins:
        return {"daily": [], "shortfalls": []}

    names = [_item_name(value) for value in foods["item"]] if "item" in foods.columns else [""] * len(foods)
    position = {}
    for index, name in enumerate(names):
        if name and name not in position:
            position[name] = index
    c0 = np.zeros((len(foods), len(vitamins)), dtype=float)
    ks = np.zeros((len(foods), len(vitamins)), dtype=float)
    index = _index(decay_table)
    for row_i, name in enumerate(names):
        food_type = foods["food_type"].iloc[row_i] if "food_type" in foods.columns else None
        for col_i, vitamin in enumerate(vitamins):
            column = VITAMIN_COLUMNS[vitamin]
            raw = foods[column].iloc[row_i]
            number = _k_value(raw)
            c0[row_i, col_i] = 0.0 if number is None else number
            ks[row_i, col_i] = _match(index, name, None if food_type is None else str(food_type), vitamin).k_per_year

    day_index = np.arange(n_days, dtype=float)[:, None, None]
    decayed = c0[None, :, :] * np.exp(-ks[None, :, :] * day_index / DAYS_PER_YEAR)
    locked = _servings_vector(_serving_map(locked_servings), position, len(foods))
    by_day = _servings_by_day(plans, position, n_days, len(foods))
    no_replan = np.einsum("f,dfv->dv", locked, decayed)
    replanned = np.einsum("df,dfv->dv", by_day, decayed)

    daily: list[dict] = []
    shortfalls: list[dict] = []
    for col_i, vitamin in enumerate(vitamins):
        minimum = float(_minimum(targets[vitamin]))
        unit = str(targets[vitamin].get("unit") or "")
        label = str(targets[vitamin].get("label") or VITAMIN_LABELS.get(vitamin, vitamin))
        short_days = 0
        first_day = None
        for day in range(n_days):
            locked_total = float(no_replan[day, col_i])
            short = _below_minimum(locked_total, minimum)
            if short:
                short_days += 1
                if first_day is None:
                    first_day = day
            daily.append(
                {
                    "day": day,
                    "vitamin": vitamin,
                    "label": label,
                    "unit": unit,
                    "target_min": minimum,
                    "no_replan_total": locked_total,
                    "replan_total": float(replanned[day, col_i]),
                    "shortfall_without_replan": short,
                    "kind": kind or "typical",
                }
            )
        if first_day is not None:
            shortfalls.append(
                {
                    "vitamin": vitamin,
                    "label": label,
                    "unit": unit,
                    "first_day": first_day,
                    "days_short": short_days,
                }
            )
    return {"daily": daily, "shortfalls": shortfalls}


def describe_shortfalls(shortfalls: Sequence[Mapping]) -> str:
    """One sentence for the no-replan shortfall list."""

    if not shortfalls:
        return "No vitamin falls short of its minimum on the locked day-0 menu."
    parts = []
    for row in shortfalls:
        label = row.get("label") or row.get("vitamin") or "A vitamin"
        parts.append(f"{label} falls short starting on day {int(row['first_day'])} ({int(row['days_short'])} days)")
    return "Without replanning, " + "; ".join(parts) + "."


def replan_mission(
    days: float,
    crew: Sequence[Mapping] | Mapping,
    *,
    replan_every: float = 30,
    safety_margin: float = 0.10,
    eva_day_fraction: float = 0.0,
    eva_hours: float = 6.5,
    foods: pd.DataFrame | None = None,
    decay_table: pd.DataFrame | None = None,
    allergies: Sequence[str] | str | None = None,
    data_dir: Path | str | None = None,
    solver: Callable | None = None,
) -> dict:
    """Re-solve the crew menu at each epoch as vitamins decay.

    Epoch starts come from :func:`epoch_starts`. At each start ``t``, foods are
    replaced with :func:`decayed_foods` at day ``t`` and :func:`optimizer.optimize_menu`
    is called. The safety margin is not applied inside that solve and does not
    inflate servings. It is recorded on the result so the packing list can keep
    applying it once, separately.

    When ``eva_day_fraction`` is 0, only the typical day is solved (every
    member's EVA hours set to 0). When the fraction is above 0, the typical
    day and the EVA day are both solved at every epoch, using the same crew
    split as :func:`mission.mission_food`. The daily series follows the
    representative day: the EVA menu when every mission day is an EVA day,
    otherwise the typical-day menu.

    The no-replan path locks day-0 servings of that representative menu and
    applies true ``C(day)`` on every later day. The replan path keeps the
    epoch's servings fixed until the next epoch, again with true ``C(day)``,
    so loss between replans is visible. Only epoch starts call the linear
    program. A day is short when the locked total is strictly below the
    minimum after a 1e-8 relative cushion, so a solver that landed on the
    bound is not marked short on day 0.

    An infeasible epoch records ``unmet`` and the day, stores empty servings,
    and the loop continues. ``plan_changes`` compares each feasible epoch with
    the previous feasible epoch of the same kind, heaviest absolute mass
    change first.

    ``solver`` defaults to :func:`optimizer.optimize_menu`. Tests may pass a
    stub with the same keyword arguments.
    """

    mission_days = _positive_number(days, "days")
    margin = _non_negative(safety_margin, "safety_margin")
    fraction = _unit_interval(eva_day_fraction)
    starts = epoch_starts(mission_days, replan_every)
    if foods is None or decay_table is None:
        package = load_all(data_dir)
        if foods is None:
            foods = package.foods
        if decay_table is None:
            decay_table = package.decay
    solve = optimize_menu if solver is None else solver
    representative = "eva" if fraction >= 1.0 - 1e-12 else "typical"
    kinds = ("typical", "eva") if fraction > 0.0 else ("typical",)

    epochs: list[dict] = []
    for start in starts:
        decayed = decayed_foods(foods, start, decay_table)
        for kind in kinds:
            eva = kind == "eva"
            try:
                solved = solve(
                    _crew_for_day(crew, eva=eva, eva_hours=eva_hours),
                    foods=decayed,
                    allergies=allergies,
                    data_dir=data_dir,
                )
            except Exception as exc:
                solved = {
                    "feasible": False,
                    "foods": [],
                    "unmet": [],
                    "total_mass_g": None,
                    "message": f"{type(exc).__name__}: {exc}",
                }
            epochs.append(_epoch_record(start, kind, solved if isinstance(solved, Mapping) else {}))

    mass = _mass_by_item(foods)
    changes = plan_deltas(epochs, mass)
    representative_epochs = [epoch for epoch in epochs if epoch["kind"] == representative]
    day0 = next((epoch for epoch in representative_epochs if epoch["day"] == starts[0]), None)
    locked = day0["servings"] if day0 is not None and day0["feasible"] else {}
    members = _crew_for_day(crew, eva=representative == "eva", eva_hours=eva_hours)
    targets = _vitamin_targets(members, data_dir)
    projected = project_daily(
        mission_days,
        foods,
        decay_table,
        locked,
        representative_epochs,
        targets,
        kind=representative,
    )
    shortfalls = projected["shortfalls"]
    failed = [epoch for epoch in epochs if not epoch["feasible"]]
    return {
        "days": _day_count(mission_days),
        "mission_days": mission_days,
        "replan_every": float(replan_every),
        "epoch_starts": starts,
        "safety_margin": margin,
        "eva_day_fraction": fraction,
        "eva_hours": float(eva_hours),
        "representative_kind": representative,
        "epochs": epochs,
        "plan_changes": changes,
        "daily": projected["daily"],
        "shortfalls": shortfalls,
        "message": _message(starts, shortfalls, margin, failed),
    }


def _message(starts: Sequence[int], shortfalls: Sequence[Mapping], safety_margin: float, failed: Sequence[Mapping]) -> str:
    listed = ", ".join(str(day) for day in starts)
    parts = [
        f"Replanned at day {listed}.",
        f"Safety margin {safety_margin:.0%} is not applied to these daily menus.",
    ]
    if failed:
        bits = []
        for epoch in failed:
            unmet = [str(name) for name in epoch.get("unmet") or []]
            detail = f"day {epoch['day']} {epoch['kind']}"
            if unmet:
                detail = f"{detail} ({', '.join(unmet)})"
            bits.append(detail)
        parts.append("Infeasible epochs: " + "; ".join(bits) + ".")
    parts.append(describe_shortfalls(shortfalls))
    return " ".join(parts)


def _epoch_record(day: int, kind: str, solved: Mapping) -> dict:
    feasible = bool(solved.get("feasible"))
    unmet = [str(name) for name in (solved.get("unmet") or [])]
    message = str(solved.get("message") or "")
    if not feasible:
        return {
            "day": int(day),
            "kind": kind,
            "feasible": False,
            "servings": {},
            "total_mass_g": None,
            "unmet": unmet,
            "message": message,
        }
    return {
        "day": int(day),
        "kind": kind,
        "feasible": True,
        "servings": _serving_map(solved.get("foods")),
        "total_mass_g": None if solved.get("total_mass_g") is None else float(solved["total_mass_g"]),
        "unmet": [],
        "message": message,
    }


def _vitamin_targets(crew: Sequence[Mapping] | Mapping, data_dir: Path | str | None) -> dict[str, dict]:
    members = [crew] if isinstance(crew, Mapping) else list(crew)
    combined: dict[str, dict] = {}
    for member in members:
        if not isinstance(member, Mapping):
            continue
        eva = member.get("eva_hours", 0.0) or 0.0
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
        for vitamin in VITAMIN_COLUMNS:
            entry = day.get(vitamin)
            if not entry or entry.get("minimum") is None:
                continue
            slot = combined.get(vitamin)
            if slot is None:
                combined[vitamin] = {
                    "minimum": float(entry["minimum"]),
                    "unit": str(entry["unit"]),
                    "label": VITAMIN_LABELS.get(vitamin, vitamin),
                }
                continue
            if slot["unit"] != str(entry["unit"]):
                raise ValueError(f"Unit mismatch for {vitamin}")
            slot["minimum"] = float(slot["minimum"]) + float(entry["minimum"])
    return combined


def _servings_by_day(
    plans: Sequence[Mapping],
    position: Mapping[str, int],
    n_days: int,
    n_foods: int,
) -> np.ndarray:
    by_day = np.zeros((n_days, n_foods), dtype=float)
    ordered = sorted(plans, key=lambda epoch: float(epoch.get("day") or 0))
    for index, epoch in enumerate(ordered):
        start = int(epoch.get("day") or 0)
        end = int(ordered[index + 1]["day"]) if index + 1 < len(ordered) else n_days
        start = max(0, min(n_days, start))
        end = max(start, min(n_days, end))
        if start >= end:
            continue
        if epoch.get("feasible"):
            vector = _servings_vector(_serving_map(epoch.get("servings")), position, n_foods)
        else:
            vector = np.zeros(n_foods, dtype=float)
        by_day[start:end, :] = vector
    return by_day


def _servings_vector(servings: Mapping[str, float], position: Mapping[str, int], n_foods: int) -> np.ndarray:
    vector = np.zeros(n_foods, dtype=float)
    for name, count in servings.items():
        slot = position.get(name)
        if slot is None:
            continue
        vector[slot] = float(count)
    return vector


def _delta_rows(
    day: int,
    kind: str,
    previous: Mapping[str, float],
    current: Mapping[str, float],
    mass_g: Mapping[str, float],
) -> list[dict]:
    rows = []
    for item in sorted(set(previous) | set(current)):
        old = float(previous.get(item, 0.0))
        new = float(current.get(item, 0.0))
        delta = new - old
        if abs(delta) <= _SERVING_FLOOR:
            continue
        grams = float(mass_g.get(item, 0.0) or 0.0)
        rows.append(
            {
                "day": day,
                "kind": kind,
                "item": item,
                "previous_servings": old,
                "new_servings": new,
                "delta": delta,
                "abs_mass_delta_g": abs(delta) * grams,
            }
        )
    return rows


def _serving_map(value: object) -> dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        mapped = {}
        for key, count in value.items():
            name = _item_name(key)
            if not name:
                continue
            try:
                number = float(count)
            except (TypeError, ValueError):
                continue
            if number > _SERVING_FLOOR:
                mapped[name] = mapped.get(name, 0.0) + number
        return mapped
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        mapped = {}
        for row in value:
            if not isinstance(row, Mapping) or "item" not in row:
                continue
            name = _item_name(row.get("item"))
            if not name:
                continue
            try:
                number = float(row.get("servings") or 0.0)
            except (TypeError, ValueError):
                continue
            if number > _SERVING_FLOOR:
                mapped[name] = mapped.get(name, 0.0) + number
        return mapped
    return {}


def _mass_by_item(foods: pd.DataFrame) -> dict[str, float]:
    if "item" not in foods.columns or "mass_g" not in foods.columns:
        return {}
    mass: dict[str, float] = {}
    items = foods["item"].tolist()
    grams = pd.to_numeric(foods["mass_g"], errors="coerce").tolist()
    for item, gram in zip(items, grams):
        name = _item_name(item)
        if not name or name in mass or gram is None or not math.isfinite(float(gram)):
            continue
        mass[name] = float(gram)
    return mass


def _below_minimum(total: float, minimum: float) -> bool:
    """True when ``total`` is strictly under ``minimum``.

    The cushion is ``max(1e-6, 1e-8 * |minimum|)``. It absorbs linprog
    roundoff so a menu that meets the bound is not reported short until
    decay actually carries the total under the minimum.
    """

    return total < minimum - max(1e-6, 1e-8 * abs(minimum))


def _minimum(entry: Mapping) -> float | None:
    if not isinstance(entry, Mapping):
        return None
    return _k_value(entry.get("minimum"))


def _match(index: _DecayIndex, item: object, food_type: object, vitamin: str) -> DecayRate:
    key = vitamin if vitamin in NUTRIENT_COLUMNS else _VITAMIN_ALIASES.get(str(vitamin).strip().casefold(), vitamin)
    tags = food_tags(_item_name(item), _food_code(food_type))
    norm = _norm_item(item)
    for row in index.items.get((norm, key), []):
        if row.k is None:
            continue
        k, clipped = _clip_k(row.k)
        return DecayRate(key, k, "decay_item", row.label, clipped)

    specific = _pick(index.food_type.get(key, []), tags, allow_generic=False)
    if specific is not None:
        if specific.k is None:
            return DecayRate(key, 0.0, "food_type", specific.label, False)
        k, clipped = _clip_k(specific.k)
        return DecayRate(key, k, "food_type", specific.label, clipped)

    grouped = _pick(index.group.get(key, []), tags, allow_generic=True)
    if grouped is not None and grouped.k is not None:
        k, clipped = _clip_k(grouped.k)
        return DecayRate(key, k, "decay_group", grouped.label, clipped)

    if "beverage" not in tags:
        for row in index.system.get(key, []):
            if row.k is None:
                continue
            k, clipped = _clip_k(row.k)
            return DecayRate(key, k, "decay_system", row.label, clipped)

    generic = _pick_generic(index.fallback.get(key, []))
    if generic is not None and generic.k is not None:
        k, clipped = _clip_k(generic.k)
        return DecayRate(key, k, "fallback_default", generic.label, clipped)
    return DecayRate(key, 0.0, "stable", "", False)


def _pick(rows: Sequence[_Row], tags: set[str], *, allow_generic: bool) -> _Row | None:
    matched: list[_Row] = []
    for row in rows:
        generic = _is_generic_category(row.category)
        if generic:
            if not allow_generic:
                continue
        elif not _tags_match_category(tags, row.category):
            continue
        matched.append(row)
    if not matched:
        return None
    matched.sort(key=lambda row: (_is_generic_category(row.category), -len(_cat_text(row.category)), row.order))
    best_generic = _is_generic_category(matched[0].category)
    best_len = len(_cat_text(matched[0].category))
    tier = [
        row
        for row in matched
        if _is_generic_category(row.category) == best_generic and len(_cat_text(row.category)) == best_len
    ]
    for row in tier:
        if row.k is not None:
            return row
    return tier[0]


def _pick_generic(rows: Sequence[_Row]) -> _Row | None:
    def rank(row: _Row) -> tuple[int, int]:
        text = " ".join(_category_tokens(row.category))
        if text == "any other food":
            return (0, row.order)
        if text == "any food":
            return (1, row.order)
        return (2, row.order)

    for row in sorted(rows, key=rank):
        if row.k is not None:
            return row
    return None


def _index(decay_table: pd.DataFrame) -> _DecayIndex:
    items: dict[tuple[str, str], list[_Row]] = {}
    food_type: dict[str, list[_Row]] = {}
    group: dict[str, list[_Row]] = {}
    system: dict[str, list[_Row]] = {}
    fallback: dict[str, list[_Row]] = {}
    if decay_table is None or decay_table.empty:
        return _DecayIndex(items, food_type, group, system, fallback)
    columns = set(decay_table.columns)
    for order, record in enumerate(decay_table.itertuples(index=False)):
        record_type = str(getattr(record, "record_type", "") or "")
        if record_type not in {"decay_item", "decay_group", "decay_system", "fallback_default"}:
            continue
        vitamin_raw = getattr(record, "vitamin", None) if "vitamin" in columns or hasattr(record, "vitamin") else None
        keys = _vitamin_keys(vitamin_raw)
        if not keys:
            continue
        category = getattr(record, "category", None)
        label_raw = getattr(record, "item_or_group", None)
        label = "" if label_raw is None or _is_missing(label_raw) else str(label_raw).strip()
        if not label:
            label = _cat_text(category)
        row = _Row(order=order, category=category, label=label, k=_k_value(getattr(record, "k_per_year", None)), norm_name=_norm_item(label_raw))
        if record_type == "decay_item":
            if not row.norm_name:
                continue
            for key in keys:
                items.setdefault((row.norm_name, key), []).append(row)
            continue
        if record_type == "decay_group":
            bucket = group
        elif record_type == "decay_system":
            bucket = system
        elif _is_generic_category(category):
            bucket = fallback
        else:
            bucket = food_type
        for key in keys:
            bucket.setdefault(key, []).append(row)
    return _DecayIndex(items, food_type, group, system, fallback)


def food_tags(item: str, food_type: str | None) -> set[str]:
    """Food-class tags used to match decay categories.

    Beverages and fresh foods do not inherit fruit, vegetable, or meat tags.
    A meat, poultry, or fish name does not also take a fruit or vegetable tag.
    """

    code = _food_code(food_type)
    tokens = set(re.findall(r"[a-z0-9]+", item.casefold())) if item else set()
    if code == "FF":
        return {"fresh"}
    if code == "B":
        return {"beverage"}
    tags: set[str] = set()
    if tokens & _POULTRY:
        tags.update({"chicken", "poultry", "meat"})
    if tokens & _BEEF:
        tags.update({"beef", "meat"})
    if tokens & _FISH or tokens & _PORK:
        tags.add("meat")
    if code == "NF" and tokens & _BREAD:
        tags.add("bread")
    elif tokens & {"tortilla", "tortillas"}:
        tags.add("bread")
    if "meat" not in tags and "chicken" not in tags:
        if tokens & _FRUIT:
            tags.add("fruit")
        if tokens & _VEGETABLE:
            tags.add("vegetable")
    return tags


def _vitamin_keys(raw: object) -> set[str]:
    if _is_missing(raw):
        return set()
    text = str(raw).strip().casefold()
    if text == "all vitamins":
        return set(VITAMIN_COLUMNS)
    if text in {"all measured nutrients", "all"}:
        return set()
    keys: set[str] = set()
    for part in re.split(r"[;,]", text):
        alias = _VITAMIN_ALIASES.get(part.strip())
        if alias:
            keys.add(alias)
    return keys


def _tags_match_category(tags: set[str], category: object) -> bool:
    tokens = set(_category_tokens(category))
    if not tokens or not tags:
        return False
    for tag in tags:
        if tag in tokens:
            return True
        for token in tokens:
            rest = token[len(tag) :] if token.startswith(tag) else None
            if rest in {"s", "es"}:
                return True
    return False


def _is_generic_category(category: object) -> bool:
    text = " ".join(_category_tokens(category))
    return text in _GENERIC_CATEGORIES or text.startswith("any food") or text.startswith("any other")


def _category_tokens(category: object) -> list[str]:
    if _is_missing(category):
        return []
    return re.findall(r"[a-z0-9]+", str(category).casefold())


def _cat_text(category: object) -> str:
    if _is_missing(category):
        return ""
    return str(category).strip()


def _norm_item(name: object) -> str:
    if _is_missing(name):
        return ""
    text = str(name).casefold().replace("*", "")
    text = " ".join(text.split())
    for suffix in (" in pouches", " in pouch"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].rstrip()
    return text


def _food_code(food_type: object) -> str:
    if _is_missing(food_type):
        return ""
    return str(food_type).strip().upper()


def _item_name(value: object) -> str:
    if _is_missing(value):
        return ""
    return str(value).strip()


def _field(record: object, columns: pd.Index, name: str) -> object:
    if name not in columns:
        return None
    return getattr(record, name, None)


def _clip_k(k: float) -> tuple[float, bool]:
    if k < 0.0:
        return 0.0, True
    return float(k), False


def _k_value(raw: object) -> float | None:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        if pd.isna(raw):
            return None
    except TypeError:
        pass
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _positive_number(value: float, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be positive, got {value!r}") from exc
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be positive, got {value!r}")
    return number


def _non_negative(value: float, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be >= 0, got {value!r}") from exc
    if isinstance(value, bool) or not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be >= 0, got {value!r}")
    return number


def _unit_interval(value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"eva_day_fraction must be between 0 and 1 inclusive, got {value!r}") from exc
    if isinstance(value, bool) or not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError(f"eva_day_fraction must be between 0 and 1 inclusive, got {value!r}")
    return number


def _day_count(days: float) -> int:
    return int(math.ceil(float(days) - 1e-9))
