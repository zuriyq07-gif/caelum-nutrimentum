"""Energy, micronutrient targets, shelf-life decay, and a one-day menu.

SciPy is used in three places:

* ``scipy.optimize.milp`` chooses whole packages against the day's targets.
* ``scipy.optimize.curve_fit`` and ``scipy.stats.linregress`` compare published
  vitamin-loss rates with a fit to the 0-, 1-, and 3-year measurements.
* ``scipy.integrate.solve_ivp`` draws the first-order curve dC/dt = -k C.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import OptimizeWarning, curve_fit, milp, Bounds, LinearConstraint
from scipy.stats import linregress

MICRO_NUTRIENTS = [
    ("Potassium", "potassium_mg", "mg"),
    ("Calcium", "calcium_mg", "mg"),
    ("Magnesium", "magnesium_mg", "mg"),
    ("Iron", "iron_mg", "mg"),
    ("Zinc", "zinc_mg", "mg"),
    ("Phosphorus", "phosphorus_mg", "mg"),
    ("Vitamin A", "vit_a_rae_ug_per_serving_est", "ug"),
    ("Vitamin C", "vit_c_mg_per_serving_est", "mg"),
    ("Vitamin E", "vit_e_mg_per_serving_est", "mg"),
    ("Vitamin K", "vit_k_ug_per_serving_est", "ug"),
    ("Thiamin", "thiamin_b1_mg_per_serving_est", "mg"),
    ("Riboflavin", "riboflavin_b2_mg_per_serving_est", "mg"),
    ("Niacin", "niacin_b3_mg_per_serving_est", "mg"),
    ("Vitamin B6", "vit_b6_mg_per_serving_est", "mg"),
    ("Vitamin B12", "vit_b12_ug_per_serving_est", "ug"),
    ("Folate", "folate_dfe_ug_per_serving_est", "ug"),
]

ENERGY_SHARE = {
    "Protein": 4.0,
    "Carbohydrate": 4.0,
    "Fat": 9.0,
    "Saturated fat": 9.0,
}


@dataclass(frozen=True)
class EerResult:
    sex: str
    age: float
    mass_kg: float
    height_m: float
    activity_factor: float
    eva_hours: float
    formula: str
    intercept: float
    age_coef: float
    mass_coef: float
    height_coef: float
    age_term: float
    activity_term: float
    base_kcal: float
    eva_kcal: float
    eer_kcal: float
    provision_kcal: float


@dataclass
class DayPlan:
    success: bool
    message: str
    target_kcal: float
    group: str
    servings: pd.DataFrame
    totals: pd.DataFrame
    foods_considered: int
    packages_chosen: int
    objective: float | None


@dataclass(frozen=True)
class DecayEstimate:
    vitamin: str
    category: str | None
    k_per_year: float | None
    t_years: float
    c0: float
    amount: float | None
    fraction_remaining: float | None
    half_life_years: float | None
    value_status: str
    note: str


def eer(
    sex: str,
    age: float,
    kg: float,
    m: float,
    af: float | None = None,
    eva_h: float = 0.0,
    rules: dict | None = None,
) -> float:
    """Estimated energy requirement in kcal/day, including EVA hours."""

    return eer_components(sex, age, kg, m, af=af, eva_h=eva_h, rules=rules).eer_kcal


def eer_components(
    sex: str,
    age: float,
    kg: float,
    m: float,
    af: float | None = None,
    eva_h: float = 0.0,
    rules: dict | None = None,
) -> EerResult:
    if rules is None:
        raise ValueError("energy rules are required")
    if sex not in {"M", "F"}:
        raise ValueError("sex must be 'M' or 'F'")
    equations = rules["eer_equations"]
    key = "male_19plus" if sex == "M" else "female_19plus"
    equation = equations[key]
    if af is None:
        af = float(equations["activity_factor"]["nominal"])
    intercept = float(equation["intercept"])
    age_coef = float(equation["age_coef"])
    mass_coef = float(equation["mass_coef"])
    height_coef = float(equation["height_coef"])
    age_term = age_coef * float(age)
    activity_term = float(af) * (mass_coef * float(kg) + height_coef * float(m))
    base = intercept + age_term + activity_term
    eva_kcal = float(eva_h) * float(rules["eva_energy"]["extra_kcal_per_eva_hour"])
    return EerResult(
        sex=sex,
        age=float(age),
        mass_kg=float(kg),
        height_m=float(m),
        activity_factor=float(af),
        eva_hours=float(eva_h),
        formula=str(equation["formula"]),
        intercept=intercept,
        age_coef=age_coef,
        mass_coef=mass_coef,
        height_coef=height_coef,
        age_term=age_term,
        activity_term=activity_term,
        base_kcal=base,
        eva_kcal=eva_kcal,
        eer_kcal=base + eva_kcal,
        provision_kcal=float(rules["default_daily_energy"]["kcal_per_crewmember_per_day"]),
    )


def life_stage_group(sex: str, age: float) -> str:
    """DRI group label used in ``micronutrient_requirements.csv``."""

    if sex not in {"M", "F"}:
        raise ValueError("sex must be 'M' or 'F'")
    if age < 19:
        raise ValueError("DRI tables in this package start at age 19")
    label = "male" if sex == "M" else "female"
    if age <= 30:
        band = "19-30"
    elif age <= 50:
        band = "31-50"
    elif age <= 70:
        band = "51-70"
    else:
        band = ">70"
    return f"{label} {band}"


def daily_targets(
    requirements: pd.DataFrame,
    rules: dict,
    sex: str,
    age: float,
    kg: float,
    kcal: float,
) -> pd.DataFrame:
    """Targets for one crewmember-day.

    Micronutrients follow the DRI for the life stage. Vitamin D is replaced
    with the NASA-STD-3001 value of 25 µg/day. Macro bands use the DRI AMDR
    against ``kcal``. Protein also has the 0.8 g/kg floor and the 35% ceiling.
    """

    group = life_stage_group(sex, age)
    vitamin_d = float(rules["spaceflight_micronutrient_exception"]["vitamin_d_ug_per_day"])
    sodium_cap = 2300.0
    history_sodium = rules["historical_iss_intake_reference"]["sodium_mg_per_day"]
    rows: list[dict] = [
        _target_row(
            "Energy",
            "kcal",
            "kcal",
            "energy",
            kcal,
            kcal,
            "EER from NASA-STD-3001 Table 7.1-1, plus 200 kcal per EVA hour.",
        ),
        _target_row(
            "Protein",
            "protein_g",
            "g",
            "protein",
            0.8 * kg,
            0.35 * kcal / 4.0,
            "Floor is 0.8 g/kg body mass. Ceiling is 35% of EER at 4 kcal/g.",
        ),
        _target_row(
            "Carbohydrate",
            "carb_g",
            "g",
            "energy_pct",
            0.45 * kcal / 4.0,
            0.65 * kcal / 4.0,
            "DRI AMDR, 45–65% of EER. HIDH suggests 50–55%.",
        ),
        _target_row(
            "Fat",
            "fat_g",
            "g",
            "energy_pct",
            0.20 * kcal / 9.0,
            0.35 * kcal / 9.0,
            "DRI AMDR, 20–35% of EER. HIDH suggests 25–35%.",
        ),
        _target_row(
            "Saturated fat",
            "sat_fat_g",
            "g",
            "max",
            None,
            0.07 * kcal / 9.0,
            "HIDH ceiling, 7% of EER.",
        ),
        _target_row(
            "Fiber",
            "fiber_g",
            "g",
            "min",
            10.0 * kcal / 1000.0,
            None,
            "HIDH lower bound, 10 g per 1,000 kcal.",
        ),
        _target_row(
            "Sodium",
            "sodium_mg",
            "mg",
            "max",
            None,
            sodium_cap,
            f"Chronic disease risk reduction intake, 2,300 mg. DRI AI is 1,500 mg. "
            f"ISS expeditions E26–E34 averaged {history_sodium}.",
        ),
    ]

    dri_vitamin_d = _requirement_value(requirements, "Vitamin D", group)
    rows.append(
        _target_row(
            "Vitamin D",
            "vit_d_ug_per_serving_est",
            "ug",
            "min",
            vitamin_d,
            _optional_ul(requirements, "Vitamin D", group),
            f"NASA-STD-3001 Rev E sets 25 µg/day for all crew. The DRI for {group} is {dri_vitamin_d:g} µg. "
            "ISS also carries vitamin D tablets.",
        )
    )
    for nutrient, column, unit in MICRO_NUTRIENTS:
        row = _lookup_requirement(requirements, nutrient, group)
        unit_text = str(row["unit"]).replace("/d", "")
        food_ul, ul_note = _food_ul(row)
        note = f"DRI {row['value_type']} for {group}."
        if ul_note:
            note = f"{note} {ul_note}"
        rows.append(
            _target_row(
                nutrient,
                column,
                unit_text or unit,
                "min",
                float(row["rda_or_ai"]),
                food_ul,
                note,
            )
        )
    return pd.DataFrame(rows)


def plan_day(
    foods: pd.DataFrame,
    requirements: pd.DataFrame,
    rules: dict,
    sex: str,
    age: float,
    kg: float,
    height_m: float,
    af: float | None = None,
    eva_h: float = 0.0,
    include_flagged: bool = False,
    max_packages: int = 1,
    max_beverages: int = 4,
) -> DayPlan:
    """Pick whole packages to meet the day's targets.

    Decision variables are integer package counts. Shortfalls and excesses are
    continuous slack variables, so a solution exists even when the menu cannot
    hit every target. USDA blanks are treated as zero inside the solver, which
    makes vitamin coverage a lower bound.
    """

    components = eer_components(sex, age, kg, height_m, af=af, eva_h=eva_h, rules=rules)
    targets = daily_targets(requirements, rules, sex, age, kg, components.eer_kcal)
    group = life_stage_group(sex, age)
    empty_servings = pd.DataFrame(columns=["item", "packages"])

    use = foods.copy()
    if not include_flagged and "quality_flag" in use.columns:
        use = use.loc[~use["quality_flag"].fillna(False).astype(bool)]
    use = use.dropna(subset=["kcal", "protein_g", "carb_g", "fat_g"])
    use = use.loc[use["kcal"] > 0].reset_index(drop=True)
    if use.empty:
        return DayPlan(
            False,
            "No foods with usable macros are available for these filters.",
            components.eer_kcal,
            group,
            empty_servings,
            _totals_from_amounts(targets, {}, components.eer_kcal),
            0,
            0,
            None,
        )

    n = len(use)
    target_kcal = float(components.eer_kcal)
    names = [
        "energy_under",
        "energy_over",
        "protein_under",
        "protein_over",
        "carb_under",
        "carb_over",
        "fat_under",
        "fat_over",
        "sat_over",
        "fiber_under",
        "sodium_over",
    ]
    micro_rows = []
    for row in targets.itertuples(index=False):
        if row.kind != "min" or not row.food_column or pd.isna(row.target_low):
            continue
        if row.food_column not in use.columns:
            raise KeyError(f"Food table has no column {row.food_column}")
        micro_rows.append(row)
        names.append(f"under::{row.nutrient}")

    n_slack = len(names)
    total = n + n_slack
    cost = np.zeros(total)
    cost[n + names.index("energy_under")] = 10.0
    cost[n + names.index("energy_over")] = 4.0
    cost[n + names.index("protein_under")] = 25.0
    cost[n + names.index("protein_over")] = 8.0
    cost[n + names.index("carb_under")] = 3.0
    cost[n + names.index("carb_over")] = 3.0
    cost[n + names.index("fat_under")] = 3.0
    cost[n + names.index("fat_over")] = 3.0
    cost[n + names.index("sat_over")] = 5.0
    cost[n + names.index("fiber_under")] = 12.0
    cost[n + names.index("sodium_over")] = 0.08
    for row in micro_rows:
        cost[n + names.index(f"under::{row.nutrient}")] = 150.0 / float(row.target_low)
    cost[:n] = 0.6

    def food_vector(column: str, fill_from: str | None = None) -> np.ndarray:
        series = use[column]
        if fill_from is not None:
            series = series.fillna(use[fill_from])
        return series.fillna(0).to_numpy(dtype=float)

    kcal = food_vector("kcal")
    protein = food_vector("protein_g")
    carb = food_vector("carb_g")
    fat = food_vector("fat_g")
    sat = food_vector("sat_fat_g", fill_from="fat_g")
    fiber = food_vector("fiber_g")
    sodium = food_vector("sodium_mg")

    constraint_rows: list[np.ndarray] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []

    def add(coeffs_by_name: dict[str, float], food_coeffs: np.ndarray, low: float, high: float) -> None:
        row = np.zeros(total)
        row[:n] = food_coeffs
        for name, value in coeffs_by_name.items():
            row[n + names.index(name)] = value
        constraint_rows.append(row)
        lower_bounds.append(low)
        upper_bounds.append(high)

    add({"energy_under": 1.0, "energy_over": -1.0}, kcal, target_kcal, target_kcal)
    protein_row = targets.set_index("nutrient").loc["Protein"]
    add({"protein_under": 1.0}, protein, float(protein_row.target_low), np.inf)
    add({"protein_over": -4.0}, 4.0 * protein, -np.inf, 0.35 * target_kcal)
    carb_row = targets.set_index("nutrient").loc["Carbohydrate"]
    add({"carb_under": 1.0}, 4.0 * carb, float(carb_row.target_low) * 4.0, np.inf)
    add({"carb_over": -1.0}, 4.0 * carb, -np.inf, float(carb_row.target_high) * 4.0)
    fat_row = targets.set_index("nutrient").loc["Fat"]
    add({"fat_under": 1.0}, 9.0 * fat, float(fat_row.target_low) * 9.0, np.inf)
    add({"fat_over": -1.0}, 9.0 * fat, -np.inf, float(fat_row.target_high) * 9.0)
    sat_row = targets.set_index("nutrient").loc["Saturated fat"]
    add({"sat_over": -1.0}, 9.0 * sat, -np.inf, float(sat_row.target_high) * 9.0)
    fiber_row = targets.set_index("nutrient").loc["Fiber"]
    add({"fiber_under": 1.0}, fiber, float(fiber_row.target_low), np.inf)
    sodium_row = targets.set_index("nutrient").loc["Sodium"]
    add({"sodium_over": -1.0}, sodium, -np.inf, float(sodium_row.target_high))
    for row in micro_rows:
        add({f"under::{row.nutrient}": 1.0}, food_vector(row.food_column), float(row.target_low), np.inf)
    if max_beverages is not None:
        beverages = use["food_type"].eq("B").to_numpy(dtype=float)
        add({}, beverages, -np.inf, float(max_beverages))

    bounds = Bounds(
        lb=np.zeros(total),
        ub=np.concatenate([np.full(n, float(max_packages)), np.full(n_slack, np.inf)]),
    )
    integrality = np.zeros(total, dtype=int)
    integrality[:n] = 1
    result = milp(
        c=cost,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(np.vstack(constraint_rows), lower_bounds, upper_bounds),
        options={"time_limit": 20, "mip_rel_gap": 0.01, "disp": False},
    )
    if not result.success or result.x is None:
        message = result.message if result.message else "The solver did not return a menu."
        return DayPlan(
            False,
            f"HiGHS did not return a menu. {message}",
            target_kcal,
            group,
            empty_servings,
            _totals_from_amounts(targets, {}, target_kcal),
            n,
            0,
            None,
        )

    packages = np.rint(result.x[:n]).astype(int)
    if np.max(np.abs(result.x[:n] - packages)) > 1e-4:
        return DayPlan(
            False,
            "HiGHS returned package counts that are not integers.",
            target_kcal,
            group,
            empty_servings,
            _totals_from_amounts(targets, {}, target_kcal),
            n,
            0,
            float(result.fun),
        )

    amounts = _achieved_amounts(use, packages, targets)
    totals = _totals_from_amounts(targets, amounts, float(amounts.get("Energy", target_kcal)))
    servings = _servings_table(use, packages)
    chosen = int(packages.sum())
    return DayPlan(
        True,
        f"HiGHS selected {chosen} packages from {n} eligible foods. "
        "At most one package of each item, and at most "
        f"{max_beverages} beverages. Blank USDA vitamin cells count as zero in the solver.",
        target_kcal,
        group,
        servings,
        totals,
        n,
        chosen,
        float(result.fun),
    )


def remaining(
    c0: float,
    vitamin: str,
    t_years: float,
    category: str = "any food",
    decay: pd.DataFrame | None = None,
) -> DecayEstimate:
    """Vitamin left after storage, using the most specific fallback rate.

    Closed form from the data package: ``C(t) = C0 * exp(-k t)``.
    """

    if decay is None:
        raise ValueError("decay table is required")
    row, k = _fallback_row(decay, vitamin, category)
    if row is None:
        return DecayEstimate(
            vitamin=vitamin,
            category=None,
            k_per_year=None,
            t_years=float(t_years),
            c0=float(c0),
            amount=None,
            fraction_remaining=None,
            half_life_years=None,
            value_status="",
            note=f"No fallback row for {vitamin}.",
        )
    note = "" if pd.isna(row.get("notes", pd.NA)) else str(row["notes"])
    status = "" if pd.isna(row.get("value_status", pd.NA)) else str(row["value_status"])
    matched = None if pd.isna(row.get("category", pd.NA)) else str(row["category"])
    if k is None:
        return DecayEstimate(
            vitamin=vitamin,
            category=matched,
            k_per_year=None,
            t_years=float(t_years),
            c0=float(c0),
            amount=None,
            fraction_remaining=None,
            half_life_years=None,
            value_status=status,
            note=note or "No numeric decay rate for this vitamin.",
        )
    fraction = math.exp(-k * float(t_years))
    half_life = math.log(2) / k if k > 1e-8 else None
    return DecayEstimate(
        vitamin=vitamin,
        category=matched,
        k_per_year=k,
        t_years=float(t_years),
        c0=float(c0),
        amount=float(c0) * fraction,
        fraction_remaining=fraction,
        half_life_years=half_life,
        value_status=status,
        note=note,
    )


def decay_curve(c0: float, k: float, t_end: float, n: int = 121) -> tuple[np.ndarray, np.ndarray]:
    """Integrate dC/dt = -k C from 0 to ``t_end`` years."""

    c0 = float(c0)
    k = float(k)
    t_end = float(t_end)
    if n < 2:
        raise ValueError("n must be at least 2")
    if t_end == 0:
        return np.array([0.0]), np.array([c0])
    times = np.linspace(0.0, t_end, n)
    solution = solve_ivp(
        lambda _t, y: [-k * y[0]],
        (0.0, t_end),
        [c0],
        t_eval=times,
        rtol=1e-7,
        atol=1e-8,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.t, solution.y[0]


def fit_decay_items(decay: pd.DataFrame) -> pd.DataFrame:
    """Fit first-order k to each measured decay item.

    Published ``k_per_year`` uses the 0-year and 3-year points only.
    ``k_curve_fit`` uses all three points with the starting concentration fixed.
    ``k_loglinear`` is a log-linear regression that lets the intercept move.
    """

    columns = [
        "item_or_group",
        "vitamin",
        "c_initial",
        "c_1yr",
        "c_3yr",
        "k_published",
        "k_curve_fit",
        "k_loglinear",
        "rmse",
        "fit_note",
    ]
    if decay.empty or "c_initial" not in decay.columns:
        return pd.DataFrame(columns=columns)
    work = decay
    if "record_type" in decay.columns:
        work = decay.loc[decay["record_type"].eq("decay_item")]
    rows = []
    for record in work.itertuples(index=False):
        measured = _fit_one(record)
        if measured is not None:
            rows.append(measured)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def _fit_one(record) -> dict | None:
    try:
        c0 = float(record.c_initial)
        c1 = float(record.c_1yr)
        c3 = float(record.c_3yr)
    except (TypeError, ValueError):
        return None
    if not np.isfinite([c0, c1, c3]).all() or c0 <= 0:
        return None
    published = getattr(record, "k_per_year", np.nan)
    published = float(published) if pd.notna(published) else np.nan
    times = np.array([0.0, 1.0, 3.0])
    observed = np.array([c0, c1, c3], dtype=float)
    note = ""
    fitted = np.nan
    rmse = np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _pcov = curve_fit(
                lambda t, k: c0 * np.exp(-k * t),
                times,
                observed,
                p0=[0.1 if not np.isfinite(published) else float(np.clip(published, -1.9, 4.9))],
                bounds=(-2.0, 5.0),
            )
        fitted = float(popt[0])
        predicted = c0 * np.exp(-fitted * times)
        rmse = float(np.sqrt(np.mean((predicted - observed) ** 2)))
    except (RuntimeError, ValueError) as exc:
        note = f"curve_fit failed: {exc}"
    log_k = np.nan
    if np.all(observed > 0):
        slope, _intercept, _r, _p, _se = linregress(times, np.log(observed))
        log_k = float(-slope)
    return {
        "item_or_group": getattr(record, "item_or_group", ""),
        "vitamin": getattr(record, "vitamin", ""),
        "c_initial": c0,
        "c_1yr": c1,
        "c_3yr": c3,
        "k_published": published,
        "k_curve_fit": fitted,
        "k_loglinear": log_k,
        "rmse": rmse,
        "fit_note": note,
    }


def _fallback_row(decay: pd.DataFrame, vitamin: str, category: str) -> tuple[pd.Series | None, float | None]:
    fallback = decay.loc[decay["record_type"].eq("fallback_default")].copy()
    vitamin_rows = fallback.loc[fallback["vitamin"].eq(vitamin)]
    if vitamin_rows.empty:
        return None, None
    categories = vitamin_rows["category"].fillna("").astype(str)
    exact = vitamin_rows.loc[categories.eq(category)]
    if exact.empty:
        contains = vitamin_rows.loc[categories.str.contains(category, regex=False)]
        if not contains.empty:
            contains = contains.assign(_len=contains["category"].fillna("").astype(str).str.len())
            exact = contains.sort_values("_len")
    if exact.empty:
        for generic in ("any other food", "any food"):
            generic_rows = vitamin_rows.loc[categories.eq(generic)]
            if not generic_rows.empty:
                exact = generic_rows
                break
    if exact.empty:
        exact = vitamin_rows
    exact = exact.copy()
    exact["_k"] = pd.to_numeric(exact["k_per_year"], errors="coerce")
    known = exact.loc[exact["_k"].notna()]
    chosen = known.iloc[0] if not known.empty else exact.iloc[0]
    k_value = chosen["_k"]
    k = None if pd.isna(k_value) else float(k_value)
    return chosen, k


def serving_against_targets(food: pd.Series, targets: pd.DataFrame) -> pd.DataFrame:
    """One package compared with a crewmember's daily targets."""

    rows = []
    for row in targets.itertuples(index=False):
        amount = food[row.food_column] if row.food_column in food.index else np.nan
        if row.kind == "max" or (pd.isna(row.target_low) and pd.notna(row.target_high)):
            reference = row.target_high
            compared_with = "maximum"
        else:
            reference = row.target_low
            compared_with = "minimum"
        percent = np.nan
        if pd.notna(amount) and pd.notna(reference) and float(reference) != 0:
            percent = float(amount) / float(reference) * 100.0
        rows.append(
            {
                "nutrient": row.nutrient,
                "per_serving": amount if pd.notna(amount) else np.nan,
                "unit": row.unit,
                "minimum": row.target_low,
                "maximum": row.target_high,
                "percent_of_reference": percent,
                "reference": compared_with,
            }
        )
    return pd.DataFrame(rows)


def _achieved_amounts(foods: pd.DataFrame, packages: np.ndarray, targets: pd.DataFrame) -> dict[str, float]:
    amounts: dict[str, float] = {}
    for row in targets.itertuples(index=False):
        column = row.food_column
        if column not in foods.columns:
            continue
        vector = foods[column].fillna(0).to_numpy(dtype=float)
        if row.nutrient == "Saturated fat":
            vector = foods[column].fillna(foods["fat_g"]).fillna(0).to_numpy(dtype=float)
        amounts[row.nutrient] = float(np.dot(packages, vector))
    return amounts


def _totals_from_amounts(targets: pd.DataFrame, amounts: dict[str, float], achieved_kcal: float) -> pd.DataFrame:
    rows = []
    for row in targets.itertuples(index=False):
        amount = amounts.get(row.nutrient, np.nan)
        share = np.nan
        if row.nutrient in ENERGY_SHARE and achieved_kcal and np.isfinite(amount):
            share = ENERGY_SHARE[row.nutrient] * amount / achieved_kcal * 100.0
        rows.append(
            {
                "nutrient": row.nutrient,
                "amount": amount,
                "unit": row.unit,
                "target_low": row.target_low,
                "target_high": row.target_high,
                "share_of_energy_pct": share,
                "status": _status(amount, row.target_low, row.target_high, row.kind),
                "note": row.note,
            }
        )
    return pd.DataFrame(rows)


def _servings_table(foods: pd.DataFrame, packages: np.ndarray) -> pd.DataFrame:
    chosen = packages > 0
    if not chosen.any():
        return pd.DataFrame(columns=["item", "packages"])
    columns = [
        "item",
        "food_type",
        "food_type_name",
        "mass_g",
        "kcal",
        "protein_g",
        "carb_g",
        "fat_g",
        "sat_fat_g",
        "fiber_g",
        "sodium_mg",
        "potassium_mg",
        "calcium_mg",
        "iron_mg",
        "vit_c_mg_per_serving_est",
        "vit_d_ug_per_serving_est",
    ]
    present = [column for column in columns if column in foods.columns]
    out = foods.loc[chosen, present].copy()
    counts = packages[chosen]
    out.insert(1, "packages", counts)
    for column in present:
        if column in {"item", "food_type", "food_type_name"}:
            continue
        out[column] = foods.loc[chosen, column].to_numpy(dtype=float) * counts
    return out.sort_values("kcal", ascending=False).reset_index(drop=True)


def _status(amount, low, high, kind: str) -> str:
    if pd.isna(amount):
        return "unknown"
    if kind == "energy":
        if low is None or pd.isna(low) or float(low) == 0:
            return "unknown"
        gap = abs(float(amount) - float(low)) / abs(float(low))
        if gap <= 0.02:
            return "met"
        return "under" if float(amount) < float(low) else "over"
    under = pd.notna(low) and float(amount) < float(low) - _tolerance(low)
    over = pd.notna(high) and float(amount) > float(high) + _tolerance(high)
    if under:
        return "under"
    if over:
        return "over"
    return "met"


def _tolerance(value) -> float:
    if value is None or pd.isna(value):
        return 0.05
    return max(0.05, 0.01 * abs(float(value)))


def _target_row(nutrient, column, unit, kind, low, high, note) -> dict:
    return {
        "nutrient": nutrient,
        "food_column": column,
        "unit": unit,
        "kind": kind,
        "target_low": None if low is None or (isinstance(low, float) and math.isnan(low)) else float(low),
        "target_high": None if high is None or (isinstance(high, float) and math.isnan(high)) else float(high),
        "note": note,
    }


def _lookup_requirement(requirements: pd.DataFrame, nutrient: str, group: str) -> pd.Series:
    hit = requirements.loc[requirements["nutrient"].eq(nutrient) & requirements["group"].eq(group)]
    if len(hit) != 1:
        raise KeyError(f"Expected one requirement row for {nutrient} / {group}, found {len(hit)}")
    return hit.iloc[0]


def _requirement_value(requirements: pd.DataFrame, nutrient: str, group: str) -> float:
    return float(_lookup_requirement(requirements, nutrient, group)["rda_or_ai"])


def _optional_ul(requirements: pd.DataFrame, nutrient: str, group: str) -> float | None:
    food_ul, _note = _food_ul(_lookup_requirement(requirements, nutrient, group))
    return food_ul


def _food_ul(row: pd.Series) -> tuple[float | None, str]:
    """Upper limit that applies to food, plus a note when the published UL does not.

    Several DRI caps cover supplements, synthetic forms, or preformed vitamin A.
    Those caps are not used as a food-menu maximum.
    """

    ul = _finite(row["ul"])
    if ul is None:
        return None, ""
    note = "" if pd.isna(row["notes"]) else str(row["notes"]).lower()
    limited = any(token in note for token in ("supplement", "synthetic", "preformed", "not food"))
    if not limited:
        return ul, ""
    return None, f"Published UL is {ul:g}; it is not applied here because the source limits supplements or a specific chemical form."


def _finite(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number
