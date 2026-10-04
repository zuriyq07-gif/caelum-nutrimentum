"""ISS mission food load: pack the standard menu for a crew.

The page collects mission inputs and calls :func:`mission.mission_food`.
Nutrient targets and the menu linear program stay in ``targets`` and ``optimizer``.
"""

from __future__ import annotations

import csv
import io
import math
from typing import Mapping, Sequence

import altair as alt
import pandas as pd
import streamlit as st

from data_loader import load_all
from mission import mission_food
from optimizer import FOOD_COLUMNS

# A standard EVA day in mission_food. Seven of those days is 45.5 hours.
STANDARD_EVA_HOURS = 6.5
DAYS_PER_WEEK = 7
MAX_STANDARD_WEEKLY_EVA = DAYS_PER_WEEK * STANDARD_EVA_HOURS
# One nutrient at thousands of percent would flatten every other bar.
VISUAL_PERCENT_CAP = 250.0

NUTRIENT_LABELS = {
    "energy": "Energy",
    "protein": "Protein",
    "carbohydrate": "Carbohydrate",
    "fat": "Fat",
    "saturated_fat": "Saturated fat",
    "fiber": "Fiber",
    "sodium": "Sodium",
    "potassium": "Potassium",
    "calcium": "Calcium",
    "magnesium": "Magnesium",
    "iron": "Iron",
    "zinc": "Zinc",
    "phosphorus": "Phosphorus",
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

DEFAULT_CREW = (
    {"age": 45, "sex": "Male", "weight_kg": 82.9, "height_m": 1.80},
    {"age": 32, "sex": "Female", "weight_kg": 65.1, "height_m": 1.70},
    {"age": 38, "sex": "Male", "weight_kg": 75.0, "height_m": 1.78},
    {"age": 29, "sex": "Female", "weight_kg": 62.0, "height_m": 1.68},
)

_NUTRIENT_ORDER = {name: index for index, name in enumerate(FOOD_COLUMNS)}


def eva_schedule(weekly_hours: float) -> dict:
    """Map mission EVA hours per week to ``mission_food`` arguments.

    Zero hours is a typical-day mission (``eva_day_fraction`` 0 and
    ``eva_hours`` 0). Up to seven standard 6.5-hour days (45.5 h), the
    fraction is weekly hours / 45.5 and each EVA day stays 6.5 hours.
    Above that, every day is an EVA day and ``eva_hours`` is the weekly
    total divided by 7. ``note`` is set only in that last case.

    Raises:
        ValueError: ``weekly_hours`` is negative or not a finite number.
    """

    hours = _require_finite(weekly_hours, "EVA hours per week")
    if hours < 0.0:
        raise ValueError(f"EVA hours per week cannot be negative, got {weekly_hours!r}")
    if hours == 0.0:
        return {"eva_day_fraction": 0.0, "eva_hours": 0.0, "note": None}
    if hours <= MAX_STANDARD_WEEKLY_EVA:
        return {
            "eva_day_fraction": hours / MAX_STANDARD_WEEKLY_EVA,
            "eva_hours": STANDARD_EVA_HOURS,
            "note": None,
        }
    per_day = hours / DAYS_PER_WEEK
    return {
        "eva_day_fraction": 1.0,
        "eva_hours": per_day,
        "note": (
            f"Weekly EVA time is above {MAX_STANDARD_WEEKLY_EVA:g} hours, so every day "
            f"is treated as an EVA day of {per_day:.2f} hours."
        ),
    }


def validate_mission_inputs(
    days: object,
    safety_margin: object,
    eva_hours_per_week: object,
    crew: Sequence[Mapping],
) -> list[str]:
    """Return human-readable problems. An empty list means the planner may run.

    Ages under 19 are rejected here because ``targets.daily_targets`` raises
    ``ValueError`` for those ages. Days must be positive, weight and height
    must be positive, and the safety margin must be at least zero.
    """

    errors: list[str] = []
    day_count = _as_float(days)
    if day_count is None or day_count <= 0.0:
        errors.append("Mission length has to be more than zero days.")
    margin = _as_float(safety_margin)
    if margin is None or margin < 0.0:
        errors.append("Safety margin cannot be negative. Use 0 for no extra food, or 0.10 for 10 percent.")
    weekly = _as_float(eva_hours_per_week)
    if weekly is None or weekly < 0.0:
        errors.append("EVA hours per week cannot be negative.")
    if not crew:
        errors.append("Add at least one crew member.")
    for index, member in enumerate(crew, start=1):
        errors.extend(_validate_member(index, member))
    return errors


def packing_list_csv(packing_list: Sequence[Mapping]) -> str:
    """CSV text with columns food, servings, mass_kg and a labeled total row."""

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["food", "servings", "mass_kg"])
    total = 0.0
    for row in packing_list:
        servings = float(row["servings"])
        mass_kg = float(row["mass_kg"])
        writer.writerow([str(row["item"]), f"{servings:.2f}", f"{mass_kg:.3f}"])
        total += mass_kg
    writer.writerow(["Total", "", f"{total:.3f}"])
    return buffer.getvalue()


def mission_average_coverage(result: Mapping) -> list[dict]:
    """Mission-average delivered nutrients as a percent of the minimum target.

    For each nutrient::

        delivered = (typical_amount * typical_days + eva_amount * eva_days) / days
        target = (typical_minimum * typical_days + eva_minimum * eva_days) / days
        percent = delivered / target * 100

    A day with zero length does not have to carry a minimum. A day that does
    count, and is missing its minimum, drops the nutrient. A weighted minimum
    of 0 is skipped. ``display_percent`` is capped for the axis; ``percent``
    stays uncapped and ``clipped`` records the cap.
    """

    days = float(result["days"])
    if not math.isfinite(days) or days <= 0.0:
        return []
    typical_days = float(result["typical_days"])
    eva_days = float(result["eva_days"])
    typical = (result.get("typical_day") or {}).get("nutrients") or {}
    eva = (result.get("eva_day") or {}).get("nutrients") or {}
    rows: list[dict] = []
    for name in list(dict.fromkeys([*typical.keys(), *eva.keys()])):
        row = _coverage_row(name, typical.get(name) or {}, eva.get(name) or {}, typical_days, eva_days, days)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda row: (_NUTRIENT_ORDER.get(row["nutrient"], 10_000), row["label"]))
    return rows


def nutrient_label(name: str) -> str:
    if name in NUTRIENT_LABELS:
        return NUTRIENT_LABELS[name]
    return name.replace("_", " ").capitalize()


def display_unit(unit: object) -> str:
    text = "" if unit is None else str(unit)
    if text == "ug":
        return "µg"
    return text


def main() -> None:
    st.set_page_config(page_title="Mission food load", layout="wide")
    _css()
    st.title("Mission food load")
    st.caption(
        "Pack the ISS standard menu for the crew on this flight. The planner solves a "
        "minimum-mass day with no spacewalk and a day with EVA, blends those days by "
        "how often the crew goes outside, and adds the safety margin on every serving."
    )

    try:
        package = load_food_package()
    except FileNotFoundError as exc:
        st.error(f"A food data file is missing, so nothing can be packed. {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"The food tables could not be loaded. {type(exc).__name__}: {exc}")
        st.stop()

    if package.foods is None or len(package.foods) == 0:
        st.info("The food table loaded, but it has no rows. There is nothing to pack.")
        st.stop()

    entered = _sidebar()
    errors = validate_mission_inputs(
        entered["days"],
        entered["safety_margin"],
        entered["eva_hours_per_week"],
        entered["crew"],
    )
    if errors:
        st.error("Fix these inputs before planning a food load. Nothing was solved.")
        for message in errors:
            st.markdown(f"- {message}")
        st.stop()

    schedule = eva_schedule(entered["eva_hours_per_week"])
    if schedule["note"]:
        st.info(schedule["note"])

    crew_key = tuple(
        (
            float(member["age"]),
            str(member["sex"]),
            float(member["weight_kg"]),
            float(member["height_m"]),
            tuple(member["allergies"]),
        )
        for member in entered["crew"]
    )
    with st.spinner("Planning the food load…"):
        try:
            result = plan_food_load(
                float(entered["days"]),
                crew_key,
                float(entered["safety_margin"]),
                float(schedule["eva_day_fraction"]),
                float(schedule["eva_hours"]),
            )
        except Exception as exc:
            st.error(
                "The planner stopped before it could build a packing list. "
                f"{type(exc).__name__}: {exc}"
            )
            st.stop()

    _render_plan(result, schedule)


@st.cache_resource
def load_food_package():
    """Load the packaged menu once per server process."""

    return load_all()


@st.cache_data(show_spinner=False)
def plan_food_load(
    days: float,
    crew: tuple[tuple[float, str, float, float, tuple[str, ...]], ...],
    safety_margin: float,
    eva_day_fraction: float,
    eva_hours: float,
) -> dict:
    """Call ``mission_food``. The cache key is the full set of inputs."""

    package = load_food_package()
    members = [
        {
            "age": age,
            "sex": sex,
            "weight_kg": weight_kg,
            "height_m": height_m,
            "allergies": list(allergies),
        }
        for age, sex, weight_kg, height_m, allergies in crew
    ]
    return mission_food(
        days,
        members,
        safety_margin=safety_margin,
        eva_day_fraction=eva_day_fraction,
        eva_hours=eva_hours,
        foods=package.foods,
    )


def _sidebar() -> dict:
    st.sidebar.header("Mission")
    st.sidebar.caption(
        "Changing the crew, allergies, or EVA hours solves the menu again. "
        "The first solve for a new crew usually takes several seconds."
    )
    days = st.sidebar.number_input(
        "Mission length (days)",
        min_value=0,
        max_value=1000,
        value=30,
        step=1,
        help="How many days of food to pack. Zero is rejected before the solver runs.",
    )
    eva_hours_per_week = st.sidebar.number_input(
        "EVA hours per week",
        min_value=0.0,
        max_value=80.0,
        value=6.5,
        step=0.5,
        format="%.1f",
        help=(
            "One number for the mission, not per person. "
            "Up to 45.5 hours a week, those hours are spread across standard 6.5-hour EVA days. "
            "Above 45.5, every day is an EVA day."
        ),
    )
    margin_percent = st.sidebar.number_input(
        "Safety margin (%)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        format="%.1f",
        help="Extra servings on top of the solved menu. 10% is a fraction of 0.10.",
    )
    crew_count = st.sidebar.number_input(
        "Number of crew",
        min_value=1,
        max_value=4,
        value=2,
        step=1,
    )
    crew = [_crew_member(index) for index in range(int(crew_count))]
    return {
        "days": int(days),
        "safety_margin": float(margin_percent) / 100.0,
        "eva_hours_per_week": float(eva_hours_per_week),
        "crew": crew,
    }


def _crew_member(index: int) -> dict:
    defaults = DEFAULT_CREW[index]
    sex_key = f"sex-{index}"
    age_key = f"age-{index}"
    sex_now = st.session_state.get(sex_key, defaults["sex"])
    age_now = st.session_state.get(age_key, defaults["age"])
    with st.sidebar.expander(f"Crew member {index + 1} · {sex_now}, {age_now} y", expanded=index < 2):
        age = st.number_input(
            "Age (years)",
            min_value=0,
            max_value=120,
            value=int(defaults["age"]),
            step=1,
            key=age_key,
            help="Targets in this package start at age 19.",
        )
        sex_label = st.selectbox("Sex", ["Male", "Female"], index=0 if defaults["sex"] == "Male" else 1, key=sex_key)
        weight_kg = st.number_input(
            "Weight (kg)",
            min_value=0.0,
            max_value=250.0,
            value=float(defaults["weight_kg"]),
            step=0.1,
            format="%.1f",
            key=f"weight-{index}",
        )
        height_m = st.number_input(
            "Height (m)",
            min_value=0.0,
            max_value=2.50,
            value=float(defaults["height_m"]),
            step=0.01,
            format="%.2f",
            key=f"height-{index}",
        )
        allergy_text = st.text_input(
            "Allergies",
            value="",
            placeholder="almond, shrimp",
            key=f"allergy-{index}",
            help="Comma-separated. A food is dropped when its name contains the word. Entries under 3 letters are ignored.",
        )
    return {
        "age": int(age),
        "sex": str(sex_label).strip().lower(),
        "weight_kg": float(weight_kg),
        "height_m": float(height_m),
        "allergies": _parse_allergies(allergy_text),
    }


def _render_plan(result: Mapping, schedule: Mapping) -> None:
    if not result.get("feasible"):
        st.error(result.get("message") or "This crew and menu cannot meet the nutrient targets.")
        st.caption("No packing list was built, and the nutrient chart is hidden because the menu is not feasible.")
        return

    total = result.get("total_mass_kg")
    if total is None:
        st.error("The planner reported a feasible menu but no total mass. Nothing is shown as packed.")
        return

    st.markdown(
        (
            "<p class='mass-kicker'>Total mission food mass</p>"
            f"<p class='mass-hero'>{float(total):,.2f}<span class='mass-unit'>kg</span></p>"
        ),
        unsafe_allow_html=True,
    )
    st.caption(str(result.get("message") or ""))
    _schedule_caption(result, schedule)

    packing = list(result.get("packing_list") or [])
    st.subheader("Packing list")
    if not packing:
        st.info("The menu solved, but no food had servings left after scaling. There is nothing to download.")
        return

    st.caption(f"{len(packing)} foods, heaviest first. Servings include the safety margin.")
    csv_text = packing_list_csv(packing)
    st.download_button(
        "Download packing list CSV",
        data=csv_text.encode("utf-8"),
        file_name="mission_packing_list.csv",
        mime="text/csv",
    )
    table = pd.DataFrame(
        {
            "Food": [str(row["item"]) for row in packing],
            "Servings": [float(row["servings"]) for row in packing],
            "Mass (kg)": [float(row["mass_kg"]) for row in packing],
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        height=min(560, 48 + 36 * len(table)),
        column_config={
            "Food": st.column_config.TextColumn("Food", width="large"),
            "Servings": st.column_config.NumberColumn("Servings", format="%.2f"),
            "Mass (kg)": st.column_config.NumberColumn("Mass (kg)", format="%.3f"),
        },
    )

    st.subheader("Nutrients vs targets")
    st.caption(
        "Each bar is one mission-average day: typical days and EVA days weighted by how many of each "
        "this flight includes. The length of the bar is delivered amount divided by the minimum target. "
        "The dashed line is 100%. Units sit in the tooltip and the table, because kcal and micrograms "
        "cannot share one axis."
    )
    coverage = mission_average_coverage(result)
    if not coverage:
        st.info("No nutrient had a positive minimum target to chart.")
        return
    _nutrient_chart(coverage)
    clipped = [row for row in coverage if row["clipped"]]
    if clipped:
        names = ", ".join(f"{row['label']} ({row['percent']:.0f}%)" for row in clipped)
        st.caption(
            f"Bars stop at {VISUAL_PERCENT_CAP:.0f}% so one large surplus does not flatten the rest. "
            f"Still above the cap: {names}."
        )
    detail = pd.DataFrame(
        {
            "Nutrient": [row["label"] for row in coverage],
            "Unit": [row["unit"] for row in coverage],
            "Delivered": [row["delivered"] for row in coverage],
            "Minimum target": [row["target_min"] for row in coverage],
            "Percent of minimum": [row["percent"] for row in coverage],
        }
    )
    st.dataframe(
        detail,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Delivered": st.column_config.NumberColumn("Delivered", format="%.2f"),
            "Minimum target": st.column_config.NumberColumn("Minimum target", format="%.2f"),
            "Percent of minimum": st.column_config.NumberColumn("Percent of minimum", format="%.1f"),
        },
    )


def _schedule_caption(result: Mapping, schedule: Mapping) -> None:
    fraction = float(result["eva_day_fraction"])
    hours = float(schedule["eva_hours"])
    if fraction == 0.0:
        eva_text = "No EVA days."
    elif fraction == 1.0:
        eva_text = f"Every day is an EVA day ({hours:.2f} h)."
    else:
        eva_text = f"EVA on {fraction * 100:.1f}% of days, {hours:.2f} h on those days."
    st.caption(
        f"{result['typical_days']:.2f} typical days and {result['eva_days']:.2f} EVA days. {eva_text}"
    )


def _nutrient_chart(coverage: Sequence[Mapping]) -> None:
    frame = pd.DataFrame(coverage)
    bars = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=2)
        .encode(
            x=alt.X(
                "display_percent:Q",
                title="Percent of minimum target",
                scale=alt.Scale(domain=[0, VISUAL_PERCENT_CAP]),
            ),
            y=alt.Y("label:N", sort=[row["label"] for row in coverage], title=None),
            color=alt.condition(
                alt.datum.percent >= 100,
                alt.value("#e2a15a"),
                alt.value("#e07a5f"),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Nutrient"),
                alt.Tooltip("unit:N", title="Unit"),
                alt.Tooltip("delivered:Q", title="Delivered (mission-average day)", format=".2f"),
                alt.Tooltip("target_min:Q", title="Minimum target", format=".2f"),
                alt.Tooltip("percent:Q", title="Percent of minimum", format=".1f"),
            ],
        )
    )
    rule = alt.Chart(pd.DataFrame({"display_percent": [100.0]})).mark_rule(
        color="#d5dde6",
        strokeDash=[4, 4],
    ).encode(x="display_percent:Q")
    chart = (
        (bars + rule)
        .properties(height=max(320, 24 * len(coverage)))
        .configure(background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#e7eef5",
            titleColor="#c5d0dc",
            gridColor="#243044",
            domainColor="#243044",
            labelLimit=180,
        )
    )
    st.altair_chart(chart, use_container_width=True)


def _coverage_row(
    name: str,
    typical: Mapping,
    eva: Mapping,
    typical_days: float,
    eva_days: float,
    days: float,
) -> dict | None:
    weighted_min = 0.0
    weighted_amount = 0.0
    unit = typical.get("unit") or eva.get("unit") or ""
    for day_nutrient, day_count in ((typical, typical_days), (eva, eva_days)):
        if day_count <= 0.0:
            continue
        minimum = day_nutrient.get("minimum")
        if minimum is None or not _is_finite_number(minimum):
            return None
        amount = day_nutrient.get("amount")
        if amount is None or not _is_finite_number(amount):
            amount = 0.0
        weighted_min += float(minimum) * day_count
        weighted_amount += float(amount) * day_count
        if day_nutrient.get("unit"):
            unit = day_nutrient["unit"]
    if weighted_min <= 0.0:
        return None
    target = weighted_min / days
    delivered = weighted_amount / days
    if target <= 0.0:
        return None
    percent = delivered / target * 100.0
    return {
        "nutrient": name,
        "label": nutrient_label(name),
        "unit": display_unit(unit),
        "delivered": delivered,
        "target_min": target,
        "percent": percent,
        "display_percent": min(percent, VISUAL_PERCENT_CAP),
        "clipped": percent > VISUAL_PERCENT_CAP,
    }


def _validate_member(index: int, member: Mapping) -> list[str]:
    who = f"Crew member {index}"
    errors: list[str] = []
    age = _as_float(member.get("age"))
    if age is None:
        errors.append(f"{who} needs a numeric age.")
    elif age < 19:
        errors.append(f"{who} is under 19. Nutrient targets in this package start at age 19.")
    sex = str(member.get("sex", "")).strip().lower()
    if sex not in {"male", "female"}:
        errors.append(f"{who} needs sex set to male or female.")
    weight = _as_float(member.get("weight_kg"))
    if weight is None or weight <= 0.0:
        errors.append(f"{who} needs a weight above 0 kg.")
    height = _as_float(member.get("height_m"))
    if height is None or height <= 0.0:
        errors.append(f"{who} needs a height above 0 m.")
    return errors


def _parse_allergies(text: object) -> list[str]:
    if text is None:
        return []
    parts = [part.strip() for part in str(text).split(",")]
    return [part for part in parts if part]


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _is_finite_number(value: object) -> bool:
    return _as_float(value) is not None


def _require_finite(value: float, label: str) -> float:
    number = _as_float(value)
    if number is None:
        raise ValueError(f"{label} must be a finite number, got {value!r}")
    return number


def _css() -> None:
    st.markdown(
        """
        <style>
          .stApp {
            background-color: #070b12;
            background-image:
              radial-gradient(1px 1px at 7% 14%, rgba(231, 238, 245, 0.75) 50%, transparent 52%),
              radial-gradient(1px 1px at 22% 68%, rgba(231, 238, 245, 0.45) 50%, transparent 52%),
              radial-gradient(1.2px 1.2px at 41% 28%, rgba(226, 161, 90, 0.7) 50%, transparent 52%),
              radial-gradient(1px 1px at 63% 16%, rgba(231, 238, 245, 0.55) 50%, transparent 52%),
              radial-gradient(1px 1px at 78% 74%, rgba(231, 238, 245, 0.4) 50%, transparent 52%),
              radial-gradient(1.4px 1.4px at 91% 22%, rgba(231, 238, 245, 0.65) 50%, transparent 52%),
              radial-gradient(1px 1px at 84% 48%, rgba(226, 161, 90, 0.45) 50%, transparent 52%),
              linear-gradient(180deg, #10192a 0%, #070b12 42%);
          }
          .block-container { padding-top: 1.25rem; padding-bottom: 3rem; max-width: 1180px; }
          section[data-testid="stSidebar"] {
            background: rgba(8, 12, 20, 0.94);
            border-right: 1px solid rgba(226, 161, 90, 0.28);
          }
          .mass-kicker {
            margin: 0.2rem 0 0;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            font-size: 0.78rem;
            color: #e2a15a;
          }
          .mass-hero {
            margin: 0.15rem 0 0.4rem;
            font-size: clamp(2.4rem, 7vw, 4.6rem);
            font-weight: 650;
            letter-spacing: -0.03em;
            line-height: 1.02;
            color: #f3e6d4;
            font-variant-numeric: tabular-nums;
          }
          .mass-unit {
            margin-left: 0.35rem;
            font-size: 0.38em;
            letter-spacing: 0.04em;
            color: #e2a15a;
          }
          [data-testid="stMetric"] {
            background: rgba(226, 161, 90, 0.08);
            border: 1px solid rgba(226, 161, 90, 0.28);
            border-left: 3px solid #e2a15a;
            padding: 0.55rem 0.75rem;
            border-radius: 0.35rem;
          }
          [data-testid="stDataFrame"] { overflow-x: auto; }
          @media (max-width: 480px) {
            .block-container { padding-left: 0.7rem; padding-right: 0.7rem; }
            .mass-hero { font-size: 2.3rem; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
