"""ISS mission food manifest: pack the standard menu for a crew.

The page collects mission inputs and calls :func:`mission.mission_food`.
Nutrient targets and the menu linear program stay in ``targets`` and ``optimizer``.
EVA hours per week are converted here and always passed as ``eva_day_fraction``.
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

# First row is the first-load crew. Later rows fill extra seats the user adds.
DEFAULT_CREW = (
    {"age": 45, "sex": "Male", "weight_kg": 82.9, "height_cm": 180},
    {"age": 38, "sex": "Female", "weight_kg": 62.0, "height_cm": 168},
    {"age": 41, "sex": "Male", "weight_kg": 77.0, "height_cm": 178},
    {"age": 34, "sex": "Female", "weight_kg": 58.0, "height_cm": 165},
    {"age": 50, "sex": "Male", "weight_kg": 85.0, "height_cm": 182},
    {"age": 29, "sex": "Female", "weight_kg": 60.0, "height_cm": 170},
)

_NUTRIENT_ORDER = {name: index for index, name in enumerate(FOOD_COLUMNS)}


def eva_schedule(weekly_hours: float) -> dict:
    """Map mission EVA hours per week to ``mission_food`` arguments.

    Nominal EVA-day length is 6.5 hours.

    * 0 hours: ``eva_day_fraction`` 0 and ``eva_hours`` 0. No EVA days.
    * More than 0 and at most 6.5: one EVA day per week (fraction 1/7)
      lasting ``weekly_hours`` (a shorter day when the week is under 6.5).
    * Above 6.5: ``n = weekly_hours / 6.5``. When ``n`` is at most 7, the
      fraction is ``n/7`` and each EVA day is 6.5 hours. When ``n`` is above
      7, every day is an EVA day and ``eva_hours`` is the weekly total / 7.

    The fraction is clipped to ``[0, 1]``. ``note`` is set only when the
    week is under one standard day, or when every day has to be an EVA day.
    ``summary`` is the sentence shown next to the inputs.

    Raises:
        ValueError: ``weekly_hours`` is negative or not a finite number.
    """

    hours = _require_finite(weekly_hours, "EVA hours per week")
    if hours < 0.0:
        raise ValueError(f"EVA hours per week cannot be negative, got {weekly_hours!r}")
    if hours == 0.0:
        fraction, eva_hours, note = 0.0, 0.0, None
        summary = "0 h/week maps to eva_day_fraction 0 and eva_hours 0. No EVA days."
    elif hours <= STANDARD_EVA_HOURS:
        fraction = _clip_unit(1.0 / DAYS_PER_WEEK)
        eva_hours = hours
        if hours < STANDARD_EVA_HOURS:
            note = (
                f"Weekly EVA time is under {STANDARD_EVA_HOURS:g} hours, so the week has "
                f"one EVA day of {hours:.2f} hours."
            )
        else:
            note = None
        summary = (
            f"{hours:g} h/week maps to one EVA day per week: "
            f"eva_day_fraction {fraction:.6g}, eva_hours {eva_hours:g}."
        )
    else:
        count = hours / STANDARD_EVA_HOURS
        if count <= DAYS_PER_WEEK:
            fraction = _clip_unit(count / DAYS_PER_WEEK)
            eva_hours = STANDARD_EVA_HOURS
            note = None
            summary = (
                f"{hours:g} h/week maps to {count:.2f} EVA days of {STANDARD_EVA_HOURS:g} h: "
                f"eva_day_fraction {fraction:.6g}, eva_hours {eva_hours:g}."
            )
        else:
            fraction = 1.0
            eva_hours = hours / DAYS_PER_WEEK
            note = (
                f"Weekly EVA time is above {MAX_STANDARD_WEEKLY_EVA:g} hours, so every day "
                f"is treated as an EVA day of {eva_hours:.2f} hours."
            )
            summary = (
                f"{hours:g} h/week maps to an EVA every day: "
                f"eva_day_fraction {fraction:.6g}, eva_hours {eva_hours:.2f}."
            )
    return {
        "eva_day_fraction": fraction,
        "eva_hours": eva_hours,
        "note": note,
        "summary": summary,
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
    """CSV text with columns item, servings, mass_kg and a labeled total row.

    An empty list is the header only, so an infeasible mission does not
    pretend the total mass is zero.
    """

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["item", "servings", "mass_kg"])
    total = 0.0
    for row in packing_list:
        servings = float(row["servings"])
        mass_kg = float(row["mass_kg"])
        writer.writerow([str(row["item"]), f"{servings:.2f}", f"{mass_kg:.3f}"])
        total += mass_kg
    if packing_list:
        writer.writerow(["Total", "", f"{total:.3f}"])
    return buffer.getvalue()


def mission_average_coverage(result: Mapping) -> list[dict]:
    """Mission-weighted delivered nutrients as a percent of the minimum target.

    For each nutrient constrained by the optimizer (``nutrients`` on each day,
    with ``amount``, ``minimum``, and ``unit``)::

        delivered = (typical_amount * typical_days + eva_amount * eva_days) / days
        target = (typical_minimum * typical_days + eva_minimum * eva_days) / days
        percent = delivered / target * 100

    A day with zero length does not have to carry a minimum. A day that does
    count, and is missing its minimum, drops the nutrient. A weighted minimum
    of 0 is skipped. ``display_percent`` is capped for the axis; ``percent``
    stays uncapped and ``clipped`` records the cap. ``days`` of 0 returns an
    empty list.
    """

    days = _as_float(result.get("days"))
    if days is None or days <= 0.0:
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
    st.set_page_config(page_title="Mission food manifest", layout="wide")
    _css()
    st.title("Mission food manifest")
    st.caption(
        "Food mass for this flight, packed from the ISS standard menu. The planner solves a "
        "minimum-mass day inside the station and a day with a spacewalk, weights those days by "
        "the EVA schedule, and adds the safety margin to every serving."
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
        st.error("Fix these inputs before packing a menu. Nothing was solved.")
        for message in errors:
            st.markdown(f"- {message}")
        st.stop()

    schedule = entered["schedule"]
    if schedule["note"]:
        st.info(schedule["note"])

    crew_key = tuple(
        (
            int(member["age"]),
            str(member["sex"]),
            float(member["weight_kg"]),
            float(member["height_m"]),
            tuple(member["allergies"]),
        )
        for member in entered["crew"]
    )
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

    _render_plan(result, schedule, len(entered["crew"]))


@st.cache_resource
def load_food_package():
    """Load the packaged menu once per server process."""

    return load_all()


@st.cache_data(show_spinner="Packing the menu…")
def plan_food_load(
    days: float,
    crew: tuple[tuple[int, str, float, float, tuple[str, ...]], ...],
    safety_margin: float,
    eva_day_fraction: float,
    eva_hours: float,
) -> dict:
    """Call ``mission_food``. The cache key is the full set of inputs.

    ``eva_day_fraction`` is required. Leaving it out would make
    ``mission_food`` read ``data/launch_library/spacewalks.json``, which is
    not in this repo.
    """

    package = load_food_package()
    members = [
        {
            "age": int(age),
            "sex": sex,
            "weight_kg": float(weight_kg),
            "height_m": float(height_m),
            "allergies": list(allergies),
        }
        for age, sex, weight_kg, height_m, allergies in crew
    ]
    tokens: list[str] = []
    seen: set[str] = set()
    for member in members:
        for token in member["allergies"]:
            if token not in seen:
                seen.add(token)
                tokens.append(token)
    return mission_food(
        days,
        members,
        safety_margin=safety_margin,
        eva_day_fraction=eva_day_fraction,
        eva_hours=eva_hours,
        foods=package.foods,
        allergies=tokens,
    )


def _sidebar() -> dict:
    st.sidebar.header("Mission")
    st.sidebar.caption(
        "Changing the crew, allergies, or EVA hours solves the menu again. "
        "The first solve for a new crew usually takes several seconds. "
        "A cached menu is reused when the inputs match."
    )
    days = st.sidebar.number_input(
        "Mission length (days)",
        min_value=1,
        max_value=1000,
        value=30,
        step=1,
        help="How many days of food to pack. At least 1.",
    )
    crew_count = st.sidebar.number_input(
        "Number of crew",
        min_value=1,
        max_value=6,
        value=1,
        step=1,
    )
    crew = [_crew_member(index) for index in range(int(crew_count))]
    eva_hours_per_week = st.sidebar.number_input(
        "EVA hours per week",
        min_value=0.0,
        max_value=80.0,
        value=6.5,
        step=0.5,
        format="%.1f",
        help=(
            "One number for the mission, not per person. Nominal EVA-day length is 6.5 h. "
            "0 leaves EVA days out. Up to 6.5 h is one EVA day that week. "
            "Above 6.5 h, hours are packed into 6.5 h EVA days. "
            "Above 45.5 h, every day is an EVA day."
        ),
    )
    schedule = eva_schedule(float(eva_hours_per_week))
    st.sidebar.caption(schedule["summary"])
    st.sidebar.caption(
        f"Passed to mission_food: eva_day_fraction {schedule['eva_day_fraction']:.6g}, "
        f"eva_hours {schedule['eva_hours']:.2f}. Nominal EVA-day length is {STANDARD_EVA_HOURS:g} h."
    )
    margin_percent = st.sidebar.number_input(
        "Safety margin (%)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        format="%.0f",
        help="Extra servings on top of the solved menu. 10% is a fraction of 0.10.",
    )
    return {
        "days": int(days),
        "safety_margin": float(margin_percent) / 100.0,
        "eva_hours_per_week": float(eva_hours_per_week),
        "crew": crew,
        "schedule": schedule,
    }


def _crew_member(index: int) -> dict:
    defaults = DEFAULT_CREW[index]
    sex_key = f"sex-{index}"
    age_key = f"age-{index}"
    sex_now = st.session_state.get(sex_key, defaults["sex"])
    age_now = st.session_state.get(age_key, defaults["age"])
    with st.sidebar.expander(f"Crew member {index + 1} · {sex_now}, {age_now} y", expanded=index == 0):
        age = st.number_input(
            "Age (years)",
            min_value=19,
            max_value=100,
            value=int(defaults["age"]),
            step=1,
            key=age_key,
            help="Targets in this package start at age 19.",
        )
        sex_label = st.selectbox(
            "Sex",
            ["Male", "Female"],
            index=0 if defaults["sex"] == "Male" else 1,
            key=sex_key,
        )
        weight_kg = st.number_input(
            "Weight (kg)",
            min_value=0.0,
            max_value=250.0,
            value=float(defaults["weight_kg"]),
            step=0.1,
            format="%.1f",
            key=f"weight-{index}",
        )
        height_cm = st.number_input(
            "Height (cm)",
            min_value=0,
            max_value=250,
            value=int(defaults["height_cm"]),
            step=1,
            key=f"height-{index}",
            help="Centimeters. Divided by 100 before the energy equation, which uses meters.",
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
        "height_m": float(height_cm) / 100.0,
        "allergies": _parse_allergies(allergy_text),
    }


def _render_plan(result: Mapping, schedule: Mapping, crew_count: int) -> None:
    if not result.get("feasible"):
        _render_infeasible(result, schedule, crew_count)
        return

    total = result.get("total_mass_kg")
    if total is None or not _is_finite_number(total):
        st.error("The planner reported a feasible menu but no total mass. Nothing is shown as packed.")
        return

    st.markdown(
        (
            "<p class='mass-kicker'>Total mission food mass</p>"
            f"<p class='mass-hero'>{float(total):,.1f}<span class='mass-unit'>kg</span></p>"
        ),
        unsafe_allow_html=True,
    )
    _mission_lines(result, schedule, crew_count)

    packing = list(result.get("packing_list") or [])
    st.subheader("Packing list")
    if not packing:
        st.info("The menu solved, but no food had servings left after scaling. There is nothing to pack.")
        _packing_table([])
        _download_packing([])
    else:
        st.caption(f"{len(packing)} foods, heaviest first. Servings include the safety margin.")
        _packing_table(packing)
        _download_packing(packing)

    _render_chart(result)


def _render_infeasible(result: Mapping, schedule: Mapping, crew_count: int) -> None:
    st.error(str(result.get("message") or "This crew and menu cannot meet the nutrient targets."))
    unmet_rows = _unmet_by_day(result)
    if unmet_rows:
        for label, unmet in unmet_rows:
            if unmet:
                names = ", ".join(nutrient_label(name) for name in unmet)
                st.markdown(f"**{label} unmet nutrients:** {names}")
            else:
                st.markdown(f"**{label}:** nutrient targets cannot be met.")
    else:
        st.markdown("No unmet-nutrient list was returned with this infeasible menu.")
    _mission_lines(result, schedule, crew_count)
    st.info("No food mass was packed. The nutrient chart is omitted because the menu is not feasible.")
    st.subheader("Packing list")
    _packing_table([])
    st.caption("No items. The download has the header row only.")
    _download_packing([])


def _mission_lines(result: Mapping, schedule: Mapping, crew_count: int) -> None:
    days = _as_float(result.get("days"))
    day_text = f"{days:g}" if days is not None else "—"
    margin = _as_float(result.get("safety_margin"))
    margin_text = _format_percent(margin) if margin is not None else "—"
    fraction = _as_float(result.get("eva_day_fraction"))
    fraction_text = f"{fraction:.6g}" if fraction is not None else "—"
    st.caption(f"{day_text} days · {crew_count} crew · {margin_text} margin · EVA-day fraction {fraction_text}")
    summary = schedule.get("summary")
    if summary:
        st.caption(str(summary))


def _packing_table(packing: Sequence[Mapping]) -> None:
    table = pd.DataFrame(
        {
            "item": [str(row["item"]) for row in packing],
            "servings": [float(row["servings"]) for row in packing],
            "mass_kg": [float(row["mass_kg"]) for row in packing],
        }
    )
    height = 120 if table.empty else min(480, 38 + 36 * len(table))
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=height,
        column_config={
            "item": st.column_config.TextColumn("item", width="large"),
            "servings": st.column_config.NumberColumn("servings", format="%.2f"),
            "mass_kg": st.column_config.NumberColumn("mass_kg", format="%.3f"),
        },
    )


def _download_packing(packing: Sequence[Mapping]) -> None:
    csv_text = packing_list_csv(packing)
    st.download_button(
        "Download packing list CSV",
        data=csv_text.encode("utf-8"),
        file_name="mission_packing_list.csv",
        mime="text/csv",
        key="packing-csv",
    )


def _render_chart(result: Mapping) -> None:
    st.subheader("Share of daily minimum, mission-weighted")
    st.caption(
        "Each bar is one mission-weighted day: typical-day totals and EVA-day totals combined by "
        "how many of each this flight includes, then divided by the weighted minimum. "
        "The dashed line is 100%. Amounts and units are in the tooltip, because kcal, grams, "
        "milligrams, and micrograms cannot share one axis. Bars may pass 100%."
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


def _unmet_by_day(result: Mapping) -> list[tuple[str, list[str]]]:
    rows: list[tuple[str, list[str]]] = []
    for label, key in (("Typical day", "typical_day"), ("EVA day", "eva_day")):
        day = result.get(key) or {}
        if day.get("feasible"):
            continue
        unmet = [str(name) for name in (day.get("unmet") or [])]
        rows.append((label, unmet))
    return rows


def _nutrient_chart(coverage: Sequence[Mapping]) -> None:
    frame = pd.DataFrame(coverage)
    bars = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=2)
        .encode(
            x=alt.X(
                "display_percent:Q",
                title="Share of daily minimum, mission-weighted",
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
                alt.Tooltip("delivered:Q", title="Achieved (mission-weighted day)", format=".2f"),
                alt.Tooltip("target_min:Q", title="Target minimum", format=".2f"),
                alt.Tooltip("percent:Q", title="Share of daily minimum (%)", format=".1f"),
            ],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"display_percent": [100.0]}))
        .mark_rule(color="#c9d3de", strokeDash=[4, 4])
        .encode(x="display_percent:Q")
    )
    chart = (
        (bars + rule)
        .properties(height=max(320, 24 * len(coverage)))
        .configure(background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#c9d3de",
            titleColor="#c9d3de",
            gridColor="#1c2836",
            domainColor="#1c2836",
            labelLimit=180,
            labelFont="IBM Plex Sans, Source Sans 3, sans-serif",
            titleFont="IBM Plex Sans, Source Sans 3, sans-serif",
        )
    )
    st.altair_chart(chart, width="stretch", theme=None)


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


def _format_percent(fraction: float) -> str:
    percent = fraction * 100.0
    if abs(percent - round(percent)) < 1e-6:
        return f"{percent:.0f}%"
    return f"{percent:.1f}%"


def _clip_unit(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


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
          html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            font-family: "IBM Plex Sans", "Source Sans 3", ui-sans-serif, system-ui, sans-serif;
          }
          .stApp {
            background-color: #070b12;
            color: #c9d3de;
            background-image:
              radial-gradient(1px 1px at 8% 14%, rgba(201, 211, 222, 0.75) 50%, transparent 52%),
              radial-gradient(1px 1px at 16% 58%, rgba(201, 211, 222, 0.35) 50%, transparent 52%),
              radial-gradient(1.2px 1.2px at 28% 24%, rgba(226, 161, 90, 0.55) 50%, transparent 52%),
              radial-gradient(1px 1px at 44% 72%, rgba(201, 211, 222, 0.4) 50%, transparent 52%),
              radial-gradient(1px 1px at 57% 18%, rgba(201, 211, 222, 0.55) 50%, transparent 52%),
              radial-gradient(1px 1px at 71% 46%, rgba(226, 161, 90, 0.32) 50%, transparent 52%),
              radial-gradient(1.3px 1.3px at 84% 16%, rgba(201, 211, 222, 0.6) 50%, transparent 52%),
              radial-gradient(1px 1px at 93% 68%, rgba(201, 211, 222, 0.3) 50%, transparent 52%),
              radial-gradient(ellipse 140% 46% at 50% 122%, transparent 70%, rgba(226, 161, 90, 0.16) 70.55%, transparent 71.15%),
              radial-gradient(ellipse 95% 34% at 58% -6%, transparent 67%, rgba(176, 196, 214, 0.12) 67.45%, transparent 68.05%),
              linear-gradient(180deg, #0c1422 0%, #070b12 40%, #05070c 100%);
            font-family: "IBM Plex Sans", "Source Sans 3", ui-sans-serif, system-ui, sans-serif;
          }
          .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label, .stApp li,
          [data-testid="stCaptionContainer"], [data-testid="stMarkdownContainer"] {
            font-family: "IBM Plex Sans", "Source Sans 3", ui-sans-serif, system-ui, sans-serif;
          }
          .stApp h1, .stApp h2, .stApp h3 { color: #d5dee8; letter-spacing: -0.02em; }
          .block-container { padding-top: 1.25rem; padding-bottom: 3rem; max-width: 1240px; }
          section[data-testid="stSidebar"] {
            background: rgba(8, 12, 20, 0.96);
            border-right: 1px solid rgba(226, 161, 90, 0.28);
          }
          section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: #c9d3de;
          }
          .mass-kicker {
            margin: 0.2rem 0 0;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            font-size: 0.78rem;
            color: #e2a15a;
          }
          .mass-hero {
            margin: 0.15rem 0 0.35rem;
            padding-bottom: 0.35rem;
            border-bottom: 1px solid rgba(226, 161, 90, 0.35);
            font-size: clamp(2.4rem, 7vw, 4.6rem);
            font-weight: 650;
            letter-spacing: -0.03em;
            line-height: 1.02;
            color: #e8b56a;
            font-variant-numeric: tabular-nums;
          }
          .mass-unit {
            margin-left: 0.35rem;
            font-size: 0.38em;
            letter-spacing: 0.04em;
            color: #e2a15a;
          }
          [data-testid="stDataFrame"] { overflow-x: auto; max-width: 100%; }
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
