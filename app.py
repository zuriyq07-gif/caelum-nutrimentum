"""ISS food ledger: menu, data quality, a one-day menu, and shelf-life decay."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from data_loader import load_all
from nutrition import (
    daily_targets,
    decay_curve,
    eer_components,
    fit_decay_items,
    life_stage_group,
    plan_day,
    remaining,
    serving_against_targets,
)

LEDGER_COLUMNS = [
    "item",
    "food_type",
    "food_type_name",
    "mass_g",
    "kcal",
    "protein_g",
    "carb_g",
    "fat_g",
    "sodium_mg",
    "calcium_mg",
    "iron_mg",
    "vit_c_mg_per_serving_est",
    "vit_d_ug_per_serving_est",
    "vit_a_rae_ug_per_serving_est",
    "match_confidence",
    "scaling_basis",
    "quality_flag",
    "quality_reason",
]

STATUS_COLORS = {
    "met": "#7dba8a",
    "under": "#e07a5f",
    "over": "#e2a15a",
    "unknown": "#8aa0b2",
}


def main() -> None:
    st.set_page_config(page_title="ISS food ledger", layout="wide")
    _css()
    st.title("ISS food ledger")
    st.caption(
        "Standard-menu items from the NASA STEMonstrations nutrition guide, with USDA "
        "vitamin estimates on the same serving. Placeholders and rows with no packaged "
        "mass are dropped. Other rows noted in the source stay in the ledger and are flagged."
    )
    try:
        package, fits = _load()
    except FileNotFoundError as exc:
        st.error(f"Data file not found: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Could not load the food package. {type(exc).__name__}: {exc}")
        st.stop()

    profile = _profile(package.energy_rules)
    targets = daily_targets(
        package.requirements,
        package.energy_rules,
        profile["sex"],
        profile["age"],
        profile["kg"],
        profile["eer"].eer_kcal,
    )
    _eer_strip(profile)

    ledger, quality, day, shelf = st.tabs(["Ledger", "Data quality", "Crew day", "Shelf life"])
    with ledger:
        _ledger(package, targets, profile)
    with quality:
        _quality(package)
    with day:
        _day(package, profile, targets)
    with shelf:
        _shelf(package, fits)

    st.caption(
        "Sources: NASA STEMonstrations Nutrition Appendix A, USDA FoodData Central "
        "(SR Legacy and FNDDS), NASEM DRI 2019, NASA-STD-3001 Volume 2 Revision E, "
        "Cooper, Douglas and Perchonok 2017. Field notes are in data/README.md."
    )


@st.cache_resource
def _load():
    package = load_all()
    fits = fit_decay_items(package.decay)
    print(package.summary_text, flush=True)
    return package, fits


def _profile(rules: dict) -> dict:
    activity = rules["eer_equations"]["activity_factor"]
    low, high = activity["allowed_range"]
    with st.expander("Crew profile", expanded=True):
        st.caption(
            "Reference masses quoted with the energy standard: 82.9 kg for the male crew "
            "average and 65.1 kg for the female crew average. Activity factor 1.25 is the nominal active value."
        )
        left, right = st.columns(2)
        sex_label = left.radio("Sex", ["Male", "Female"], horizontal=True)
        age = right.number_input("Age (years)", min_value=19, max_value=90, value=45, step=1)
        sex = "M" if sex_label == "Male" else "F"
        mass_default = 82.9 if sex == "M" else 65.1
        mass_col, height_col = st.columns(2)
        kg = mass_col.number_input(
            "Body mass (kg)",
            min_value=40.0,
            max_value=150.0,
            value=mass_default,
            step=0.1,
            format="%.1f",
            key=f"mass-{sex}",
        )
        height_m = height_col.number_input(
            "Height (m)",
            min_value=1.40,
            max_value=2.20,
            value=1.80,
            step=0.01,
            format="%.2f",
            help="The EER equation takes height in meters.",
        )
        af_col, eva_col = st.columns(2)
        af = af_col.slider("Activity factor", min_value=float(low), max_value=float(high), value=float(activity["nominal"]), step=0.05)
        eva_h = eva_col.slider("EVA hours", min_value=0.0, max_value=8.0, value=0.0, step=0.5)
    return {
        "sex": sex,
        "sex_label": sex_label,
        "age": int(age),
        "kg": float(kg),
        "height_m": float(height_m),
        "af": float(af),
        "eva_h": float(eva_h),
        "eer": eer_components(sex, int(age), float(kg), float(height_m), af=float(af), eva_h=float(eva_h), rules=rules),
        "group": life_stage_group(sex, int(age)),
    }


def _eer_strip(profile: dict) -> None:
    eer = profile["eer"]
    left, right = st.columns(2)
    gap = eer.eer_kcal - eer.provision_kcal
    left.metric(
        "Estimated energy requirement",
        f"{eer.eer_kcal:,.0f} kcal",
        delta=f"{gap:+,.0f} vs {eer.provision_kcal:,.0f} kcal provision",
        delta_color="off",
    )
    right.metric("DRI life stage", profile["group"])
    st.caption(
        f"{eer.formula} → {eer.intercept:g} + ({eer.age_coef:g})×{eer.age:g} "
        f"+ {eer.activity_factor:g}×({eer.mass_coef:g}×{eer.mass_kg:g} + {eer.height_coef:g}×{eer.height_m:g}) "
        f"= {eer.base_kcal:,.0f} kcal, plus {eer.eva_hours:g} EVA h × 200 kcal = {eer.eer_kcal:,.0f} kcal."
    )


def _ledger(package, targets: pd.DataFrame, profile: dict) -> None:
    foods = package.foods
    flagged = int(foods["quality_flag"].sum())
    dropped_names = ", ".join(package.dropped["item"].tolist())
    top_left, top_right = st.columns(2)
    bottom_left, bottom_right = st.columns(2)
    top_left.metric("Menu rows", f"{len(package.menu)}")
    top_right.metric("In the ledger", f"{len(foods)}")
    bottom_left.metric("Dropped", f"{len(package.dropped)}")
    bottom_right.metric("Flagged", f"{flagged}")
    st.caption(f"Dropped: {dropped_names}. Flagged rows remain available and are marked in the table.")

    query = st.text_input("Search items", placeholder="tuna, coffee, tortilla")
    type_names = sorted(foods["food_type_name"].dropna().unique().tolist())
    confidence_levels = [level for level in ["high", "medium", "low", "none"] if level in set(foods["match_confidence"])]
    type_pick = st.pills("Food type", type_names, selection_mode="multi", default=type_names, key="ledger-types")
    confidence_pick = st.pills(
        "USDA match",
        confidence_levels,
        selection_mode="multi",
        default=confidence_levels,
        key="ledger-confidence",
    )
    quality_pick = st.pills(
        "Quality",
        ["Clear", "Flagged"],
        selection_mode="multi",
        default=["Clear", "Flagged"],
        key="ledger-quality",
    )

    view = foods
    if query:
        view = view.loc[view["item"].str.contains(query, case=False, na=False)]
    type_pick = _as_list(type_pick)
    confidence_pick = _as_list(confidence_pick)
    quality_pick = _as_list(quality_pick)
    if type_pick:
        view = view.loc[view["food_type_name"].isin(type_pick)]
    else:
        view = view.iloc[0:0]
    if confidence_pick:
        view = view.loc[view["match_confidence"].isin(confidence_pick)]
    else:
        view = view.iloc[0:0]
    allowed_flags = set()
    if "Clear" in quality_pick:
        allowed_flags.add(False)
    if "Flagged" in quality_pick:
        allowed_flags.add(True)
    view = view.loc[view["quality_flag"].isin(allowed_flags)]
    view = view.reset_index(drop=True)
    st.caption(f"Showing {len(view)} of {len(foods)} ledger rows.")

    if view.empty:
        st.info("No menu items match these filters.")
        return

    show = [column for column in LEDGER_COLUMNS if column in view.columns]
    st.dataframe(
        view[show],
        height=460,
        hide_index=True,
        column_config={
            "kcal": st.column_config.NumberColumn("kcal", format="%.1f"),
            "mass_g": st.column_config.NumberColumn("mass_g", format="%.1f"),
            "quality_flag": st.column_config.CheckboxColumn("flagged"),
            "vit_c_mg_per_serving_est": st.column_config.NumberColumn("vit C mg", format="%.2f"),
            "vit_d_ug_per_serving_est": st.column_config.NumberColumn("vit D µg", format="%.2f"),
            "vit_a_rae_ug_per_serving_est": st.column_config.NumberColumn("vit A µg RAE", format="%.1f"),
        },
    )
    with st.expander("All columns"):
        st.dataframe(view, height=360, hide_index=True)
        st.download_button(
            "Download ledger CSV",
            data=package.foods.to_csv(index=False).encode("utf-8"),
            file_name="iss_foods_merged.csv",
            mime="text/csv",
        )

    choice = st.selectbox("Inspect one serving", view["item"].tolist(), index=None, placeholder="Choose an item")
    if choice is None:
        st.caption(f"Choose an item to compare one package with the {profile['group']} targets. Vitamin D uses the 25 µg flight value.")
        return
    food = view.set_index("item").loc[choice]
    _food_detail(food, targets, profile)


def _food_detail(food: pd.Series, targets: pd.DataFrame, profile: dict) -> None:
    st.subheader(str(food.name))
    st.caption(
        f"{food['food_type']} · {food['food_type_name']} · USDA {food['match_confidence']} · {food['fdc_description']}"
    )
    if bool(food["quality_flag"]):
        st.warning(f"{food['quality_reason']}. {food['data_quality_notes']}")
    if str(food["scaling_basis"]).startswith("not scalable"):
        st.info("This USDA match could not be scaled to a serving, so the vitamin estimates are blank.")
    elif food["match_notes"]:
        st.caption(str(food["match_notes"]))

    mass, kcal, protein, sodium = st.columns(4)
    mass.metric("Packaged mass", _fmt(food["mass_g"], "g"))
    kcal.metric("Energy", _fmt(food["kcal"], "kcal"))
    protein.metric("Protein", _fmt(food["protein_g"], "g"))
    sodium.metric("Sodium", _fmt(food["sodium_mg"], "mg"))
    if pd.notna(food["atwater_kcal"]) and pd.notna(food["kcal"]):
        gap = food["atwater_gap_pct"]
        gap_text = "blank" if pd.isna(gap) else f"{gap:+.0f}%"
        st.caption(
            f"Atwater 4P+4C+9F is {food['atwater_kcal']:.0f} kcal. "
            f"Listed energy differs by {gap_text}. Scaling basis: {food['scaling_basis']}."
        )

    coverage = serving_against_targets(food, targets)
    st.markdown(f"**One package against the {profile['group']} day**")
    st.dataframe(
        coverage,
        hide_index=True,
        column_config={
            "per_serving": st.column_config.NumberColumn("Per serving", format="%.2f"),
            "minimum": st.column_config.NumberColumn("Daily minimum", format="%.1f"),
            "maximum": st.column_config.NumberColumn("Daily maximum", format="%.1f"),
            "percent_of_reference": st.column_config.NumberColumn("% of reference", format="%.1f"),
        },
    )


def _quality(package) -> None:
    st.markdown("**Row counts**")
    st.dataframe(
        package.row_counts,
        hide_index=True,
        column_config={"detail": st.column_config.TextColumn("detail", width="large")},
    )
    with st.expander("Printed summary", expanded=True):
        st.code(package.summary_text, language="text")

    st.markdown("**Missing values**")
    tables = package.row_counts["table"].tolist()
    table = st.selectbox("Table", tables, index=tables.index("foods") if "foods" in tables else 0)
    part = package.missing_values.loc[package.missing_values["table"].eq(table)].copy()
    if table == "macro_guidelines":
        st.caption("Low or high is blank when the guideline bounds only one side. Those cells are left out of the missing-value list.")
        st.dataframe(package.macro_guidelines, hide_index=True)
    elif part.empty:
        st.success(f"{table} has no missing values in the cleaned columns that this report counts.")
    else:
        part["missing_pct_label"] = (part["missing_pct"] * 100).round(1)
        chart = (
            alt.Chart(part)
            .mark_bar(color="#e2a15a")
            .encode(
                x=alt.X("missing:Q", title="Missing cells"),
                y=alt.Y("column:N", sort="-x", title=None),
                tooltip=[
                    alt.Tooltip("column:N", title="Column"),
                    alt.Tooltip("missing:Q", title="Missing"),
                    alt.Tooltip("missing_pct_label:Q", title="Percent of rows", format=".1f"),
                ],
            )
            .properties(height=max(140, 22 * len(part)))
        )
        st.altair_chart(chart)
        st.dataframe(
            part.drop(columns=["missing_pct_label"]),
            hide_index=True,
            column_config={"missing_pct": st.column_config.NumberColumn("missing fraction", format="%.1%")},
        )

    st.markdown("**Dropped**")
    st.dataframe(package.dropped[["item", "quality_reason", "data_quality_notes", "mass_g", "kcal", "sodium_mg", "match_confidence"]], hide_index=True)
    st.markdown("**Flagged**")
    flagged = package.foods.loc[package.foods["quality_flag"], ["item", "quality_reason", "data_quality_notes", "mass_g", "kcal", "iron_mg", "sat_fat_g"]]
    st.dataframe(flagged, hide_index=True)


def _day(package, profile: dict, targets: pd.DataFrame) -> None:
    eer = profile["eer"]
    st.markdown("**Targets for this crewmember**")
    st.caption(
        "Micronutrients use the DRI for the life stage, except vitamin D, which NASA-STD-3001 sets at 25 µg/day. "
        "Carbohydrate and fat bands are the DRI acceptable ranges. Protein uses 0.8 g/kg and a 35% energy ceiling."
    )
    st.dataframe(
        targets.drop(columns=["food_column", "kind"]),
        hide_index=True,
        column_config={
            "target_low": st.column_config.NumberColumn("Minimum", format="%.1f"),
            "target_high": st.column_config.NumberColumn("Maximum", format="%.1f"),
            "note": st.column_config.TextColumn("note", width="large"),
        },
    )
    with st.expander("Macro guidelines from the energy rules"):
        st.dataframe(package.macro_guidelines, hide_index=True)
        st.dataframe(package.energy_parameters, hide_index=True)

    include_flagged = st.checkbox("Include flagged foods", value=False)
    bev_col, pkg_col = st.columns(2)
    max_beverages = bev_col.slider("Maximum beverages", min_value=0, max_value=8, value=4)
    max_packages = pkg_col.slider("Maximum packages of one item", min_value=1, max_value=3, value=1)
    st.caption(
        "The day menu is an integer linear program (SciPy, HiGHS). Each food is 0, 1, or up to the package cap. "
        "It meets numeric targets and does not assign foods to breakfast, lunch, and dinner. "
        "Slacks let the menu miss a target when the pantry cannot cover it. Blank USDA vitamin cells count as zero."
    )

    signature = (
        profile["sex"],
        profile["age"],
        round(profile["kg"], 2),
        round(profile["height_m"], 2),
        round(profile["af"], 2),
        round(profile["eva_h"], 2),
        include_flagged,
        int(max_beverages),
        int(max_packages),
    )
    if st.button("Build day menu", type="primary"):
        with st.spinner("Solving the package menu…"):
            try:
                plan = plan_day(
                    package.foods,
                    package.requirements,
                    package.energy_rules,
                    sex=profile["sex"],
                    age=profile["age"],
                    kg=profile["kg"],
                    height_m=profile["height_m"],
                    af=profile["af"],
                    eva_h=profile["eva_h"],
                    include_flagged=include_flagged,
                    max_packages=int(max_packages),
                    max_beverages=int(max_beverages),
                )
            except Exception as exc:
                plan = None
                st.error(f"The menu solver failed. {type(exc).__name__}: {exc}")
        if plan is not None:
            st.session_state["day_plan"] = plan
            st.session_state["day_sig"] = signature

    plan = st.session_state.get("day_plan")
    if plan is None:
        st.info("Build a day menu to see a package combination for these targets.")
        return
    if st.session_state.get("day_sig") != signature:
        st.info("The crew profile or the menu limits changed. Build the day menu again.")
        return
    if not plan.success:
        st.error(plan.message)
        return

    st.caption(plan.message)
    totals = plan.totals.set_index("nutrient")
    energy = float(totals.loc["Energy", "amount"])
    protein = float(totals.loc["Protein", "amount"])
    sodium = float(totals.loc["Sodium", "amount"])
    vitamin_d = float(totals.loc["Vitamin D", "amount"])
    row1_left, row1_right = st.columns(2)
    row2_left, row2_right = st.columns(2)
    row1_left.metric("Packages", f"{plan.packages_chosen}")
    row1_right.metric(
        "Energy",
        f"{energy:,.0f} kcal",
        delta=f"{energy - eer.eer_kcal:+,.0f} vs EER",
        delta_color="off",
    )
    row2_left.metric(
        "Protein",
        f"{protein:.0f} g",
        delta=f"{protein - float(totals.loc['Protein', 'target_low']):+.0f} g vs 0.8 g/kg",
        delta_color="off",
    )
    row2_right.metric(
        "Sodium",
        f"{sodium:,.0f} mg",
        delta=f"{sodium - 2300:+,.0f} mg vs 2,300 mg cap",
        delta_color="off",
    )
    st.caption(f"Vitamin D from these foods: {vitamin_d:.1f} µg of the 25 µg flight target.")
    if totals.loc["Vitamin D", "status"] == "under":
        st.info("USDA matches for this menu do not cover 25 µg of vitamin D. The flight food system also carries vitamin D tablets.")
    if totals.loc["Sodium", "status"] == "over":
        st.info("Hitting the energy target with this packaged menu goes past 2,300 mg sodium. Observed ISS intakes were higher still (about 3,823 ± 785 mg).")

    chart_rows = plan.totals.dropna(subset=["target_low"]).copy()
    chart_rows = chart_rows.loc[chart_rows["target_low"] > 0]
    chart_rows["percent"] = chart_rows["amount"] / chart_rows["target_low"] * 100
    bars = (
        alt.Chart(chart_rows)
        .mark_bar()
        .encode(
            x=alt.X("percent:Q", title="Percent of minimum"),
            y=alt.Y("nutrient:N", sort=chart_rows["nutrient"].tolist(), title=None),
            color=alt.Color(
                "status:N",
                scale=alt.Scale(
                    domain=list(STATUS_COLORS),
                    range=list(STATUS_COLORS.values()),
                ),
                legend=alt.Legend(title="Status"),
            ),
            tooltip=["nutrient", alt.Tooltip("percent:Q", format=".0f"), "status", alt.Tooltip("amount:Q", format=".1f")],
        )
        .properties(height=max(280, 18 * len(chart_rows)))
    )
    rule = alt.Chart(pd.DataFrame({"percent": [100]})).mark_rule(color="#d5dde6", strokeDash=[4, 4]).encode(x="percent:Q")
    st.altair_chart(bars + rule)

    st.markdown("**Packages**")
    st.dataframe(
        plan.servings,
        hide_index=True,
        column_config={
            "packages": st.column_config.NumberColumn("packages", format="%d"),
            "kcal": st.column_config.NumberColumn("kcal", format="%.0f"),
            "protein_g": st.column_config.NumberColumn("protein g", format="%.1f"),
            "sodium_mg": st.column_config.NumberColumn("sodium mg", format="%.0f"),
        },
    )
    st.markdown("**Day vs targets**")
    st.dataframe(
        plan.totals.drop(columns=["note"]),
        hide_index=True,
        column_config={
            "amount": st.column_config.NumberColumn("Amount", format="%.1f"),
            "target_low": st.column_config.NumberColumn("Minimum", format="%.1f"),
            "target_high": st.column_config.NumberColumn("Maximum", format="%.1f"),
            "share_of_energy_pct": st.column_config.NumberColumn("% of energy", format="%.1f"),
        },
    )
    st.download_button(
        "Download day menu CSV",
        data=plan.servings.to_csv(index=False).encode("utf-8"),
        file_name="iss_day_menu.csv",
        mime="text/csv",
    )


def _shelf(package, fits: pd.DataFrame) -> None:
    st.markdown("**Shelf life by food type**")
    st.caption("Bounds are at ambient storage. Intermediate-moisture and fresh-food rows are gaps in the fetched sources. Rates below are for 21 °C; this package has no temperature model.")
    shelf_rows = package.decay.loc[package.decay["record_type"].eq("shelf_life")].copy()
    columns = [
        "food_type_code",
        "category",
        "shelf_life_years_max",
        "shelf_life_bound",
        "value_status",
        "notes",
    ]
    st.dataframe(shelf_rows[[column for column in columns if column in shelf_rows.columns]], hide_index=True)

    st.markdown("**Fallback decay rate**")
    st.caption(
        "Pick the most specific category for a vitamin. The amount at the selected year uses "
        "C(t) = C0 exp(−kt). The line is that same model integrated with SciPy (dC/dt = −kC)."
    )
    fallback = package.decay.loc[package.decay["record_type"].eq("fallback_default")].copy()
    vitamins = fallback["vitamin"].dropna().drop_duplicates().tolist()
    vitamin = st.selectbox("Vitamin", vitamins)
    categories = fallback.loc[fallback["vitamin"].eq(vitamin), "category"].dropna().astype(str).drop_duplicates().tolist()
    category = st.selectbox("Category", categories, key=f"decay-category-{vitamin}")
    year_col, amount_col = st.columns(2)
    years = year_col.slider("Years of storage", min_value=0.0, max_value=3.0, value=1.5, step=0.1)
    c0 = amount_col.number_input("Starting amount", min_value=0.0, value=100.0, step=5.0)
    estimate = remaining(c0, vitamin, years, category, package.decay)

    if estimate.amount is None:
        st.warning(estimate.note or "No numeric rate for this vitamin.")
    else:
        left, right = st.columns(2)
        left.metric("Remaining", f"{estimate.amount:.1f}", delta=f"{estimate.fraction_remaining * 100:.1f}% of start", delta_color="off")
        if estimate.half_life_years is None:
            right.metric("Half-life", "Not defined")
            if estimate.k_per_year == 0:
                st.caption("The fallback rate is zero, so the curve stays flat. For riboflavin and vitamin E that zero is a placeholder where the papers did not publish a number.")
        else:
            right.metric("Half-life", f"{estimate.half_life_years:.2f} years")
        st.caption(
            f"k = {estimate.k_per_year:.4f} per year · category “{estimate.category}” · {estimate.value_status or 'derived default'}"
        )
        if estimate.note:
            st.caption(estimate.note)
        times, curve = decay_curve(estimate.c0, estimate.k_per_year, 3.0)
        curve_frame = pd.DataFrame({"years": times, "amount": curve})
        point = pd.DataFrame({"years": [estimate.t_years], "amount": [estimate.amount]})
        line = (
            alt.Chart(curve_frame)
            .mark_line(color="#e2a15a")
            .encode(
                x=alt.X("years:Q", title="Years at 21 °C"),
                y=alt.Y("amount:Q", title="Amount"),
            )
        )
        dot = alt.Chart(point).mark_point(filled=True, size=90, color="#f4f7fb").encode(x="years:Q", y="amount:Q")
        st.altair_chart(line + dot, height=280)

    st.markdown("**Measured items, published k vs SciPy**")
    st.caption(
        "Published k uses only the start and the 3-year point. "
        "The curve fit uses the 0-, 1-, and 3-year points with the starting amount fixed. "
        "The log-linear fit lets the intercept move."
    )
    if fits.empty:
        st.info("No decay items had three concentrations to fit.")
        return
    plot_fits = fits.dropna(subset=["k_published", "k_curve_fit"]).copy()
    if not plot_fits.empty:
        upper = float(max(plot_fits["k_published"].max(), plot_fits["k_curve_fit"].max()))
        lower = float(min(plot_fits["k_published"].min(), plot_fits["k_curve_fit"].min(), 0))
        guide = pd.DataFrame({"k_published": [lower, upper], "k_curve_fit": [lower, upper]})
        points = (
            alt.Chart(plot_fits)
            .mark_circle(size=70, color="#7eb8c9")
            .encode(
                x=alt.X("k_published:Q", title="Published k (per year)"),
                y=alt.Y("k_curve_fit:Q", title="SciPy curve-fit k (per year)"),
                tooltip=["item_or_group", "vitamin", alt.Tooltip("k_published:Q", format=".3f"), alt.Tooltip("k_curve_fit:Q", format=".3f"), alt.Tooltip("rmse:Q", format=".2f")],
            )
        )
        identity = (
            alt.Chart(guide)
            .mark_line(strokeDash=[4, 4], color="#8aa0b2")
            .encode(x="k_published:Q", y="k_curve_fit:Q")
        )
        st.altair_chart(points + identity, height=320)
    st.dataframe(
        fits,
        hide_index=True,
        column_config={
            "k_published": st.column_config.NumberColumn("k published", format="%.3f"),
            "k_curve_fit": st.column_config.NumberColumn("k curve fit", format="%.3f"),
            "k_loglinear": st.column_config.NumberColumn("k log-linear", format="%.3f"),
            "rmse": st.column_config.NumberColumn("RMSE", format="%.2f"),
        },
    )

    labeled = fits.copy()
    labeled["label"] = labeled["item_or_group"].astype(str) + " · " + labeled["vitamin"].astype(str)
    if labeled["label"].duplicated().any():
        labeled["label"] = labeled["label"] + " · " + labeled.index.astype(str)
    label = st.selectbox("Compare one measured item", labeled["label"].tolist())
    chosen = labeled.loc[labeled["label"].eq(label)].iloc[0]
    measured = pd.DataFrame(
        {
            "years": [0, 1, 3],
            "amount": [chosen["c_initial"], chosen["c_1yr"], chosen["c_3yr"]],
        }
    )
    curve_rows = []
    if pd.notna(chosen["k_published"]):
        times, curve = decay_curve(chosen["c_initial"], chosen["k_published"], 3)
        curve_rows.append(pd.DataFrame({"years": times, "amount": curve, "series": "Published k"}))
    if pd.notna(chosen["k_curve_fit"]):
        times, curve = decay_curve(chosen["c_initial"], chosen["k_curve_fit"], 3)
        curve_rows.append(pd.DataFrame({"years": times, "amount": curve, "series": "SciPy curve fit"}))
    points = alt.Chart(measured).mark_point(filled=True, size=80, color="#f4f7fb").encode(
        x=alt.X("years:Q", title="Years at 21 °C"),
        y=alt.Y("amount:Q", title="Amount"),
    )
    if curve_rows:
        lines = (
            alt.Chart(pd.concat(curve_rows, ignore_index=True))
            .mark_line()
            .encode(
                x=alt.X("years:Q", title="Years at 21 °C"),
                y=alt.Y("amount:Q", title="Amount"),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(domain=["Published k", "SciPy curve fit"], range=["#e2a15a", "#7eb8c9"]),
                    legend=alt.Legend(title=None),
                ),
            )
        )
        st.altair_chart(lines + points, height=280)
    else:
        st.altair_chart(points, height=280)
        st.caption("This item has measurements but no rate to draw.")


def _fmt(value, unit: str) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):,.1f} {unit}"


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _css() -> None:
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.4rem; max-width: 1180px; }
          [data-testid="stMetric"] {
            background: rgba(226, 161, 90, 0.08);
            border: 1px solid rgba(226, 161, 90, 0.28);
            border-left: 3px solid #e2a15a;
            padding: 0.55rem 0.75rem;
            border-radius: 0.35rem;
          }
          [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
