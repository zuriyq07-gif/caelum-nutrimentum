"""Shelf-life decay, epoch boundaries, and replanned vitamin totals."""

import math

import pandas as pd
import pytest

from decay import (
    DAYS_PER_YEAR,
    concentration,
    decayed_foods,
    describe_shortfalls,
    epoch_starts,
    match_decay_rate,
    plan_deltas,
    project_daily,
    replan_mission,
)
from nutrition import remaining

CREW = {"age": 45, "sex": "M", "weight_kg": 82.9, "height_m": 1.80}


def _fruit_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "record_type": "fallback_default",
                "category": "fruit (any type)",
                "vitamin": "vitamin_c",
                "item_or_group": "DEFAULT",
                "k_per_year": 0.3005,
            },
            {
                "record_type": "fallback_default",
                "category": "fortified beverage powder",
                "vitamin": "vitamin_c",
                "item_or_group": "DEFAULT",
                "k_per_year": 0.0,
            },
        ]
    )


def _foods() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item": "Peaches",
                "food_type": "T",
                "mass_g": 80.0,
                "kcal": 50.0,
                "protein_g": 1.0,
                "iron_mg": 4.0,
                "vit_c_mg_per_serving_est": 100.0,
            },
            {
                "item": "Tea",
                "food_type": "B",
                "mass_g": 10.0,
                "kcal": 2.0,
                "protein_g": 0.0,
                "iron_mg": 0.1,
                "vit_c_mg_per_serving_est": 100.0,
            },
        ]
    )


def test_concentration_uses_year_fraction_not_raw_days():
    k = 0.3005
    assert concentration(100, k, 0) == pytest.approx(100, rel=1e-6)
    assert concentration(100, k, DAYS_PER_YEAR) == pytest.approx(100 * math.exp(-k), rel=1e-6)
    k_per_day = k / DAYS_PER_YEAR
    assert concentration(100, k, 10) == pytest.approx(100 * math.exp(-k_per_day * 10), rel=1e-6)
    # A raw day count in the exponent would be exp(-0.3005 * 365.25), not exp(-0.3005).
    assert concentration(100, k, DAYS_PER_YEAR) != pytest.approx(100 * math.exp(-k * DAYS_PER_YEAR), rel=1e-3)


def test_concentration_matches_nutrition_remaining_on_one_year(package):
    estimate = remaining(100, "vitamin_c", 1.0, "fruit", package.decay)
    assert estimate.k_per_year == pytest.approx(0.3005)
    assert concentration(100, estimate.k_per_year, DAYS_PER_YEAR) == pytest.approx(estimate.amount, rel=1e-6)


def test_decayed_foods_scales_fruit_and_leaves_a_zero_rate_type_unchanged():
    foods = _foods()
    original = foods.copy()
    day0 = decayed_foods(foods, 0, _fruit_table())
    pd.testing.assert_frame_equal(day0, foods)
    pd.testing.assert_frame_equal(foods, original)

    out = decayed_foods(foods, DAYS_PER_YEAR, _fruit_table()).set_index("item")
    assert out.loc["Peaches", "vit_c_mg_per_serving_est"] == pytest.approx(100 * math.exp(-0.3005), rel=1e-6)
    assert out.loc["Tea", "vit_c_mg_per_serving_est"] == pytest.approx(100, rel=1e-6)
    assert out.loc["Peaches", "kcal"] == 50
    assert out.loc["Peaches", "mass_g"] == 80
    assert out.loc["Peaches", "protein_g"] == 1
    assert out.loc["Peaches", "iron_mg"] == 4
    pd.testing.assert_frame_equal(foods, original)


def test_item_rate_beats_fruit_fallback_and_negative_k_is_clipped():
    table = pd.concat(
        [
            _fruit_table(),
            pd.DataFrame(
                [
                    {
                        "record_type": "decay_item",
                        "category": "fruit",
                        "vitamin": "vitamin_c",
                        "item_or_group": "Applesauce",
                        "k_per_year": 0.5941,
                    },
                    {
                        "record_type": "decay_item",
                        "category": "cereal (fortified)",
                        "vitamin": "vitamin_c",
                        "item_or_group": "Oatmeal w/ Brown Sugar",
                        "k_per_year": -0.0729,
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    applesauce = match_decay_rate("Applesauce", "T", "vitamin_c", table)
    peaches = match_decay_rate("Peaches", "T", "vitamin_c", table)
    oatmeal = match_decay_rate("Oatmeal w/ Brown Sugar", "R", "vitamin_c", table)
    assert applesauce.level == "decay_item"
    assert applesauce.k_per_year == pytest.approx(0.5941)
    assert peaches.level == "food_type"
    assert peaches.k_per_year == pytest.approx(0.3005)
    assert oatmeal.level == "decay_item"
    assert oatmeal.clipped
    assert oatmeal.k_per_year == 0

    foods = pd.DataFrame(
        [
            {"item": "Oatmeal w/ Brown Sugar", "food_type": "R", "vit_c_mg_per_serving_est": 80.0, "kcal": 10.0},
        ]
    )
    out = decayed_foods(foods, DAYS_PER_YEAR, table)
    assert out.loc[0, "vit_c_mg_per_serving_est"] == pytest.approx(80, rel=1e-6)
    assert out.loc[0, "kcal"] == 10


def test_group_range_is_not_averaged_and_fresh_food_stays_put():
    ranged = pd.DataFrame(
        [
            {
                "record_type": "decay_group",
                "category": "fruit products",
                "vitamin": "vitamin_c",
                "item_or_group": "fruit products (range low)",
                "k_per_year": 0.1286,
            },
            {
                "record_type": "decay_group",
                "category": "fruit products",
                "vitamin": "vitamin_c",
                "item_or_group": "fruit products (range high)",
                "k_per_year": 0.5907,
            },
        ]
    )
    fruit = match_decay_rate("Peaches", "T", "vitamin_c", ranged)
    assert fruit.level == "decay_group"
    assert fruit.k_per_year == pytest.approx(0.1286)
    assert fruit.k_per_year != pytest.approx((0.1286 + 0.5907) / 2)

    fresh = pd.DataFrame(
        [
            {
                "record_type": "fallback_default",
                "category": "fresh food (FF)",
                "vitamin": "all vitamins",
                "item_or_group": "DEFAULT",
                "k_per_year": None,
            },
            {
                "record_type": "decay_system",
                "category": "whole menu",
                "vitamin": "vitamin_c",
                "item_or_group": "system total",
                "k_per_year": 0.2545,
            },
            {
                "record_type": "fallback_default",
                "category": "any food",
                "vitamin": "vitamin_c",
                "item_or_group": "DEFAULT",
                "k_per_year": 0.5,
            },
        ]
    )
    salad = match_decay_rate("Garden Salad", "FF", "vitamin_c", fresh)
    assert salad.level == "food_type"
    assert salad.k_per_year == 0


def test_mineral_with_a_rate_decays_and_iron_does_not():
    table = pd.DataFrame(
        [
            {
                "record_type": "decay_system",
                "category": "whole menu",
                "vitamin": "potassium",
                "item_or_group": "system total",
                "k_per_year": 0.0039,
            }
        ]
    )
    foods = pd.DataFrame(
        [
            {"item": "Rice", "food_type": "T", "mass_g": 20.0, "kcal": 70.0, "potassium_mg": 100.0, "iron_mg": 4.0},
            {"item": "Tea", "food_type": "B", "mass_g": 5.0, "kcal": 1.0, "potassium_mg": 100.0, "iron_mg": 0.2},
        ]
    )
    out = decayed_foods(foods, DAYS_PER_YEAR, table).set_index("item")
    assert out.loc["Rice", "potassium_mg"] == pytest.approx(100 * math.exp(-0.0039), rel=1e-6)
    assert out.loc["Tea", "potassium_mg"] == pytest.approx(100, rel=1e-6)
    assert out.loc["Rice", "iron_mg"] == 4
    assert out.loc["Rice", "kcal"] == 70
    assert out.loc["Rice", "mass_g"] == 20


def test_packaged_table_uses_item_fruit_beverage_and_system_rates(package):
    fruit = match_decay_rate("Fruit Cocktail", "T", "vitamin_c", package.decay)
    applesauce = match_decay_rate("Applesauce", "T", "vitamin_c", package.decay)
    cider = match_decay_rate("Apple Cider", "B", "vitamin_c", package.decay)
    granola = match_decay_rate("Granola", "R", "vitamin_c", package.decay)
    tortillas = match_decay_rate("Tortillas", "NF", "thiamin", package.decay)
    steak = match_decay_rate("Beef Steak", "I", "vitamin_b6", package.decay)
    broccoli = match_decay_rate("Broccoli au Gratin", "R", "vitamin_c", package.decay)
    assert fruit.level == "food_type"
    assert fruit.k_per_year == pytest.approx(0.3005)
    assert applesauce.level == "decay_item"
    assert applesauce.k_per_year == pytest.approx(0.5941)
    assert cider.level == "food_type"
    assert cider.k_per_year == 0
    assert granola.level == "decay_system"
    assert granola.k_per_year == pytest.approx(0.2545)
    assert tortillas.level == "food_type"
    assert tortillas.k_per_year == pytest.approx(0.0786)
    assert steak.level == "food_type"
    assert steak.k_per_year == pytest.approx(0.0828)
    assert broccoli.level == "decay_item"
    assert broccoli.k_per_year == pytest.approx(0.1095)

    foods = package.foods.loc[package.foods["item"].isin(["Fruit Cocktail", "Apple Cider", "Applesauce"])].copy()
    out = decayed_foods(foods, DAYS_PER_YEAR, package.decay).set_index("item")
    base = foods.set_index("item")
    for item, k in (("Fruit Cocktail", 0.3005), ("Apple Cider", 0.0), ("Applesauce", 0.5941)):
        c0 = float(base.loc[item, "vit_c_mg_per_serving_est"])
        assert out.loc[item, "vit_c_mg_per_serving_est"] == pytest.approx(c0 * math.exp(-k), rel=1e-6)
        assert out.loc[item, "kcal"] == pytest.approx(float(base.loc[item, "kcal"]))


def test_epoch_starts_follow_the_open_end():
    assert epoch_starts(30) == [0]
    assert epoch_starts(31) == [0, 30]
    assert epoch_starts(90) == [0, 30, 60]
    assert epoch_starts(60) == [0, 30]
    assert epoch_starts(10, replan_every=4) == [0, 4, 8]


def test_shortfall_flags_the_day_the_locked_menu_crosses_the_target():
    k = math.log(2) * DAYS_PER_YEAR / 10.0
    foods = pd.DataFrame(
        [{"item": "Pill", "food_type": "T", "mass_g": 10.0, "vit_c_mg_per_serving_est": 100.0}]
    )
    table = pd.DataFrame(
        [
            {
                "record_type": "fallback_default",
                "category": "any food",
                "vitamin": "vitamin_c",
                "item_or_group": "DEFAULT",
                "k_per_year": k,
            }
        ]
    )
    epochs = [{"day": 0, "kind": "typical", "feasible": True, "servings": {"Pill": 1.0}}]
    projected = project_daily(
        20,
        foods,
        table,
        {"Pill": 1.0},
        epochs,
        {"vitamin_c": {"minimum": 50.0, "unit": "mg", "label": "Vitamin C"}},
    )
    by_day = {row["day"]: row for row in projected["daily"] if row["vitamin"] == "vitamin_c"}
    expected_short = []
    for day in range(20):
        total = concentration(100, k, day)
        flag = by_day[day]["shortfall_without_replan"]
        assert by_day[day]["no_replan_total"] == pytest.approx(total, rel=1e-6)
        if total > 50 + 1e-4:
            assert flag is False
        if total < 50 - 1e-3:
            assert flag is True
            expected_short.append(day)
        elif flag:
            expected_short.append(day)
    assert by_day[9]["shortfall_without_replan"] is False
    assert by_day[11]["shortfall_without_replan"] is True
    assert expected_short
    assert projected["shortfalls"] == [
        {
            "vitamin": "vitamin_c",
            "label": "Vitamin C",
            "unit": "mg",
            "first_day": expected_short[0],
            "days_short": len(expected_short),
        }
    ]
    assert f"day {expected_short[0]}" in describe_shortfalls(projected["shortfalls"])
    assert describe_shortfalls([]) == "No vitamin falls short of its minimum on the locked day-0 menu."


def test_plan_deltas_sort_by_absolute_mass_and_skip_unchanged_items():
    epochs = [
        {"day": 0, "kind": "typical", "feasible": True, "servings": {"A": 1.0, "B": 2.0}},
        {"day": 30, "kind": "typical", "feasible": False, "servings": {}},
        {"day": 60, "kind": "typical", "feasible": True, "servings": {"A": 1.5, "C": 0.4}},
    ]
    rows = plan_deltas(epochs, {"A": 10.0, "B": 5.0, "C": 100.0})
    assert [(row["item"], row["day"], row["previous_servings"], row["new_servings"], row["delta"]) for row in rows] == [
        ("C", 60, 0.0, 0.4, 0.4),
        ("B", 60, 2.0, 0.0, -2.0),
        ("A", 60, 1.0, 1.5, 0.5),
    ]
    assert rows[0]["abs_mass_delta_g"] == pytest.approx(40)
    assert rows[1]["abs_mass_delta_g"] == pytest.approx(10)
    assert rows[2]["abs_mass_delta_g"] == pytest.approx(5)


def _pill_foods() -> pd.DataFrame:
    return pd.DataFrame(
        [{"item": "Pill", "food_type": "T", "mass_g": 10.0, "kcal": 1.0, "vit_c_mg_per_serving_est": 100.0}]
    )


def _flat_decay() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "record_type": "fallback_default",
                "category": "any food",
                "vitamin": "vitamin_c",
                "item_or_group": "DEFAULT",
                "k_per_year": 0.0,
            }
        ]
    )


def _menu(servings: float, *, feasible: bool = True, unmet: list[str] | None = None) -> dict:
    if not feasible:
        return {
            "feasible": False,
            "foods": [{"item": "Pill", "servings": servings, "mass_g": servings * 10}],
            "total_mass_g": servings * 10,
            "unmet": unmet or ["vitamin_c"],
            "nutrients": {},
            "message": "Infeasible. Unmet nutrients: vitamin_c.",
        }
    return {
        "feasible": True,
        "foods": [{"item": "Pill", "servings": servings, "mass_g": servings * 10}],
        "total_mass_g": servings * 10,
        "unmet": [],
        "nutrients": {},
        "message": "ok",
    }


def test_replan_calls_the_solver_once_per_epoch_not_per_day():
    calls = {"n": 0, "hours": []}

    def solver(crew, *, foods, allergies=None, data_dir=None):
        calls["n"] += 1
        hours = crew["eva_hours"] if isinstance(crew, dict) else crew[0]["eva_hours"]
        calls["hours"].append(hours)
        servings = 1.0 if calls["n"] == 1 else 2.0
        return _menu(servings)

    result = replan_mission(
        31,
        CREW,
        replan_every=30,
        safety_margin=0.10,
        eva_day_fraction=0,
        foods=_pill_foods(),
        decay_table=_flat_decay(),
        solver=solver,
    )
    assert result["epoch_starts"] == [0, 30]
    assert calls["n"] == 2
    assert calls["hours"] == [0, 0]
    assert result["epochs"][0]["servings"] == {"Pill": 1.0}
    assert result["epochs"][1]["servings"] == {"Pill": 2.0}
    assert result["plan_changes"][0]["delta"] == pytest.approx(1.0)
    assert result["plan_changes"][0]["abs_mass_delta_g"] == pytest.approx(10)
    assert "not applied" in result["message"]


def test_replan_keeps_going_when_an_epoch_is_infeasible():
    calls = {"n": 0}

    def solver(crew, *, foods, allergies=None, data_dir=None):
        calls["n"] += 1
        if calls["n"] == 2:
            return _menu(9.0, feasible=False)
        return _menu(1.0)

    result = replan_mission(
        90,
        CREW,
        eva_day_fraction=0,
        foods=_pill_foods(),
        decay_table=_flat_decay(),
        solver=solver,
    )
    assert result["epoch_starts"] == [0, 30, 60]
    assert calls["n"] == 3
    assert result["epochs"][1]["feasible"] is False
    assert result["epochs"][1]["servings"] == {}
    assert result["epochs"][1]["unmet"] == ["vitamin_c"]
    assert result["epochs"][2]["feasible"] is True
    assert result["epochs"][2]["servings"] == {"Pill": 1.0}
    assert result["plan_changes"] == []


def test_eva_fraction_solves_both_days_and_charts_the_representative_menu():
    calls = {"hours": []}

    def solver(crew, *, foods, allergies=None, data_dir=None):
        hours = crew["eva_hours"] if isinstance(crew, dict) else crew[0]["eva_hours"]
        calls["hours"].append(hours)
        return _menu(4.0 if hours else 1.0)

    typical = replan_mission(
        30,
        CREW,
        eva_day_fraction=0,
        eva_hours=6.5,
        foods=_pill_foods(),
        decay_table=_flat_decay(),
        solver=solver,
    )
    assert calls["hours"] == [0]
    assert typical["representative_kind"] == "typical"
    assert typical["daily"][0]["no_replan_total"] == pytest.approx(100)

    calls["hours"].clear()
    mixed = replan_mission(
        30,
        CREW,
        eva_day_fraction=1 / 7,
        eva_hours=6.5,
        foods=_pill_foods(),
        decay_table=_flat_decay(),
        solver=solver,
    )
    assert calls["hours"] == [0, 6.5]
    assert mixed["representative_kind"] == "typical"
    assert {epoch["kind"] for epoch in mixed["epochs"]} == {"typical", "eva"}
    assert mixed["daily"][0]["no_replan_total"] == pytest.approx(100)

    calls["hours"].clear()
    eva = replan_mission(
        30,
        CREW,
        eva_day_fraction=1,
        eva_hours=6.5,
        foods=_pill_foods(),
        decay_table=_flat_decay(),
        solver=solver,
    )
    assert calls["hours"] == [0, 6.5]
    assert eva["representative_kind"] == "eva"
    assert eva["daily"][0]["no_replan_total"] == pytest.approx(400)
