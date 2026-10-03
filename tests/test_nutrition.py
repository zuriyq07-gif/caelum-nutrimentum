import math

import numpy as np
import pandas as pd
import pytest

from nutrition import (
    daily_targets,
    decay_curve,
    eer,
    eer_components,
    fit_decay_items,
    life_stage_group,
    plan_day,
    remaining,
    serving_against_targets,
)


def test_life_stage_bands():
    assert life_stage_group("M", 19) == "male 19-30"
    assert life_stage_group("M", 30) == "male 19-30"
    assert life_stage_group("M", 31) == "male 31-50"
    assert life_stage_group("M", 50) == "male 31-50"
    assert life_stage_group("F", 51) == "female 51-70"
    assert life_stage_group("F", 70) == "female 51-70"
    assert life_stage_group("F", 71) == "female >70"
    with pytest.raises(ValueError):
        life_stage_group("M", 18)


def test_eer_reference_male(package):
    result = eer_components("M", 45, 82.9, 1.80, af=1.25, eva_h=6.5, rules=package.energy_rules)
    assert result.intercept == 662
    assert result.base_kcal == pytest.approx(3094.8875)
    assert result.eva_kcal == pytest.approx(1300)
    assert result.eer_kcal == pytest.approx(4394.8875)
    assert eer("M", 45, 82.9, 1.80, eva_h=6.5, rules=package.energy_rules) == pytest.approx(4394.8875)
    assert result.provision_kcal == 3035


def test_daily_targets_use_dri_and_nasa_vitamin_d(package):
    male = daily_targets(package.requirements, package.energy_rules, "M", 45, 82.9, 3000).set_index("nutrient")
    female = daily_targets(package.requirements, package.energy_rules, "F", 40, 65.1, 2300).set_index("nutrient")
    older = daily_targets(package.requirements, package.energy_rules, "M", 75, 82.9, 2500).set_index("nutrient")
    assert male.loc["Vitamin D", "target_low"] == 25
    assert older.loc["Vitamin D", "target_low"] == 25
    assert male.loc["Iron", "target_low"] == 8
    assert female.loc["Iron", "target_low"] == 18
    assert male.loc["Protein", "target_low"] == pytest.approx(0.8 * 82.9)
    assert male.loc["Sodium", "target_high"] == 2300
    assert male.loc["Iron", "target_high"] == 45
    assert pd.isna(male.loc["Magnesium", "target_high"])
    assert "not applied" in male.loc["Magnesium", "note"]
    assert pd.isna(male.loc["Niacin", "target_high"])
    assert male.loc["Fiber", "target_low"] == pytest.approx(30)
    assert len(male) == 24


def test_serving_coverage_for_almonds(package):
    targets = daily_targets(package.requirements, package.energy_rules, "M", 45, 82.9, 3000)
    almonds = package.foods.set_index("item").loc["Almonds"]
    coverage = serving_against_targets(almonds, targets).set_index("nutrient")
    assert coverage.loc["Protein", "per_serving"] == 11
    assert coverage.loc["Protein", "percent_of_reference"] == pytest.approx(11 / (0.8 * 82.9) * 100)
    assert coverage.loc["Vitamin E", "per_serving"] == pytest.approx(10.755)


def test_remaining_fruit_vitamin_c(package):
    estimate = remaining(100, "vitamin_c", 1.5, "fruit", package.decay)
    assert estimate.category == "fruit (any type)"
    assert estimate.k_per_year == pytest.approx(0.3005)
    assert estimate.amount == pytest.approx(100 * math.exp(-0.3005 * 1.5))
    times, amounts = decay_curve(100, estimate.k_per_year, 1.5)
    assert amounts[-1] == pytest.approx(estimate.amount, rel=1e-3)
    assert estimate.half_life_years == pytest.approx(math.log(2) / 0.3005)

    flat = remaining(80, "vitamin_c", 2, "fortified beverage powder", package.decay)
    assert flat.k_per_year == 0
    assert flat.amount == pytest.approx(80)

    folate = remaining(100, "folate", 1.5, "any food", package.decay)
    assert folate.amount is None


def test_decay_curve_half_life():
    k = math.log(2) / 2
    _times, amounts = decay_curve(100, k, 2)
    assert amounts[0] == pytest.approx(100, rel=1e-6)
    assert amounts[-1] == pytest.approx(50, rel=1e-3)


def test_curve_fit_recovers_known_rate():
    k_true = 0.4
    c0 = 100.0
    frame = pd.DataFrame(
        [
            {
                "record_type": "decay_item",
                "item_or_group": "Synthetic",
                "vitamin": "vitamin_c",
                "c_initial": c0,
                "c_1yr": c0 * np.exp(-k_true),
                "c_3yr": c0 * np.exp(-k_true * 3),
                "k_per_year": k_true,
            }
        ]
    )
    fit = fit_decay_items(frame).iloc[0]
    assert fit["k_curve_fit"] == pytest.approx(k_true, rel=1e-3)
    assert fit["k_loglinear"] == pytest.approx(k_true, rel=1e-3)
    assert fit["rmse"] == pytest.approx(0, abs=1e-6)


def test_published_applesauce_fit(package):
    fit = fit_decay_items(package.decay)
    assert len(fit) > 10
    applesauce = fit.loc[fit["item_or_group"].eq("Applesauce") & fit["vitamin"].eq("vitamin_c")].iloc[0]
    assert applesauce["k_published"] == pytest.approx(0.5941)
    assert applesauce["k_curve_fit"] > 0
    assert np.isfinite(applesauce["k_loglinear"])


@pytest.mark.parametrize("sex,kg", [("M", 82.9), ("F", 65.1)])
def test_day_plan_hits_energy(package, sex, kg):
    plan = plan_day(
        package.foods,
        package.requirements,
        package.energy_rules,
        sex=sex,
        age=45,
        kg=kg,
        height_m=1.80,
        af=1.25,
        eva_h=0,
    )
    assert plan.success, plan.message
    assert plan.packages_chosen >= 8
    assert plan.servings["packages"].eq(1).all()
    assert plan.servings["food_type"].eq("B").sum() <= 4
    totals = plan.totals.set_index("nutrient")
    energy = float(totals.loc["Energy", "amount"])
    assert energy == pytest.approx(plan.target_kcal, rel=0.05)
    assert totals.loc["Energy", "status"] == "met"
    assert float(totals.loc["Protein", "amount"]) >= 0.8 * kg - 1
