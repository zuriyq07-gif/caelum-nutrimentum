"""Crew-day targets: NASA EER, HIDH macros, and spaceflight micronutrient rows."""

import pytest

from nutrition import eer
from targets import daily_targets

REQUIRED_NUTRIENTS = [
    "energy",
    "protein",
    "carbohydrate",
    "fat",
    "saturated_fat",
    "fiber",
    "vitamin_a",
    "vitamin_c",
    "vitamin_d",
    "vitamin_e",
    "vitamin_k",
    "thiamin",
    "riboflavin",
    "niacin",
    "vitamin_b6",
    "vitamin_b12",
    "folate",
    "calcium",
    "iron",
    "magnesium",
    "zinc",
    "potassium",
    "sodium",
    "phosphorus",
]


def test_male_eva_day_uses_nasa_energy_and_spaceflight_rows(package):
    male = daily_targets(45, "M", 82.9, 1.80, eva_hours=6.5, activity_factor=1.25)
    spelled = daily_targets(45, "male", 82.9, 1.80, eva_hours=6.5, activity_factor=1.25)
    assert male == spelled
    _assert_target_dict(male)

    expected_kcal = eer("M", 45, 82.9, 1.80, af=1.25, eva_h=6.5, rules=package.energy_rules)
    assert male["energy"]["minimum"] == pytest.approx(4394.8875)
    assert male["energy"]["maximum"] == pytest.approx(4394.8875)
    assert male["energy"]["minimum"] == pytest.approx(expected_kcal)
    assert male["energy"]["unit"] == "kcal"

    # Protein floor is 0.8 g/kg body weight, not a percent of energy.
    assert male["protein"]["minimum"] == pytest.approx(0.8 * 82.9)
    assert male["protein"]["unit"] == "g"
    _assert_macro_bands(male)

    assert male["vitamin_d"]["minimum"] == 25
    assert male["vitamin_d"]["minimum"] != 15
    assert male["vitamin_d"]["unit"] == "ug"

    # NASA spaceflight iron minimum is 8 mg for all crew. The male 31-50 DRI is also 8 mg.
    # The NASA band maximum is target_max 10 mg.
    assert male["iron"]["minimum"] == 8
    assert male["iron"]["maximum"] == 10
    assert male["iron"]["unit"] == "mg"

    assert male["sodium"]["minimum"] == 1500
    assert male["sodium"]["maximum"] == 2300
    assert male["sodium"]["unit"] == "mg"

    assert male["potassium"]["minimum"] == 4700
    assert male["calcium"]["minimum"] == 1200
    assert male["calcium"]["maximum"] == 2000
    assert male["copper"]["minimum"] == pytest.approx(0.5)
    assert male["copper"]["maximum"] == pytest.approx(9)
    assert male["copper"]["unit"] == "mg"


def test_female_nominal_day_prefers_nasa_iron_over_dri(package):
    age, kg, height = 32, 65.1, 1.70
    # Published female EER: 354 - 6.91*age + AF*(9.36*kg + 726*m), AF nominal 1.25, no EVA.
    activity_factor = 1.25
    expected_kcal = 354 - 6.91 * age + activity_factor * (9.36 * kg + 726 * height)

    female = daily_targets(age, "female", kg, height, eva_hours=0)
    explicit = daily_targets(age, "F", kg, height, eva_hours=0.0, activity_factor=activity_factor)
    assert female["energy"] == explicit["energy"]
    _assert_target_dict(female)

    assert female["energy"]["minimum"] == pytest.approx(expected_kcal)
    assert female["energy"]["maximum"] == pytest.approx(expected_kcal)
    assert female["energy"]["minimum"] == pytest.approx(
        eer("F", age, kg, height, af=activity_factor, eva_h=0, rules=package.energy_rules)
    )
    assert female["energy"]["unit"] == "kcal"

    assert female["protein"]["minimum"] == pytest.approx(0.8 * kg)
    _assert_macro_bands(female)

    assert female["vitamin_d"]["minimum"] == 25
    assert female["vitamin_d"]["minimum"] != 15
    assert female["vitamin_d"]["unit"] == "ug"

    # DRI for female 31-50 iron is 18 mg. NASA spaceflight iron applies to all crew,
    # so the minimum is the NASA value 8 mg and target_max is 10 mg, not the DRI 18.
    assert female["iron"]["minimum"] == 8, (
        "NASA spaceflight iron minimum is 8 mg for all crew, not the female 31-50 DRI of 18 mg"
    )
    assert female["iron"]["maximum"] == 10, (
        "NASA spaceflight iron target_max is 10 mg, not the DRI of 18 mg or the UL of 45 mg"
    )
    assert female["iron"]["unit"] == "mg"

    # NASA all-crew vitamin C is 90 mg; the female DRI is 75 mg.
    assert female["vitamin_c"]["minimum"] == 90
    assert female["sodium"]["minimum"] == 1500
    assert female["sodium"]["maximum"] == 2300


def test_supplement_only_uls_are_not_food_caps():
    male = daily_targets(45, "M", 82.9, 1.80, activity_factor=1.25)
    for nutrient in ("magnesium", "niacin", "folate", "vitamin_e"):
        assert "maximum" not in male[nutrient], nutrient
    # Preformed vitamin A UL is not a food cap. The NASA food band is 700–900 µg.
    assert male["vitamin_a"]["minimum"] == 700
    assert male["vitamin_a"]["maximum"] == 900
    assert male["vitamin_a"]["unit"] == "ug"


def test_age_bands_and_vitamin_d_above_70():
    age_30 = daily_targets(30, "M", 80, 1.80)
    age_31 = daily_targets(31, "M", 80, 1.80)
    age_70 = daily_targets(70, "F", 60, 1.65)
    age_71 = daily_targets(71, "F", 60, 1.65)
    # Chloride has no NASA row, so the DRI band is the target. 30 and 31 share 2.3 g;
    # female 70 is the 51-70 band and 71 is >70.
    assert age_30["chloride"]["minimum"] == pytest.approx(2.3)
    assert age_31["chloride"]["minimum"] == pytest.approx(2.3)
    assert age_70["chloride"]["minimum"] == pytest.approx(2.0)
    assert age_71["chloride"]["minimum"] == pytest.approx(1.8)
    assert age_71["chloride"]["unit"] == "g"
    # Male 19-30 DRI magnesium is 400 mg. The NASA male row is 420 mg.
    assert age_30["magnesium"]["minimum"] == 420
    assert "maximum" not in age_30["magnesium"]
    # >70 DRI vitamin D is 20 µg. Flight value stays 25 µg.
    assert age_71["vitamin_d"]["minimum"] == 25
    assert age_30["vitamin_d"]["minimum"] == 25


def test_rejects_age_under_19_and_unknown_sex():
    with pytest.raises(ValueError):
        daily_targets(18, "F", 60, 1.65)
    with pytest.raises(ValueError):
        daily_targets(18.9, "male", 70, 1.75)
    with pytest.raises(ValueError):
        daily_targets(32, "unknown", 65.1, 1.70)
    with pytest.raises(ValueError):
        daily_targets(32, "", 65.1, 1.70)


def _assert_target_dict(targets: dict) -> None:
    assert isinstance(targets, dict)
    for key in REQUIRED_NUTRIENTS:
        assert key in targets, key
    for name, entry in targets.items():
        assert isinstance(entry, dict), name
        unit = entry.get("unit")
        assert isinstance(unit, str) and unit, name
        assert unit in {"kcal", "g", "mg", "ug"}, name
        assert not unit.endswith("/d") and not unit.endswith("/day"), name
        numbers = [entry[bound] for bound in ("minimum", "maximum") if bound in entry]
        assert numbers, name
        assert all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in numbers), name


def _assert_macro_bands(targets: dict) -> None:
    """Carbohydrate and fat expose low and high gram targets from the HIDH percents."""

    kcal = targets["energy"]["minimum"]
    carb = targets["carbohydrate"]
    fat = targets["fat"]
    assert carb["unit"] == "g"
    assert fat["unit"] == "g"
    assert carb["minimum"] == pytest.approx(0.50 * kcal / 4.0)
    assert carb["maximum"] == pytest.approx(0.55 * kcal / 4.0)
    assert carb["minimum"] < carb["maximum"]
    assert fat["minimum"] == pytest.approx(0.25 * kcal / 9.0)
    assert fat["maximum"] == pytest.approx(0.35 * kcal / 9.0)
    assert fat["minimum"] < fat["maximum"]
    assert targets["protein"]["maximum"] == pytest.approx(0.35 * kcal / 4.0)
    assert targets["saturated_fat"]["maximum"] == pytest.approx(0.07 * kcal / 9.0)
    assert "minimum" not in targets["saturated_fat"]
    assert targets["fiber"]["minimum"] == pytest.approx(10.0 * kcal / 1000.0)
    assert targets["fiber"]["maximum"] == pytest.approx(14.0 * kcal / 1000.0)
    assert targets["fiber"]["unit"] == "g"
