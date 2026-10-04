"""Minimum-mass menu: one crew day, an allergy, and a vitamin the foods cannot cover."""

import pandas as pd
import pytest

from optimizer import FOOD_COLUMNS, optimize_menu
from targets import daily_targets

MALE = {
    "age": 45,
    "sex": "M",
    "weight_kg": 82.9,
    "height_m": 1.80,
    "eva_hours": 0,
}

RESULT_KEYS = {"feasible", "foods", "total_mass_g", "nutrients", "unmet", "message"}


@pytest.fixture(scope="module")
def male_menu():
    return optimize_menu([MALE])


def test_single_male_minimum_mass_menu(male_menu):
    result = male_menu
    _assert_result_shape(result)
    if result["feasible"]:
        assert result["unmet"] == []
        assert result["total_mass_g"] > 0
        reported = 0.0
        for food in result["foods"]:
            assert food["servings"] > 1e-4
            assert food["servings"] <= 2.0 + 1e-6
            assert food["mass_g"] > 0
            reported += food["mass_g"]
        assert result["total_mass_g"] == pytest.approx(reported, abs=0.1)
        sodium = result["nutrients"]["sodium"]
        assert sodium["amount"] <= 2300.0 + 1e-4
        assert sodium["maximum"] is not None and sodium["maximum"] <= 2300.0
        energy = result["nutrients"]["energy"]
        assert energy["maximum"] is None
        assert energy["amount"] >= energy["minimum"] - 1e-3
        for info in result["nutrients"].values():
            if info["minimum"] is not None:
                assert info["met"] is True
                assert info["amount"] >= info["minimum"] - max(1e-3, 1e-5 * info["minimum"])
            if info["maximum"] is not None:
                assert info["amount"] <= info["maximum"] + max(1e-3, 1e-5 * abs(info["maximum"]))
        again = optimize_menu([MALE])
        assert again["feasible"] is True
        assert again["total_mass_g"] == pytest.approx(result["total_mass_g"], abs=1e-4)
        assert [food["item"] for food in again["foods"]] == [food["item"] for food in result["foods"]]
        for first, second in zip(result["foods"], again["foods"]):
            assert second["servings"] == pytest.approx(first["servings"], abs=1e-6)
            assert second["mass_g"] == pytest.approx(first["mass_g"], abs=1e-4)
    else:
        assert result["unmet"]
        for name in result["unmet"]:
            assert name in result["message"]


def test_almond_allergy_excludes_almonds_and_does_not_cut_mass(male_menu):
    allergic = optimize_menu([{**MALE, "allergies": ["almond"]}])
    _assert_result_shape(allergic)
    for food in allergic["foods"]:
        assert "almond" not in food["item"].casefold()
    if male_menu["feasible"]:
        if allergic["feasible"]:
            assert allergic["total_mass_g"] >= male_menu["total_mass_g"] - 0.1
        else:
            assert allergic["unmet"]
            for name in allergic["unmet"]:
                assert name in allergic["message"]


def test_foods_without_vitamin_d_report_it_unmet():
    foods = pd.DataFrame(
        [
            {
                "item": "Rice",
                "mass_g": 100.0,
                "kcal": 130.0,
                "protein_g": 2.4,
                "carb_g": 28.0,
                "fat_g": 0.3,
                "sodium_mg": 1.0,
            },
            {
                "item": "Bread",
                "mass_g": 50.0,
                "kcal": 130.0,
                "protein_g": 4.0,
                "carb_g": 24.0,
                "fat_g": 1.0,
                "sodium_mg": 150.0,
            },
        ]
    )
    result = optimize_menu([dict(MALE, sex="male")], foods=foods)
    _assert_result_shape(result)
    assert result["feasible"] is False
    assert result["unmet"]
    assert "vitamin_d" in result["unmet"]
    assert "vitamin_d" in result["message"]
    for name in result["unmet"]:
        assert name in result["message"]


def test_allergy_shorter_than_three_characters_is_ignored():
    targets = daily_targets(MALE["age"], MALE["sex"], MALE["weight_kg"], MALE["height_m"], eva_hours=0)
    foods = pd.DataFrame([_food_inside_bands("Almonds", 100.0, targets)])
    kept = optimize_menu([MALE], foods=foods, allergies=["a"])
    blocked = optimize_menu([MALE], foods=foods, allergies="almond")
    assert kept["feasible"] is True
    assert any("almond" in food["item"].casefold() for food in kept["foods"])
    assert blocked["feasible"] is False
    assert "vitamin_d" in blocked["unmet"]
    assert "vitamin_d" in blocked["message"]
    assert all("almond" not in food["item"].casefold() for food in blocked["foods"])


def _food_inside_bands(item: str, mass_g: float, targets: dict) -> dict:
    """One serving whose nutrients sit inside the optimizer's enforced bands."""

    row: dict = {"item": item, "mass_g": mass_g}
    for name, column in FOOD_COLUMNS.items():
        entry = targets[name]
        low = entry.get("minimum")
        high = entry.get("maximum")
        if name == "energy":
            high = None
        if low is not None and high is not None:
            value = (float(low) + float(high)) / 2.0
        elif low is not None:
            value = float(low) * 1.05
        elif high is not None:
            value = float(high) * 0.5
        else:
            continue
        row[column] = value
    return row


def _assert_result_shape(result: dict) -> None:
    assert RESULT_KEYS <= set(result)
    assert isinstance(result["feasible"], bool)
    assert isinstance(result["foods"], list)
    assert isinstance(result["nutrients"], dict)
    assert isinstance(result["unmet"], list)
    assert isinstance(result["message"], str) and result["message"]
    assert isinstance(result["total_mass_g"], float)
    for food in result["foods"]:
        assert set(food) >= {"item", "servings", "mass_g"}
        assert isinstance(food["item"], str) and food["item"]
        assert food["servings"] > 1e-4
    for name, info in result["nutrients"].items():
        assert isinstance(name, str) and name
        assert set(info) >= {"amount", "unit", "minimum", "maximum", "met"}
        assert isinstance(info["met"], bool)
        assert isinstance(info["unit"], str) and info["unit"]
