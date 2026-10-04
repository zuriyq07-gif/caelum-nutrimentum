"""Mission packing: scale two minimum-mass days, and stop when a day is infeasible."""

import json
from pathlib import Path

import pandas as pd
import pytest

from mission import EVA_FRACTION_FORMULA, eva_fraction_from_spacewalks, mission_food
from optimizer import FOOD_COLUMNS
from targets import daily_targets

CREW = {
    "age": 45,
    "sex": "M",
    "weight_kg": 82.9,
    "height_m": 1.80,
}

RESULT_KEYS = {
    "feasible",
    "days",
    "typical_days",
    "eva_days",
    "eva_day_fraction",
    "eva_day_fraction_source",
    "safety_margin",
    "packing_list",
    "total_mass_kg",
    "typical_day",
    "eva_day",
    "message",
}

SPACEWALKS = Path(__file__).resolve().parents[1] / "data" / "launch_library" / "spacewalks.json"


def test_margin_and_scale():
    foods = _two_foods(CREW)
    result = mission_food(
        10,
        CREW,
        safety_margin=0.10,
        eva_day_fraction=0.25,
        foods=foods,
        allergies=None,
    )
    assert RESULT_KEYS <= set(result)
    assert result["feasible"] is True
    assert result["days"] == pytest.approx(10)
    assert result["eva_day_fraction"] == pytest.approx(0.25)
    assert result["typical_days"] == pytest.approx(7.5)
    assert result["eva_days"] == pytest.approx(2.5)
    assert result["safety_margin"] == pytest.approx(0.10)
    assert result["typical_day"]["feasible"] is True
    assert result["eva_day"]["feasible"] is True
    assert "caller" in result["eva_day_fraction_source"]
    assert "ISS EVA count" in result["eva_day_fraction_source"]
    assert EVA_FRACTION_FORMULA in result["eva_day_fraction_source"]

    typical = _servings(result["typical_day"])
    eva = _servings(result["eva_day"])
    assert result["packing_list"]
    masses = [row["mass_kg"] for row in result["packing_list"]]
    assert masses == sorted(masses, reverse=True)
    assert result["total_mass_kg"] == pytest.approx(sum(masses), abs=1e-12)

    same_single_serving = (
        len(typical) == 1
        and typical.keys() == eva.keys()
        and next(iter(typical.values())) == pytest.approx(1.0)
        and next(iter(eva.values())) == pytest.approx(1.0)
    )
    if same_single_serving:
        assert len(result["packing_list"]) == 1
        row = result["packing_list"][0]
        grams = _grams_per_serving(result, row["item"])
        assert row["servings"] == pytest.approx(10 * 1 * 1.10)
        assert row["mass_kg"] == pytest.approx(row["servings"] * grams / 1000.0)
    else:
        for row in result["packing_list"]:
            expected = (typical.get(row["item"], 0.0) * 7.5 + eva.get(row["item"], 0.0) * 2.5) * 1.10
            grams = _grams_per_serving(result, row["item"])
            assert row["servings"] == pytest.approx(expected, abs=1e-3)
            assert row["mass_kg"] == pytest.approx(row["servings"] * grams / 1000.0, rel=1e-9)
        assert result["total_mass_kg"] == pytest.approx(
            sum(row["mass_kg"] for row in result["packing_list"]),
            abs=1e-12,
        )


def test_missing_spacewalks_file_requires_fraction(tmp_path):
    foods = pd.DataFrame([{"item": "Rice", "mass_g": 100.0, "kcal": 100.0}])
    with pytest.raises((FileNotFoundError, ValueError), match="eva_day_fraction"):
        mission_food(10, CREW, foods=foods, data_dir=tmp_path)


def test_infeasible_menu_names_the_failed_day():
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
    result = mission_food(10, [CREW], foods=foods, eva_day_fraction=0.2)
    assert result["feasible"] is False
    assert result["packing_list"] == []
    assert result["total_mass_kg"] is None
    assert result["message"]
    for label, day in (("typical", result["typical_day"]), ("EVA", result["eva_day"])):
        assert day["feasible"] is False
        assert day["unmet"]
        assert label in result["message"]
        for nutrient in day["unmet"]:
            assert nutrient in result["message"]
    assert "vitamin_d" in result["message"]


def test_eva_fraction_from_spacewalks_file():
    if not SPACEWALKS.is_file():
        with pytest.raises((FileNotFoundError, ValueError), match="eva_day_fraction"):
            eva_fraction_from_spacewalks(SPACEWALKS)
        return
    fraction = eva_fraction_from_spacewalks(SPACEWALKS)
    assert isinstance(fraction, float)
    assert 0.0 <= fraction <= 1.0


def test_eva_fraction_formula_on_synthetic_cache(tmp_path):
    iss = {"name": "International Space Station"}
    tiangong = {"name": "Tiangong"}
    path = tmp_path / "spacewalks.json"
    path.write_text(
        json.dumps(
            {
                "results": [
                    {"start": "2020-01-01T00:00:00Z", "spacestation": iss},
                    {"start": "2020-01-11T00:00:00Z", "spacestation": iss},
                    {"start": "2020-01-05T00:00:00Z", "spacestation": tiangong},
                ]
            }
        ),
        encoding="utf-8",
    )
    # Two ISS EVAs, ten days apart. The Tiangong row is not in the count.
    assert eva_fraction_from_spacewalks(path) == pytest.approx(2 / 10)

    clipped = tmp_path / "cluster.json"
    clipped.write_text(
        json.dumps(
            {
                "results": [
                    {"start": "2020-01-01T00:00:00Z", "spacestation": iss},
                    {"start": "2020-01-01T12:00:00Z", "spacestation": iss},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert eva_fraction_from_spacewalks(clipped) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"days": 0},
        {"days": -3},
        {"safety_margin": -0.01},
        {"eva_day_fraction": -0.01},
        {"eva_day_fraction": 1.5},
    ],
)
def test_rejects_non_positive_days_and_negative_margin(kwargs):
    foods = pd.DataFrame([{"item": "Rice", "mass_g": 100.0, "kcal": 100.0}])
    call = {
        "days": 10,
        "safety_margin": 0.10,
        "eva_day_fraction": 0.25,
        "foods": foods,
    }
    call.update(kwargs)
    with pytest.raises(ValueError):
        mission_food(call.pop("days"), CREW, **call)


def _two_foods(crew: dict) -> pd.DataFrame:
    """One serving of Base meets a typical day. Boost adds the EVA macro gap.

    Micronutrient floors do not rise on an EVA day, and iron's 8–10 mg band
    cannot scale with the extra energy, so the EVA menu needs a second food
    with those micronutrients left at 0.
    """

    typical = daily_targets(crew["age"], crew["sex"], crew["weight_kg"], crew["height_m"], eva_hours=0)
    eva = daily_targets(crew["age"], crew["sex"], crew["weight_kg"], crew["height_m"], eva_hours=6.5)
    base = {"item": "Base", "mass_g": 100.0}
    boost = {"item": "Boost", "mass_g": 40.0}
    scaling = {"energy", "carbohydrate", "fat", "fiber"}
    for name, column in FOOD_COLUMNS.items():
        low = typical[name].get("minimum")
        if name in scaling:
            base[column] = float(low)
            boost[column] = float(eva[name]["minimum"]) - float(low)
        elif low is not None:
            base[column] = float(low)
            boost[column] = 0.0
        else:
            base[column] = 0.0
            boost[column] = 0.0
    return pd.DataFrame([base, boost])


def _servings(day: dict) -> dict[str, float]:
    totals: dict[str, float] = {}
    for food in day["foods"]:
        totals[food["item"]] = totals.get(food["item"], 0.0) + float(food["servings"])
    return totals


def _grams_per_serving(result: dict, item: str) -> float:
    for day in (result["typical_day"], result["eva_day"]):
        for food in day["foods"]:
            if food["item"] == item and food["servings"] > 0:
                return float(food["mass_g"]) / float(food["servings"])
    raise AssertionError(f"{item} is not on either day menu")
