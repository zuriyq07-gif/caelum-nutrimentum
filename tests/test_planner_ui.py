"""UI helpers for the mission planner: EVA schedule, CSV rows, nutrient blend."""

import pytest

from app import (
    VISUAL_PERCENT_CAP,
    eva_schedule,
    mission_average_coverage,
    packing_list_csv,
    validate_mission_inputs,
)

CREW = [
    {"age": 45, "sex": "male", "weight_kg": 82.9, "height_m": 1.80, "allergies": []},
]


def test_eva_schedule_zero_is_typical_days_only():
    schedule = eva_schedule(0)
    assert schedule["eva_day_fraction"] == 0.0
    assert schedule["eva_hours"] == 0.0
    assert schedule["note"] is None


def test_eva_schedule_spreads_a_standard_week():
    schedule = eva_schedule(6.5)
    assert schedule["eva_day_fraction"] == pytest.approx(6.5 / 45.5)
    assert schedule["eva_hours"] == pytest.approx(6.5)
    assert schedule["note"] is None

    full = eva_schedule(45.5)
    assert full["eva_day_fraction"] == pytest.approx(1.0)
    assert full["eva_hours"] == pytest.approx(6.5)
    assert full["note"] is None


def test_eva_schedule_above_a_full_week_uses_every_day():
    schedule = eva_schedule(70)
    assert schedule["eva_day_fraction"] == pytest.approx(1.0)
    assert schedule["eva_hours"] == pytest.approx(10.0)
    assert schedule["note"]
    assert "every day" in schedule["note"].lower()
    assert "10.00" in schedule["note"]


def test_eva_schedule_rejects_negative_hours():
    with pytest.raises(ValueError, match="negative"):
        eva_schedule(-1)


def test_validate_rejects_zero_days_and_underage_crew():
    errors = validate_mission_inputs(0, 0.10, 6.5, [{"age": 10, "sex": "male", "weight_kg": 40, "height_m": 1.5}])
    text = " ".join(errors)
    assert "more than zero days" in text
    assert "under 19" in text


def test_validate_rejects_non_positive_body_size_and_negative_margin():
    errors = validate_mission_inputs(
        30,
        -0.1,
        -2,
        [{"age": 45, "sex": "male", "weight_kg": 0, "height_m": 0}],
    )
    text = " ".join(errors)
    assert "Safety margin" in text
    assert "EVA hours" in text
    assert "weight" in text
    assert "height" in text


def test_validate_accepts_the_default_crew():
    assert validate_mission_inputs(30, 0.10, 6.5, CREW) == []


def test_packing_list_csv_columns_and_total():
    csv_text = packing_list_csv(
        [
            {"item": "Tuna salad, kit", "servings": 12.345, "mass_kg": 1.23456},
            {"item": "Tortilla", "servings": 4, "mass_kg": 0.2},
        ]
    )
    lines = csv_text.strip().split("\n")
    assert lines[0] == "food,servings,mass_kg"
    assert lines[1] == '"Tuna salad, kit",12.35,1.235'
    assert lines[2] == "Tortilla,4.00,0.200"
    assert lines[3] == "Total,,1.435"


def test_mission_average_blends_days_and_skips_empty_minimums():
    result = {
        "days": 10,
        "typical_days": 7.5,
        "eva_days": 2.5,
        "typical_day": {
            "nutrients": {
                "energy": {"amount": 2000, "minimum": 2000, "unit": "kcal"},
                "sodium": {"amount": 1500, "minimum": 0, "unit": "mg"},
                "vitamin_d": {"amount": 10, "minimum": None, "unit": "ug"},
                "folate": {"amount": 100, "minimum": 1, "unit": "ug"},
            }
        },
        "eva_day": {
            "nutrients": {
                "energy": {"amount": 3000, "minimum": 2500, "unit": "kcal"},
                "sodium": {"amount": 1600, "minimum": 0, "unit": "mg"},
                "vitamin_d": {"amount": 25, "minimum": 25, "unit": "ug"},
                "folate": {"amount": 100, "minimum": 1, "unit": "ug"},
            }
        },
    }
    rows = {row["nutrient"]: row for row in mission_average_coverage(result)}
    assert set(rows) == {"energy", "folate"}
    energy = rows["energy"]
    assert energy["delivered"] == pytest.approx(2250)
    assert energy["target_min"] == pytest.approx(2125)
    assert energy["percent"] == pytest.approx(2250 / 2125 * 100)
    assert energy["unit"] == "kcal"
    assert energy["clipped"] is False
    folate = rows["folate"]
    assert folate["percent"] == pytest.approx(10_000)
    assert folate["display_percent"] == pytest.approx(VISUAL_PERCENT_CAP)
    assert folate["clipped"] is True
    assert folate["unit"] == "µg"


def test_mission_average_ignores_a_day_with_zero_length():
    result = {
        "days": 10,
        "typical_days": 10,
        "eva_days": 0,
        "typical_day": {"nutrients": {"protein": {"amount": 80, "minimum": 66, "unit": "g"}}},
        "eva_day": {"nutrients": {"protein": {"amount": 90, "minimum": None, "unit": "g"}}},
    }
    rows = mission_average_coverage(result)
    assert len(rows) == 1
    assert rows[0]["delivered"] == pytest.approx(80)
    assert rows[0]["target_min"] == pytest.approx(66)
    assert rows[0]["label"] == "Protein"
