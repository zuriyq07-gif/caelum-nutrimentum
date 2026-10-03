import pandas as pd

from data_loader import main


def test_source_row_counts(package):
    counts = package.row_counts.set_index("table")
    assert counts.loc["iss_menu", "rows"] == 186
    assert counts.loc["usda_vitamins", "rows"] == 186
    assert counts.loc["micronutrient_requirements", "rows"] == 273
    assert counts.loc["shelf_life_decay", "rows"] == 76
    assert counts.loc["foods", "rows"] == 183
    assert counts.loc["dropped", "rows"] == 3
    assert counts.loc["eer_equations", "rows"] == 2
    assert len(package.menu) == 186
    assert len(package.vitamins) == 186


def test_drop_and_flag_lists(package):
    assert set(package.dropped["item"]) == {
        "Caribbean Chicken",
        "Drinking Water Container",
        "Milk",
    }
    assert set(package.foods.loc[package.foods["quality_flag"], "item"]) == {
        "Chicken in Pouches",
        "Decaf. Coffee w/ C & A/S",
        "Decaf. Coffee w/ C & S",
        "Decaf. Coffee w/ Cream",
        "Green Beans w/ Mushrooms",
        "Grits w/ Butter",
        "Hot and Sour Soup",
        "Kona Coffee w/ C&S",
        "Lemon Meringue Pudding",
        "Maple Muffin Top",
        "Tea with Cream",
        "Tuna Salad Spread",
    }
    assert set(package.foods["item"]).isdisjoint(package.dropped["item"])
    assert len(package.foods) + len(package.dropped) == 186


def test_merge_keeps_macros_minerals_and_vitamins(package):
    foods = package.foods.set_index("item")
    almonds = foods.loc["Almonds"]
    assert almonds["protein_g"] == 11
    assert almonds["kcal"] == 243
    assert almonds["calcium_mg"] == 126
    assert almonds["vit_e_mg_per_serving_est"] == 10.755
    assert almonds["match_confidence"] == "high"
    assert almonds["quality_flag"] == False
    assert "food_type_x" not in package.foods.columns
    assert "food_type_y" not in package.foods.columns
    assert package.foods["match_confidence"].ne("none").all()
    assert package.foods["fdc_id"].notna().all()
    assert set(package.dropped["match_confidence"]) == {"none"}


def test_mineral_zeros_and_suspect_iron(package):
    foods = package.foods.set_index("item")
    # Sodium was printed; the other mineral zeros are the cells the note marks as missing.
    assert foods.loc["Chicken in Pouches", "sodium_mg"] == 260
    assert pd.isna(foods.loc["Chicken in Pouches", "potassium_mg"])
    assert foods.loc["Maple Muffin Top", "sodium_mg"] == 230
    assert pd.isna(foods.loc["Maple Muffin Top", "calcium_mg"])
    assert foods.loc["Lemon Meringue Pudding", "sodium_mg"] == 140
    assert pd.isna(foods.loc["Lemon Meringue Pudding", "iron_mg"])
    assert foods.loc["Tuna Salad Spread", "sodium_mg"] == 410
    assert pd.isna(foods.loc["Tuna Salad Spread", "magnesium_mg"])
    assert foods.loc["Grits w/ Butter", "iron_mg"] == 13
    assert foods.loc["Grits w/ Butter", "quality_reason"] == "iron value suspect"
    assert foods.loc["Chicken in Pouches", "quality_reason"] == "minerals likely missing"
    assert (
        foods.loc["Decaf. Coffee w/ C & S", "quality_reason"]
        == "saturated fat blank; kcal far from 4P+4C+9F"
    )
    assert package.minerals_recoded >= 14
    assert package.foods["sat_fat_g"].isna().sum() == 5


def test_atwater_gap_matches_kcal_notes(package):
    foods = package.foods.set_index("item")
    noted = package.foods.loc[package.foods["data_quality_notes"].str.contains("kcal check")]
    assert len(noted) == 3
    assert noted["atwater_gap_pct"].abs().gt(35).all()
    beans = foods.loc["Green Beans w/ Mushrooms"]
    assert beans["atwater_kcal"] == 4 * beans["protein_g"] + 4 * beans["carb_g"] + 9 * beans["fat_g"]


def test_green_tea_is_kept_but_not_scalable(package):
    tea = package.foods.set_index("item").loc["Green Tea"]
    assert tea["quality_flag"] == False
    assert str(tea["scaling_basis"]).startswith("not scalable")
    assert pd.isna(tea["serving_equiv_g"])
    assert "Green Tea" in package.summary_text
    assert "not scalable" in package.summary_text


def test_energy_tables(package):
    equations = package.eer_equations.set_index("sex")
    assert equations.loc["M", "intercept"] == 662
    assert equations.loc["F", "intercept"] == 354
    assert equations.loc["M", "age_coef"] == -9.53
    hidh_carb = package.macro_guidelines.loc[
        package.macro_guidelines["source"].eq("NASA HIDH")
        & package.macro_guidelines["nutrient"].eq("carbohydrate")
    ].iloc[0]
    assert hidh_carb["low"] == 50
    assert hidh_carb["high"] == 55
    provision = package.energy_parameters.loc[
        package.energy_parameters["section"].eq("provision")
        & package.energy_parameters["unit"].eq("kcal/day"),
        "value",
    ].iloc[0]
    assert provision == "3035"
    assert package.energy_rules["spaceflight_micronutrient_exception"]["vitamin_d_ug_per_day"] == 25


def test_decay_keeps_qualitative_bounds(package):
    text = package.decay["fraction_remaining_end"].astype(str)
    assert text.str.contains(">=0.85", regex=False).any()
    assert text.str.contains("undetectable", regex=False).any()
    applesauce = package.decay.loc[
        package.decay["item_or_group"].eq("Applesauce") & package.decay["vitamin"].eq("vitamin_c")
    ].iloc[0]
    assert applesauce["k_per_year"] == 0.5941


def test_missing_summary_and_cli(package, capsys):
    missing = package.missing_values
    sat = missing.loc[missing["table"].eq("foods") & missing["column"].eq("sat_fat_g")]
    assert sat.iloc[0]["missing"] == 5
    assert "Dropped: 3" in package.summary_text
    assert "Flagged (kept): 12" in package.summary_text
    assert "Caribbean Chicken" in package.summary_text
    assert "Missing values" in package.summary_text
    assert "662" in package.summary_text
    main([])
    printed = capsys.readouterr().out
    assert "Dropped: 3" in printed
    assert "iss_menu" in printed
    assert "186" in printed
    assert "273" in printed
    assert "76" in printed
