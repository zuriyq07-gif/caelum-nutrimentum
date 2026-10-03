# Astronaut Food Optimizer: Data Package

Built 2026-10-03 (ET). Every value comes from a fetched source. Columns or fields that are derived, digitized or assumed are labelled as such (`value_status`, `*_DERIVED`, `*_est`, `data_quality_notes`).

| File | Rows / size | What it is |
|---|---|---|
| `iss_menu.csv` | 186 items | ISS standard menu: mass, macros and minerals (NASA STEMonstrations Nutrition, Appendix A) |
| `micronutrient_requirements.csv` | 273 rows | NASEM DRI (2019 Appendix J) plus NASA spaceflight guidelines |
| `energy_rules.json` | 1 object | NASA EER equations, activity factor, EVA +200 kcal/h, macro rules |
| `shelf_life_decay.csv` | 76 rows | Shelf life by food type, vitamin decay data, derived first-order k values, fallback k for user foods |
| `usda_vitamins.csv` | 186 items | Each menu item matched to USDA FDC (SR Legacy / FNDDS): 11 vitamins per 100 g, plus a per-serving estimate |
| `launch_library/*.json` | 100 / 100 / 100 / 14 | Launch Library 2 (v2.3.0) cache: spacewalks, dockings, expeditions, upcoming ISS resupply and crew flights |
| `crew_timeline_template.json` | 109 events | Real ISS day plan (10/03/2014) parsed into meal, exercise, sleep and work blocks, plus a derived template |
| `food_system_structure.json` | 1 object | ISS food-system composition (75% standard / 25% specialty), menu categories, food types |
| `scripts/` | | Build scripts, the vision transcription and raw OCR (for reproducibility) |

---
## 1. iss_menu.csv
**Source:** NASA STEMonstrations "Nutrition" educator guide, Appendix A (PDF pp. 10–18): https://www.nasa.gov/wp-content/uploads/2023/03/stemonstrations-nutrition.pdf

**Method:** Pages were rendered at 300 dpi and transcribed visually. Tesseract OCR was used as an independent cross-check. It produced 156 disagreements, almost all OCR dropping decimal points. All 27 disputed cells were re-checked on zoomed crops, and the visual reading was confirmed in every case.

**Columns:**
- `item`
- `food_type`: T Thermostabilized, R Rehydratable, I Irradiated, IM Intermediate Moisture, NF Natural Form, B Beverage, FF Fresh Food
- `food_type_name`
- `secondary_code_raw`: the second code printed under each item. Russian-language food-type abbreviations (e.g. НФ, Н, СБ) or packaging codes (T/O, C/P, C/O) as printed; the meaning of the packaging codes is not defined in the source.
- `mass_g`, `kcal`, `protein_g`, `carb_g`, `fat_g`, `sat_fat_g`, `fiber_g` (g)
- `sodium_mg`, `potassium_mg`, `magnesium_mg`, `iron_mg`, `zinc_mg`, `calcium_mg`, `phosphorus_mg` (mg)
- `source_page`
- `data_quality_notes`

**Caveats:**
- `mass_g` is the **packaged mass**. For R and B items this is dry or powder mass, not the rehydrated mass.
- Blank cells mean blank in the source.
- Rows flagged in `data_quality_notes`:
  - Caribbean Chicken: no mass, all zeros
  - Milk: no mass, only Na
  - Drinking Water Container: placeholder row
  - Chicken in Pouches and Maple Muffin Top: all minerals 0, likely missing rather than true zero
  - Lemon Meringue Pudding and Tuna Salad Spread: blank minerals
  - Cream coffees and teas: blank saturated fat
  - Grits w/ Butter: iron printed as 13 mg (verified; possibly a source typo for 1.3)
  - Three rows where kcal and Atwater (4P+4C+9F) disagree by more than 35%
- The source lists 186 items; NASA describes the standard menu as "about 200".
- **No vitamins** in this table. Use `usda_vitamins.csv` for vitamins.

## 2. micronutrient_requirements.csv
**Columns:** nutrient, unit, group (life-stage or `nasa_spaceflight all crew`), rda_or_ai, ul, source, value_type, target_max, differs_from_dri, notes

**Sources:**
- NASEM DRI Summary Tables (Appendix J, 2019): https://www.nationalacademies.org/read/25353/chapter/28
- NASA HIDH Rev 1 Tables 7.2-2/7.2-3: https://www.nasa.gov/wp-content/uploads/2023/03/human-integration-design-handbook-revision-1.pdf

The current NASA-STD-3001 Rev E applies DRIs except vitamin D (25 µg/d). See `scripts/micronutrient_requirements_README.md`.

## 3. energy_rules.json
**Contents:**
- EER, men: `662 − 9.53·age + AF·(15.9·kg + 539.6·m)`
- EER, women: `354 − 6.91·age + AF·(9.36·kg + 726·m)`
- Activity factor AF = 1.25 nominal (range 1.0–1.25 per JSC 67378)
- Default provision 3,035 kcal/crewmember/day
- EVA +200 kcal (837 kJ) per EVA hour
- Suited-nutrition rule (in-suit nutrition for EVAs > 4 h)
- HIDH macro guidelines (protein 0.8 g/kg and ≤35% of energy; carbohydrate 50–55%; fat 25–35%; saturated fat <7%; fiber 10–14 g per 1,000 kcal; etc.)
- DRI AMDRs
- Historical ISS intakes (OCHMO-TB-013)

**Sources:**
- NASA-STD-3001 Vol 2 Rev E, Table 7.1-1, [V2 7003]/[V2 7004]/[V2 7100]/[V2 11025]: https://ntrs.nasa.gov/api/citations/20250004555/downloads/NASA-STD-3001%20Vol%202%20Rev%20E%20FINAL_05022025.pdf
- OCHMO-TB-013: https://www.nasa.gov/wp-content/uploads/2023/12/ochmo-tb-013-food-and-nutrition.pdf
- HIDH Rev 1
- NASEM AMDR: https://www.nationalacademies.org/read/27957/chapter/5

**Caveat:** HIDH prints the men's intercept as 622, while Rev E and the DRI print 662. Use 662.

## 4. shelf_life_decay.csv
Long format, one row per record. `record_type` is one of:

| record_type | Meaning |
|---|---|
| `shelf_life` | Max shelf life by food type (Hurdle 2022 slides; NTRS 2025 gives 3 yr for the standard menu). The B row is an **assumption** (freeze-dried bound); IM and FF are gaps. |
| `decay_item` | Vitamin C per item at 0/1/3 yr, 21 °C, **digitized** from npj Microgravity 2017 Fig. 2 (675-px figure, about ±3–5 mg) |
| `decay_group` | Group losses from the paper text (B6: 14.5% average, chicken 26%, beef 22%; vitamin C in fruit 32–83%) and Fig. 3 (B1 median: breads about 21%, meats about 70%) |
| `decay_system` | Whole-menu % RDI delivered at 0/1/3 yr, pixel-measured from Fig. 1 (A, B1, B6, B12, C, D, K, Ca, K) |
| `decay_bound` | Qualitative or bound results (Hurdle 2022: thiamin ≥85% at 44 months, folic acid undetectable at 44 months, strawberry nutrients <85% at 43 months, niacin and B6 stable; Food Fortification Stability study: thiamin lost rapidly at 35 °C) |
| `fallback_default` | **Derived** default k per vitamin and category for user-entered or unmapped foods |

**Derivation:** first-order decay, `C(t) = C0·exp(−k·t)`, so `k = −ln(C_t/C0)/t` (per year) and `half_life = ln2/k`. Negative k (vitamin increases in fortified items, which the paper attributes to encapsulation) is clipped to 0 in the fallbacks.

**How to use:** pick the most specific `fallback_default` row for the food's category and vitamin; otherwise use the "any other food" or "any food" row.

**Gaps:** B2, E and folate have no quantitative k (placeholder 0 or blank, flagged). There is no fresh-food decay data. All decay data are at 21 °C; no temperature (Arrhenius) model is sourced.

**Sources:**
- Cooper, Douglas & Perchonok 2017, npj Microgravity 3:17: https://www.nature.com/articles/s41526-017-0022-z
- NASA Hurdle Approach 2022: https://ntrs.nasa.gov/api/citations/20220001872/downloads/Hurdle%20Approach_2022_FINAL_%20No%20audio.pdf
- Food Fortification Stability Study: https://ntrs.nasa.gov/citations/20160001048

## 5. usda_vitamins.csv
Each menu item is hand-matched to a USDA FoodData Central entry, using bulk CSVs (no API limits):
- SR Legacy 2018-04: 114 matches
- FNDDS (FDC release 2024-10-31): 69 matches
- 3 items not matched (Caribbean Chicken, Milk, Drinking Water Container)

**Match confidence:** 81 high, 67 medium, 35 low, 3 none. The reason for each match is in `match_notes` (e.g. "cheese sauce not represented").

**Columns:**
- `fdc_id`, `fdc_description`, `fdc_dataset`, `match_confidence`, `match_notes`, `usda_kcal_per_100g`
- `{vit}_per100g`, **sourced** from USDA. Vitamins: vit_a_rae_ug, vit_c_mg, vit_d_ug, vit_e_mg, vit_k_ug, thiamin_b1_mg, riboflavin_b2_mg, niacin_b3_mg, vit_b6_mg, vit_b12_ug, folate_dfe_ug
- `scaling_basis`, `serving_equiv_g`, `{vit}_per_serving_est`: **DERIVED**
  - mass basis: menu mass_g when the USDA form matches the packaged form (T/I/NF/IM, or dry powders)
  - energy basis: menu kcal ÷ USDA kcal/100 g when the ISS item is dry and USDA is as-eaten
  - 1 item not scalable

**Caveats:**
- These are generic US foods, not NASA formulations. Fortification differs, especially for drinks and cereals.
- 36 vitamin cells are blank in USDA itself (mostly vitamins D, K and E in SR Legacy).
- Live endpoint for new user foods: `https://api.nal.usda.gov/fdc/v1/foods/search?api_key=DEMO_KEY&query=...` (DEMO_KEY is heavily rate-limited; get a free key).
- Bulk downloads: https://fdc.nal.usda.gov/download-datasets

## 6. launch_library/
**Source:** The Space Devs Launch Library 2, v2.3.0 (`https://ll.thespacedevs.com/2.3.0/...`). Fetched 2026-10-03. Free tier is 15 requests/hour; 9 were used.

| File | Contents |
|---|---|
| `spacewalks.json` | 100 most recent of 500 (ordering=-start). 71 ISS, 27 Tiangong. Covers 2018-08 to 2026-09-01 (Expedition 75 EVA 4). |
| `docking_events.json` | 100 of 576. 73 ISS. Covers 2021-08 to 2026-10-01. |
| `expeditions.json` | 100 of 165 (ISS, Tiangong, Mir). |
| `upcoming_launches_all_raw.json` | First 100 of 470 upcoming launches (unfiltered). |
| `upcoming_launches.json` | 14 filtered ISS flights. Fields: name, net_utc, net_precision, status, provider, mission_type, orbit, mission_description, category (`cargo_resupply` or `crew_rotation_or_private_crew`), source_search. |

**Caveats:**
- Filter by `spacestation.name == "International Space Station"` for ISS-only views.
- Many future dates are placeholders. Check `net_precision` (Month/Quarter/Year).
- Times are UTC. Convert for display.

## 7. crew_timeline_template.json
**Source:** ISS Form 24 for 10/03/2014 (Radiogram 6767u): https://www.nasa.gov/wp-content/uploads/2014/09/100314_tl.pdf (times as printed, GMT)

**Contents:**
- `events`: 109 events with start, end, crew, activity, category and duration_min. Category is **derived** by keyword.
- `per_crew_minutes_by_category`, `per_crew_meals`, `per_crew_exercise`
- `derived_template`:
  - Blocks: sleep 21:30–06:00 (510 min), breakfast 06:40–07:30, lunch 13:00–14:00, dinner inside the 19:30–21:30 pre-sleep block (source note 2)
  - Exercise: about 145–150 min per crewmember that day, in 2 sessions on ARED, CEVIS, BD-2 or VELO

**Caveat:** This is one Russian-segment form (US activities are in OSTPV), so USOS work time is undercounted.

## 8. food_system_structure.json
**Source:** NTRS 2025 "Space Food System Overview" (X. Wu): https://ntrs.nasa.gov/api/citations/20250004456/downloads/ISS%20Food%20System%20Overview%20-%20ION%20052025.pdf

**Sourced contents:**
- Container composition: standard menu about 75%, specialty about 25% (crew-specific menu, coffee/tea preference, shelf-stable kit, fresh food kit, cold stowage, holiday BOB, science containers), others about 1%
- Standard menu: about 200 items, 3-year shelf life, inventory tracked per category at container level
- Category list: the slide says "10 categories" but lists 11
- Food types, mission phases, and future 5-year shelf-life needs

**Derived contents** (`*_DERIVED`): counts by food type, and a keyword-based category assignment for the 186 menu items.

---
## Not included / known gaps
- **No live ISS food inventory.** NASA does not publish one. Simulate it from container composition, menu items and resupply flights.
- No vitamin values in the NASA menu source; vitamins come from generic USDA matches.
- No decay kinetics for B2, E or folate; none for fresh food; no temperature dependence.
- Launch Library caches are snapshots. Refresh within the 15 requests/hour limit.

## Python loader (pandas)
```python
import json, pandas as pd
from pathlib import Path
D = Path("data")  # adjust to where you unzipped

menu  = pd.read_csv(D/"iss_menu.csv")
vits  = pd.read_csv(D/"usda_vitamins.csv")
reqs  = pd.read_csv(D/"micronutrient_requirements.csv")
decay = pd.read_csv(D/"shelf_life_decay.csv")
energy   = json.loads((D/"energy_rules.json").read_text())
timeline = json.loads((D/"crew_timeline_template.json").read_text())
foodsys  = json.loads((D/"food_system_structure.json").read_text())
ll = {n: json.loads((D/"launch_library"/f"{n}.json").read_text())
      for n in ["spacewalks", "docking_events", "expeditions", "upcoming_launches"]}

# menu + vitamins in one table
foods = menu.merge(vits.drop(columns=["food_type"]), on="item", how="left")

# EER for a crewmember (+ EVA hours)
def eer(sex, age, kg, m, af=energy["eer_equations"]["activity_factor"]["nominal"], eva_h=0):
    e = energy["eer_equations"]["male_19plus" if sex == "M" else "female_19plus"]
    base = e["intercept"] + e["age_coef"]*age + af*(e["mass_coef"]*kg + e["height_coef"]*m)
    return base + eva_h*energy["eva_energy"]["extra_kcal_per_eva_hour"]

# vitamin remaining after t years (fallback k for any food)
fb = decay[decay.record_type == "fallback_default"]
def remaining(c0, vitamin, t_years, category="any food"):
    rows = fb[(fb.vitamin == vitamin) & (fb.category.str.contains(category, regex=False))]
    if rows.empty: rows = fb[(fb.vitamin == vitamin)]
    k = pd.to_numeric(rows.k_per_year, errors="coerce").dropna()
    import math
    return c0 * math.exp(-k.iloc[0]*t_years) if len(k) else None

iss_upcoming = pd.json_normalize(ll["upcoming_launches"]["results"])
print(eer("M", 45, 82.9, 1.80, eva_h=6.5), remaining(100, "vitamin_c", 1.5, "fruit"))
```
