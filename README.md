# ISS food ledger

A small Python project for the ISS standard menu. It loads the nutrition package in `data/`, joins each food to a USDA vitamin match, and drops or flags the rows called out in `data_quality_notes`.

The Streamlit app is a ledger, a data-quality report, a one-day menu built with SciPy’s HiGHS solver, and a shelf-life view that integrates first-order vitamin loss.

Source notes, citations, and caveats live in [data/README.md](data/README.md). `crew_timeline_template.json` and `food_system_structure.json` are included for the rest of that package. This loader reads the five analysis files below.

| File | What the loader does with it |
|---|---|
| `data/iss_menu.csv` | Macros and minerals per packaged serving |
| `data/usda_vitamins.csv` | Vitamins per 100 g and per serving, joined on `item` |
| `data/micronutrient_requirements.csv` | DRI life-stage rows and NASA spaceflight rows |
| `data/shelf_life_decay.csv` | Shelf-life bounds, measured loss, and fallback rates |
| `data/energy_rules.json` | Flattened to parameter, EER, and macro-guideline tables |

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python data_loader.py
streamlit run app.py --server.port 8765
```

`python data_loader.py` prints row counts, the dropped and flagged items, and missing values. The same text is on the app’s Data quality tab.

## Use the tables

```python
from data_loader import load_all
from nutrition import eer, remaining

pkg = load_all()
print(pkg.summary_text)
foods = pkg.foods  # macros, minerals, and vitamins per serving

print(eer("M", 45, 82.9, 1.80, eva_h=6.5, rules=pkg.energy_rules))
print(remaining(100, "vitamin_c", 1.5, "fruit", pkg.decay).amount)
```

Quality rules, taken from `data_quality_notes`:

- **Drop** a row when the note calls it a placeholder or says the packaged mass is blank. That removes Caribbean Chicken, Milk, and Drinking Water Container.
- **Flag** every other non-empty note and keep the row. That covers missing minerals, blank saturated fat, kcal values far from 4P+4C+9F, and the printed 13 mg iron on Grits w/ Butter.
- On rows noted as “likely missing data, not true zero,” mineral zeros are stored as missing so they are not summed as real zeros.

Vitamin D targets in the app follow NASA-STD-3001 (25 µg/day). Other micronutrients follow the DRI for the selected age and sex.

## Tests

```bash
python -m pytest
```
