# ISS mission food load

Streamlit planner that packs the ISS standard menu for a crew. The app calls `mission.mission_food`, which solves a minimum-mass typical day and an EVA day, blends them by the EVA-day fraction, and adds a safety margin on servings.

Nutrient targets and the menu linear program stay in `targets.py`, `optimizer.py`, and `mission.py`. Source notes for the food tables live in [data/README.md](data/README.md).

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.port 8765
```

## Sidebar

- Mission length in days
- Number of crew, and for each person age, sex, weight (kg), height (m), and allergies (comma-separated)
- EVA hours per week for the mission (one number, not per person)
- Safety margin, default 10% (`0.10`)

EVA hours per week are converted before the `mission_food` call. Zero hours plans typical days only (`eva_day_fraction=0`, `eva_hours=0`). Up to 45.5 hours a week (seven standard 6.5-hour EVA days), `eva_day_fraction` is weekly hours / 45.5 and `eva_hours` stays 6.5. Above 45.5, every day is an EVA day and `eva_hours` is the weekly total divided by 7.

The page shows total food mass, the packing list (heaviest first), a chart of the mission-average day as a percent of each minimum target, and a CSV download named `mission_packing_list.csv`.

## Tests

```bash
python -m pytest
```
