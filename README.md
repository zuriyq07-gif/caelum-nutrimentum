# ISS mission food manifest

Streamlit planner that packs the ISS standard menu for a crew. The app calls `mission.mission_food`, which solves a minimum-mass typical day and an EVA day, blends them by the EVA-day fraction, and adds a safety margin on servings.

Nutrient targets and the menu linear program stay in `targets.py`, `optimizer.py`, and `mission.py`. Source notes for the food tables live in [data/README.md](data/README.md).

## Run

```bash
~/.local/bin/streamlit run app.py --server.address 0.0.0.0 --server.port 8765 --server.headless true
```

A virtualenv works too: install `requirements.txt`, then `streamlit run app.py --server.port 8765`.

## Sidebar

- Mission length in days (at least 1). Default 30.
- Number of crew (1–6). Default is one male, age 45, 82.9 kg, 180 cm, no allergies. Each person has age, sex, weight (kg), height (cm), and allergies (comma-separated).
- EVA hours per week for the mission (one number, not per person). Default 6.5.
- Safety margin in percent, default 10 (`0.10` on the call).

EVA hours per week are converted before the `mission_food` call, and that fraction is always passed in. `data/launch_library/spacewalks.json` is not used.

- 0 h/week: `eva_day_fraction` 0 and `eva_hours` 0. No EVA days.
- More than 0 and at most 6.5 h/week: one EVA day per week (`eva_day_fraction` 1/7) lasting the weekly hours.
- Above 6.5 h/week: `n = weekly_hours / 6.5`. When `n` is at most 7, `eva_day_fraction` is `n/7` and each EVA day is 6.5 h. When `n` is above 7, every day is an EVA day (`eva_day_fraction` 1) and `eva_hours` is the weekly total divided by 7.

The page shows total food mass, the packing list (heaviest first), a chart titled “Share of daily minimum, mission-weighted”, and a CSV download named `mission_packing_list.csv` with columns `item`, `servings`, and `mass_kg`.

`decay.py` replans that menu every 30 days as vitamins decay. Content on mission day `t` is `C0 * exp(-k_per_year * t / 365.25)`, using the yearly rates in `data/shelf_life_decay.csv`. Under the nutrient chart, a Shelf life section plots each vitamin against its minimum: the locked day-0 menu, the replanned menu, and the target. Days the locked menu would miss a minimum are marked, and a table lists serving changes at later epochs. The safety margin stays on the packing list and is not applied again inside those daily solves.

## Mission assistant

The assistant can change the mission, crew, EVA hours, and resupply delay, and it can report nutrient shortfalls. It uses the xAI API: `grok-4.7` chat completions with tools, and Grok Voice to hear a question and speak the answer. Set `XAI_API_KEY` in the environment, or `xai_api_key` in `.streamlit/secrets.toml` (that file is gitignored). Without a key, the manifest still packs and the panel says the assistant needs a key.

## Tests

```bash
python -m pytest
```
