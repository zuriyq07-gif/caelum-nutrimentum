## Inspiration

A long mission launches with a fixed food mass. Vitamins decay, so a menu that meets day-0 targets can miss them later. Caelum Nutrimentum packs the least mass that still covers the crew, using real NASA data.

## What it does

We set mission length, a simulated crew, weekly EVA hours, and a 10% safety margin. A linear programming optimizer, `scipy.optimize.linprog`, minimizes packaged mass for a typical day and an EVA day, with nutrient minimums and upper limits, a sodium cap, at most two servings per person per day, and allergy exclusion. An EVA-day fraction blends those days over the mission length, then the margin scales them. Vitamin decay modeling uses `C(t)=C0*exp(-k*t)`, with yearly `k` converted by `t/365.25`. We replan every 30 days. Charts show each vitamin against its target and the locked day-0 shortfall days. Vitamin A and vitamin D go short first on the default crew. Resupply delay lengthens the packed horizon and ages stored food.

## How we built it

We used the NASA STEMonstrations ISS menu, NASA-STD-3001 and HIDH energy rules, NASEM micronutrients, USDA FoodData Central vitamins, and Cooper, Douglas, and Perchonok decay rates. Spacewalks and launches are real in the data notes, but the spacewalks file is not shipped, so the EVA fraction is passed in from weekly hours. Inventory and crew profiles are simulated. Grok Voice is push-to-talk: speech-to-text, tool calls on the real planner for mission length, crew, EVA hours, resupply delay, and nutrient shortfalls, then text-to-speech. The assistant summarizes the change in total mass and the packing list. The browser never sees the API key. The stack is Python, Streamlit, SciPy, the Grok API, and Cursor.

## Challenges we ran into

Nutrient units cannot share one axis, so bars are percents of each minimum. Yearly `k` is scaled by `t/365.25` to match mission days. Upper limits that apply only to supplements are not food caps. An infeasible menu names the missing nutrient. Voice tools use the same optimizer as the sidebar, so the spoken mass matches the screen.

## Accomplishments we're proud of

The packing list, decay charts, and voice tools share one planner.

## What we learned

Meeting day-0 targets does not keep later days covered.

## What's next

We ship a Launch Library EVA cache, and add a key-free demo path for the planner.
