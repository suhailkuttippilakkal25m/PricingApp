# PricePilot — Restaurant Pricing Decision Simulator

## What this is
A beginner-friendly Streamlit prototype for comparing restaurant prices at the **store × product** level. It estimates price elasticity from synthetic history (or lets the user set an assumed elasticity), calculates expected orders/revenue/contribution, applies guardrails, and recommends the highest-contribution feasible price.

**All data and resulting recommendations are synthetic and illustrative.** This is not a real restaurant chain's data and does not establish causal customer response.

## What is included
- `app.py` — working Streamlit application
- `requirements.txt` — Python packages needed
- `presentation.pptx` — five-slide hiring-case presentation
- `presentation_script.md` — 90-second speaking script
- `decision_log.md` — model choices and assumptions
- `test_checklist.md` — manual test plan

## Easiest way to run on a Mac
1. Download and unzip the `PricePilot_Submission.zip` package.
2. Install Python 3.11 or newer if it is not already installed. Get it from https://www.python.org/downloads/macos/
3. Open **Terminal**.
4. Move into the unzipped folder. If it is in Downloads, type:
   `cd ~/Downloads/pricepilot`
   (If you unzipped it somewhere else, use that folder's path.)
5. Create a separate environment:
   `python3 -m venv .venv`
6. Activate it:
   `source .venv/bin/activate`
7. Install the packages:
   `python -m pip install --upgrade pip`
   `pip install -r requirements.txt`
8. Start the app:
   `streamlit run app.py`
9. Your browser should open the app. If not, copy the local URL shown in Terminal (usually `http://localhost:8501`) into your browser.

To stop the app, return to Terminal and press `Control + C`.

## How to demo it
1. Select **S01 — Central Office** and **Signature Burger**.
2. Start with the default settings and note the current price, baseline orders, elasticity and recommendation.
3. Open **Scenario comparison** and compare contribution with expected orders.
4. Switch to **S04 — University Road** or another store and note that the recommendation can change because the underlying synthetic demand sensitivity differs.
5. Change the demand-loss limit or the elasticity mode to demonstrate how assumptions affect the decision.
6. Open **Data & model** to show that the dataset is synthetic and download the CSV if useful.
7. Finish with the **Pilot plan** tab.

## Important definitions
- Net price: price actually realized after discounts.
- Revenue: net price × expected orders.
- Contribution/order: net price − variable cost/order.
- Total contribution: expected orders × contribution/order.
- Contribution margin: contribution/order ÷ net price.
- Price elasticity: approximate percentage change in demand for a 1% price change, holding included model controls constant.
- Guardrails: conditions a price must satisfy to be considered feasible.

## Assumptions and limitations
- The dataset is generated inside the app; it is not public or company data.
- A synthetic log-log regression is estimated with promotion, demand-index and weekday controls.
- Historical observational relationships do not necessarily equal causal price effects.
- Costs are simplified to a constant variable cost per order. Packaging, delivery commissions, payment fees, labor and fixed costs are excluded unless added.
- Competitor prices are synthetic and are only a rough comparison.
- Repeat purchase, conversion, complaints and customer lifetime value are not included in the sample data; they should be measured in a real pilot.
- The 10% demand-loss, 40% margin and 5% competitor-premium defaults are illustrative choices, not industry standards.
- The app should be used for a hiring-case demonstration, not for automatic real-world price deployment.

## Troubleshooting
- If `python3` is not found, install Python and reopen Terminal.
- If `streamlit` is not found, make sure `.venv` is activated and run `pip install -r requirements.txt` again.
- If port 8501 is occupied, run `streamlit run app.py --server.port 8502`.
- If the app is not opening, use the `Local URL` printed in Terminal.
