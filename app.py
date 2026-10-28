import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date, timedelta

st.set_page_config(page_title="PricePilot | Restaurant Pricing Simulator", page_icon="🍔", layout="wide")

# -----------------------------
# Synthetic data generation
# -----------------------------
@st.cache_data
def make_synthetic_data(seed=42, days=180):
    """Create explicitly synthetic daily store-product observations."""
    rng = np.random.default_rng(seed)
    stores = [
        {"store_id": "S01", "store_name": "Central Office", "location_type": "Office district", "base_demand": 115, "elasticity": -1.25, "demand_index": 1.08},
        {"store_id": "S02", "store_name": "North Residential", "location_type": "Residential", "base_demand": 92, "elasticity": -1.85, "demand_index": 0.98},
        {"store_id": "S03", "store_name": "Mall Food Court", "location_type": "Mall", "base_demand": 130, "elasticity": -1.10, "demand_index": 1.12},
        {"store_id": "S04", "store_name": "University Road", "location_type": "Student area", "base_demand": 105, "elasticity": -2.10, "demand_index": 1.00},
        {"store_id": "S05", "store_name": "High Street", "location_type": "Mixed retail", "base_demand": 100, "elasticity": -1.50, "demand_index": 1.04},
    ]
    products = [
        {"product_id": "P01", "product_name": "Signature Burger", "current_price": 200.0, "unit_cost": 90.0, "competitor_price": 205.0},
        {"product_id": "P02", "product_name": "Cold Coffee", "current_price": 140.0, "unit_cost": 48.0, "competitor_price": 145.0},
    ]
    end_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    dates = pd.date_range(end=end_date, periods=days, freq="D")
    rows = []
    for store in stores:
        for product in products:
            store_factor = store["base_demand"] / 100
            if product["product_id"] == "P02":
                product_factor = 1.25
            else:
                product_factor = 1.0
            base_price = product["current_price"]
            base_demand = store["base_demand"] * product_factor
            for d in dates:
                dow = d.dayofweek
                # Illustrative weekly demand patterns
                dow_factor = {0: 0.96, 1: 0.98, 2: 1.00, 3: 1.02, 4: 1.12, 5: 1.18, 6: 1.08}[dow]
                demand_index = max(0.75, rng.normal(store["demand_index"], 0.08))
                promotion_flag = int(rng.random() < 0.16)
                list_price = base_price * rng.uniform(0.94, 1.06)
                discount = (rng.choice([5, 10, 15]) if promotion_flag else 0.0)
                net_price = max(1.0, list_price - discount)
                competitor_price = product["competitor_price"] * rng.uniform(0.97, 1.03)
                # Demand-generating process: price, promotion, local demand, weekday and noise
                expected = (
                    base_demand
                    * (net_price / base_price) ** store["elasticity"]
                    * (1.18 if promotion_flag else 1.0)
                    * demand_index
                    * dow_factor
                    * np.exp(rng.normal(0, 0.10))
                )
                orders = max(0, int(round(expected)))
                rows.append({
                    "date": d.date().isoformat(),
                    "store_id": store["store_id"],
                    "store_name": store["store_name"],
                    "location_type": store["location_type"],
                    "product_id": product["product_id"],
                    "product_name": product["product_name"],
                    "list_price": round(float(list_price), 2),
                    "discount": round(float(discount), 2),
                    "net_price": round(float(net_price), 2),
                    "orders": orders,
                    "unit_cost": product["unit_cost"],
                    "competitor_price": round(float(competitor_price), 2),
                    "promotion_flag": promotion_flag,
                    "demand_index": round(float(demand_index), 3),
                    "day_of_week": d.strftime("%A"),
                })
    return pd.DataFrame(rows), stores, products

df, store_master, product_master = make_synthetic_data()

# -----------------------------
# Helper functions
# -----------------------------
def estimate_elasticity(data):
    """OLS log-log regression with promotion, log demand index and weekday controls."""
    d = data.copy()
    d = d[(d["orders"] > 0) & (d["net_price"] > 0) & (d["demand_index"] > 0)].copy()
    if len(d) < 30 or d["net_price"].nunique() < 5:
        return None, None, len(d), "Insufficient usable observations or price variation."
    y = np.log(d["orders"].astype(float).to_numpy())
    log_price = np.log(d["net_price"].astype(float).to_numpy())
    log_demand = np.log(d["demand_index"].astype(float).to_numpy())
    promo = d["promotion_flag"].astype(float).to_numpy()
    dow = pd.get_dummies(d["day_of_week"], drop_first=True, dtype=float)
    X = np.column_stack([
        np.ones(len(d)), log_price, promo, log_demand, dow.to_numpy()
    ])
    try:
        beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ beta
        dof = max(1, len(y) - X.shape[1])
        sigma2 = float((residuals @ residuals) / dof)
        cov = sigma2 * np.linalg.pinv(X.T @ X)
        se = float(np.sqrt(max(cov[1, 1], 0)))
        elasticity = float(beta[1])
        # Flag unreliable / implausible estimates rather than using them blindly
        if not np.isfinite(elasticity) or elasticity >= -0.05 or elasticity < -5:
            return elasticity, se, len(d), "Estimate is implausible for this prototype; use assumed elasticity or investigate data."
        return elasticity, se, len(d), "OLS estimate; observational association, not guaranteed causal."
    except Exception as exc:
        return None, None, len(d), f"Model could not be estimated: {exc}"

def simulate_scenarios(baseline_orders, baseline_price, unit_cost, elasticity, candidate_prices,
                       competitor_price, max_demand_loss_pct, min_margin_pct, max_comp_premium_pct,
                       use_comp_guardrail=True):
    rows = []
    q0 = max(float(baseline_orders), 0.01)
    p0 = max(float(baseline_price), 0.01)
    for price in sorted(set([round(float(p), 2) for p in candidate_prices])):
        expected_orders = q0 * (price / p0) ** float(elasticity)
        revenue = expected_orders * price
        contribution_per_order = price - unit_cost
        contribution = expected_orders * contribution_per_order
        margin = contribution_per_order / price if price > 0 else -1
        demand_change_pct = (expected_orders / q0 - 1) * 100
        comp_premium_pct = ((price / competitor_price) - 1) * 100 if competitor_price > 0 else np.nan
        reasons = []
        if demand_change_pct < -max_demand_loss_pct:
            reasons.append("Demand loss limit")
        if margin * 100 < min_margin_pct:
            reasons.append("Minimum margin")
        if use_comp_guardrail and comp_premium_pct > max_comp_premium_pct:
            reasons.append("Competitor premium")
        if price <= unit_cost:
            reasons.append("Non-positive contribution/order")
        rows.append({
            "Candidate price (₹)": price,
            "Expected orders/day": expected_orders,
            "Revenue/day (₹)": revenue,
            "Contribution/order (₹)": contribution_per_order,
            "Contribution/day (₹)": contribution,
            "Contribution margin (%)": margin * 100,
            "Order change (%)": demand_change_pct,
            "Competitor premium (%)": comp_premium_pct,
            "Feasible": len(reasons) == 0,
            "Constraint notes": "Pass" if not reasons else ", ".join(reasons),
        })
    return pd.DataFrame(rows)

# -----------------------------
# Interface
# -----------------------------
st.title("🍔 PricePilot")
st.subheader("Store-level pricing decision simulator")
st.caption("A working prototype using synthetic restaurant data. All figures and recommendations are illustrative, not actual company results.")

with st.sidebar:
    st.header("Decision inputs")
    st.markdown("### 1. Select the decision unit")
    store_options = {f'{s["store_id"]} — {s["store_name"]} ({s["location_type"]})': s["store_id"] for s in store_master}
    selected_store_label = st.selectbox("Store", list(store_options.keys()), index=0)
    selected_store = store_options[selected_store_label]
    product_options = {f'{p["product_id"]} — {p["product_name"]}': p["product_id"] for p in product_master}
    selected_product_label = st.selectbox("Product", list(product_options.keys()), index=0)
    selected_product = product_options[selected_product_label]

    st.markdown("### 2. Demand model")
    elasticity_mode = st.radio("Elasticity source", ["Estimate from synthetic history", "Assume elasticity (sensitivity)"])
    assumed_elasticity = st.slider("Assumed price elasticity", min_value=-4.0, max_value=-0.2, value=-1.5, step=0.1,
                                   help="Negative values indicate that higher prices reduce demand.")
    lookback_days = st.slider("Baseline lookback (days)", 14, 90, 28, 7)

    st.markdown("### 3. Guardrails")
    max_price_change_pct = st.slider("Maximum price movement (%)", 2, 20, 10, 1)
    max_demand_loss_pct = st.slider("Maximum forecast demand loss (%)", 0, 40, 10, 1)
    min_margin_pct = st.slider("Minimum contribution margin (%)", 0, 80, 40, 1)
    max_comp_premium_pct = st.slider("Maximum premium vs competitor (%)", 0, 30, 5, 1)
    use_comp_guardrail = st.checkbox("Apply competitor premium guardrail", value=True)
    price_step = st.selectbox("Candidate price step (₹)", [1, 2, 5, 10], index=2)

filtered = df[(df["store_id"] == selected_store) & (df["product_id"] == selected_product)].copy()
filtered["date"] = pd.to_datetime(filtered["date"])
filtered = filtered.sort_values("date")
current_price = float(product_master[[p["product_id"] for p in product_master].index(selected_product)]["current_price"])
unit_cost = float(product_master[[p["product_id"] for p in product_master].index(selected_product)]["unit_cost"])
competitor_master = float(product_master[[p["product_id"] for p in product_master].index(selected_product)]["competitor_price"])
recent = filtered[filtered["date"] >= (filtered["date"].max() - pd.Timedelta(days=lookback_days-1))]
baseline_orders = float(recent["orders"].mean()) if len(recent) else float(filtered["orders"].mean())
historical_price = float(recent["net_price"].mean()) if len(recent) else current_price
competitor_price = float(recent["competitor_price"].mean()) if len(recent) else competitor_master
est_elasticity, elasticity_se, n_obs, elasticity_note = estimate_elasticity(filtered)
if elasticity_mode == "Estimate from synthetic history" and est_elasticity is not None and -5 < est_elasticity < -0.05:
    selected_elasticity = est_elasticity
    elasticity_label = "Estimated from historical synthetic data"
else:
    selected_elasticity = assumed_elasticity
    elasticity_label = "User-assumed sensitivity scenario"
    if elasticity_mode == "Estimate from synthetic history":
        elasticity_note = "Historical estimate was unavailable or implausible; using the selected assumed elasticity."

# Use the current master price as the baseline price for scenario decisions.
baseline_price = current_price
min_price = baseline_price * (1 - max_price_change_pct / 100)
max_price = baseline_price * (1 + max_price_change_pct / 100)
candidate_prices = np.arange(
    math_floor := np.floor(min_price / price_step) * price_step,
    np.ceil(max_price / price_step) * price_step + price_step * 0.5,
    price_step
)
candidate_prices = [p for p in candidate_prices if p > 0 and min_price - 1e-6 <= p <= max_price + 1e-6]
candidate_prices.append(baseline_price)

scenarios = simulate_scenarios(
    baseline_orders=baseline_orders,
    baseline_price=baseline_price,
    unit_cost=unit_cost,
    elasticity=selected_elasticity,
    candidate_prices=candidate_prices,
    competitor_price=competitor_price,
    max_demand_loss_pct=max_demand_loss_pct,
    min_margin_pct=min_margin_pct,
    max_comp_premium_pct=max_comp_premium_pct,
    use_comp_guardrail=use_comp_guardrail,
)
current_row = scenarios.iloc[(scenarios["Candidate price (₹)"] - baseline_price).abs().argsort()[:1]].iloc[0]
feasible = scenarios[scenarios["Feasible"]].copy()
recommendation = feasible.sort_values("Contribution/day (₹)", ascending=False).iloc[0] if not feasible.empty else None

# Top line
st.info("Decision unit: **one store × one product**. Optimize contribution while monitoring revenue, order volume, margin and competitive position.")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Baseline orders/day", f"{baseline_orders:,.1f}")
k2.metric("Current menu price", f"₹{baseline_price:,.0f}")
k3.metric("Variable cost/order", f"₹{unit_cost:,.0f}")
k4.metric("Elasticity used", f"{selected_elasticity:.2f}", help=elasticity_label)

tab1, tab2, tab3, tab4 = st.tabs(["Recommendation", "Scenario comparison", "Data & model", "Pilot plan"])

with tab1:
    if recommendation is None:
        st.error("No candidate price satisfies all active guardrails. Keep the current price temporarily or review the constraints and data.")
    else:
        rec_price = float(recommendation["Candidate price (₹)"])
        current_contrib = float(current_row["Contribution/day (₹)"])
        rec_contrib = float(recommendation["Contribution/day (₹)"])
        delta = rec_contrib - current_contrib
        delta_pct = (delta / current_contrib * 100) if current_contrib else np.nan
        order_delta = float(recommendation["Order change (%)"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Recommended test price", f"₹{rec_price:,.0f}", f"{rec_price-baseline_price:+.0f} vs current")
        c2.metric("Expected contribution/day", f"₹{rec_contrib:,.0f}", f"{delta:+,.0f} ({delta_pct:+.1f}%) vs current")
        c3.metric("Expected order change", f"{order_delta:+.1f}%", help="Modelled change versus recent baseline volume.")
        st.markdown("#### Why this price?")
        if abs(rec_price - baseline_price) < 0.01:
            st.write("The current price is the best feasible candidate under the selected assumptions and guardrails.")
        else:
            st.write(f"The model selects **₹{rec_price:,.0f}** because it has the highest forecast contribution among candidates that pass the active guardrails.")
        st.write(f"- Estimated/assumed elasticity: **{selected_elasticity:.2f}** ({elasticity_label}).")
        st.write(f"- Forecast orders: **{recommendation['Expected orders/day']:.1f}/day**, compared with the baseline of {baseline_orders:.1f}/day.")
        st.write(f"- Contribution margin: **{recommendation['Contribution margin (%)']:.1f}%**.")
        st.write(f"- Competitor price reference: **₹{competitor_price:.0f}**; forecast premium: **{recommendation['Competitor premium (%)']:.1f}%**.")
        st.warning("This is a modelled scenario using synthetic data, not a causal forecast. Pilot the recommendation before rollout.")
    st.markdown("#### Balanced scorecard")
    if recommendation is not None:
        scorecard = pd.DataFrame([
            {"Dimension": "Financial", "Metric": "Contribution/day", "Result": f"₹{recommendation['Contribution/day (₹)']:,.0f}", "Role": "Primary objective"},
            {"Dimension": "Commercial", "Metric": "Revenue/day", "Result": f"₹{recommendation['Revenue/day (₹)']:,.0f}", "Role": "Monitor"},
            {"Dimension": "Commercial", "Metric": "Expected orders/day", "Result": f"{recommendation['Expected orders/day']:,.1f}", "Role": "Guardrail"},
            {"Dimension": "Unit economics", "Metric": "Contribution margin", "Result": f"{recommendation['Contribution margin (%)']:.1f}%", "Role": "Guardrail"},
            {"Dimension": "Competitive", "Metric": "Premium vs competitor", "Result": f"{recommendation['Competitor premium (%)']:.1f}%", "Role": "Guardrail"},
            {"Dimension": "Customer", "Metric": "Repeat purchase / complaints", "Result": "Not available", "Role": "Monitor in real pilot"},
        ])
        st.dataframe(scorecard, use_container_width=True, hide_index=True)

with tab2:
    st.markdown("### Candidate price comparison")
    plot_df = scenarios.copy()
    fig = px.line(plot_df, x="Candidate price (₹)", y="Contribution/day (₹)", markers=True,
                  title="Expected daily contribution by candidate price")
    if recommendation is not None:
        fig.add_vline(x=float(recommendation["Candidate price (₹)"]), line_dash="dash",
                      annotation_text="Recommended", annotation_position="top")
    st.plotly_chart(fig, use_container_width=True)
    fig2 = px.line(plot_df, x="Candidate price (₹)", y="Expected orders/day", markers=True,
                   title="Expected daily orders by candidate price")
    st.plotly_chart(fig2, use_container_width=True)
    display_cols = ["Candidate price (₹)", "Expected orders/day", "Revenue/day (₹)", "Contribution/day (₹)",
                    "Contribution margin (%)", "Order change (%)", "Competitor premium (%)", "Feasible", "Constraint notes"]
    st.dataframe(scenarios[display_cols].round(2), use_container_width=True, hide_index=True)
    st.download_button("Download scenario comparison CSV", scenarios.to_csv(index=False).encode("utf-8"),
                       file_name="pricepilot_scenarios.csv", mime="text/csv")

with tab3:
    st.markdown("### Historical data for selected store × product")
    st.caption("Synthetic data generated inside this app. The price, order and competitor observations are not real company records.")
    st.write(f"Usable observations for elasticity estimate: **{n_obs}**")
    st.write(f"Elasticity status: {elasticity_note}")
    if est_elasticity is not None:
        st.write(f"Regression elasticity estimate: **{est_elasticity:.2f}**; approximate standard error: **{elasticity_se:.2f}**.")
    st.write(f"Elasticity used in the simulator: **{selected_elasticity:.2f}**.")
    st.write("The regression controls for promotion, log demand index and weekday. It remains observational and may be biased by omitted variables or price-setting decisions.")
    hist = filtered[["date", "net_price", "orders", "discount", "promotion_flag", "competitor_price", "demand_index"]].copy()
    st.dataframe(hist.tail(30), use_container_width=True, hide_index=True)
    st.download_button("Download all synthetic historical data", df.to_csv(index=False).encode("utf-8"),
                       file_name="pricepilot_synthetic_history.csv", mime="text/csv")
    st.markdown("#### Assumptions")
    st.markdown("""
    - All observations are synthetic and generated with a known illustrative demand process.
    - Candidate prices are interpreted as realized net prices; historical discounts are already reflected in historical net price.
    - Variable cost is assumed constant per order and does not include unprovided platform commissions, packaging, labor or fixed costs.
    - Fractional expected orders represent average forecasts, not literal partial orders.
    - Guardrail values are configurable demo settings, not company policies.
    """)

with tab4:
    st.markdown("### Suggested controlled pilot")
    st.markdown("""
    1. Choose eligible, comparable stores or randomized customer groups.
    2. Keep the current price in the control group and test the recommended price in the treatment group.
    3. Run the pilot long enough to cover normal weekday/weekend patterns; set duration and sample size based on variability and the minimum effect worth detecting.
    4. Compare contribution per store-day, orders, revenue, conversion, repeat purchase and complaints where available.
    5. Scale only if actual contribution improves by a commercially meaningful amount without unacceptable customer or competitive effects.
    """)
    st.markdown("**Primary outcome:** difference in contribution per store-day between treatment and control.")
    st.markdown("**Secondary outcomes:** orders, revenue, conversion, average transaction value, repeat purchase and complaints.")
    st.warning("If stores are not randomized, use matched comparisons cautiously and account for baseline differences, seasonality and possible customer switching.")

st.divider()
st.caption("PricePilot prototype • Synthetic data • Decision support only • Do not deploy prices automatically from this prototype.")
