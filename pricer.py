import streamlit as st
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import date, timedelta

# --- CLASS DEFINITION: The "Engine" ---
class BlackScholes:
    def __init__(self, S, K, T, r, sigma, option_type="call"):
        self.S = S           # Spot Price
        self.K = K           # Strike Price
        self.T = T           # Time to Maturity (in years)
        self.r = r           # Risk-free Interest Rate (decimal)
        self.sigma = sigma   # Volatility (decimal)
        self.type = option_type.lower()

    def _d1_d2(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    def price(self):
        d1, d2 = self._d1_d2()
        if self.type == "call":
            price = self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        return price

    def delta(self):
        d1, _ = self._d1_d2()
        if self.type == "call":
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1

    def gamma(self):
        d1, _ = self._d1_d2()
        return norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        d1, _ = self._d1_d2()
        return self.S * norm.pdf(d1) * np.sqrt(self.T) / 100 

    def theta(self):
        d1, d2 = self._d1_d2()
        term1 = -(self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
        if self.type == "call":
            theta = term1 - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            theta = term1 + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
        return theta / 365 

    def rho(self):
        _, d2 = self._d1_d2()
        if self.type == "call":
            rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            rho = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)
        return rho / 100 

# --- MAIN APP LOGIC ---
st.set_page_config(page_title="Option Pricing Tool", layout="wide")

# Custom CSS 
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .stMetric {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.header("1. Market Data Parameters")
ticker = st.sidebar.text_input("Ticker Symbol", value="SPY").upper()

try:
    stock_data = yf.Ticker(ticker)
    current_price = stock_data.history(period="1d")['Close'].iloc[-1]
    st.sidebar.success(f"Spot Price (S): ${current_price:.2f}")
except:
    st.sidebar.error("Invalid Ticker")
    current_price = 100.0

try:
    tnx = yf.Ticker("^TNX")
    risk_free_rate_default = tnx.history(period="1d")['Close'].iloc[-1] / 100
except:
    risk_free_rate_default = 0.045

st.sidebar.header("2. Option Structure")
option_type = st.sidebar.selectbox("Option Type", ["Call", "Put"])
strike_price = st.sidebar.number_input("Strike Price (K)", value=float(round(current_price, 0)), step=1.0)
expiry_date = st.sidebar.date_input("Expiry Date", value=date.today() + timedelta(days=30))

days_to_expiry = (expiry_date - date.today()).days
T = days_to_expiry / 365.0

st.sidebar.header("3. Volatility & Rates")
sigma = st.sidebar.slider("Implied Volatility (σ)", 0.05, 1.50, 0.20, 0.01)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (r)", value=risk_free_rate_default, format="%.4f")
buy_price = st.sidebar.number_input("Purchase Price (for P&L)", value=0.0, step=0.1)
st.sidebar.markdown("---")
st.sidebar.caption("Created by: **Ori Divon**")

# --- DASHBOARD ---
if T > 0:
    bs_model = BlackScholes(current_price, strike_price, T, risk_free_rate, sigma, option_type)
    theo_price = bs_model.price()
    delta = bs_model.delta()
    gamma = bs_model.gamma()
    vega = bs_model.vega()
    theta = bs_model.theta()
    rho = bs_model.rho()
    
    pnl = theo_price - buy_price if buy_price > 0 else 0

    st.title(f"📊 Option Pricing Tool: {ticker}")
    with st.expander("ℹ️ - About this Project"):
    st.markdown("""
    ### **Project Overview**
    This tool was engineered to bridge the gap between theoretical valuation models and real-time market data. I leveraged knowledge from my **Fixed Income and Derivatives** coursework, combined with **Python** development, to create a fully functional pricing engine.

    ### **Key Technical Features**
    * **Black-Scholes Model:** Custom implementation of the closed-form solution.
    * **Real-Time Data:** Automated fetching of Spot Prices and Risk-Free Rates via API.
    * **Scenario Analysis:** Interactive P&L heatmaps to visualize risk.
    
    *Built by **Ori Divon**.*
    """)
# -----------------------
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Theoretical Price", f"${theo_price:.2f}", f"{pnl:+.2f} (P&L)" if buy_price > 0 else None)
    col2.metric("Delta (Direction)", f"{delta:.3f}")
    col3.metric("Vega (Vol Risk)", f"{vega:.3f}")
    col4.metric("Theta (Time Decay)", f"{theta:.3f}")

    st.markdown("---")

    # --- HEATMAP: Debugged and Corrected ---
    
    # Define ranges for calculation
    vol_ticks = np.linspace(sigma * 0.8, sigma * 1.2, 10)
    spot_range = np.linspace(current_price * 0.90, current_price * 1.10, 10)

    z = []
    for v in vol_ticks:
        row = []
        for s in spot_range:
            sim_model = BlackScholes(s, strike_price, T, risk_free_rate, v, option_type)
            sim_price = sim_model.price()
            sim_pnl = sim_price - buy_price if buy_price > 0 else sim_price
            row.append(sim_pnl)
        z.append(row)

    st.subheader("🛠 Scenario Analysis: Spot vs. Volatility") 
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=np.round(spot_range, 2),
        y=vol_ticks, # FIX: Use full-precision data to prevent stacking/overlap
        colorscale='RdBu',
        zmid=0,
        text=np.round(z, 2), # Using two decimal places for the text inside cells
        texttemplate=" %{text} ",
        textfont={"size": 12}, # Smaller font to help with crowding
        hoverongaps=False
    ))

    fig.update_layout(
        title=f"P&L Sensitivity Matrix (Strike: {strike_price})",
        xaxis_title="Spot Price ($)",
        yaxis_title="Volatility (σ)",
        height=700, # Increased height for vertical spacing
        width=1100, # Increased width for horizontal spacing
        yaxis=dict(
            # FIX: Force all 10 labels to display cleanly
            tickvals=vol_ticks, 
            ticktext=[f"{v:.2f}" for v in vol_ticks], 
            tickmode='array' 
        )
    )

    st.plotly_chart(fig)

else:
    st.error("Expiration date must be in the future.")

# --- DOCUMENTATION ---
with st.expander("ℹ️ Model Assumptions & Methodology"):
    
    # 1. The Math (Highlighted)
    st.markdown("### 1. The Mathematics")
    st.info("The model uses the Black-Scholes-Merton (1973) closed-form solution:")
    st.latex(r"C = S N(d_1) - K e^{-rT} N(d_2)")
    
    st.markdown("---")

    # 2. Split layout for readability
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 2. Key Assumptions")
        st.markdown("""
        * **European Exercise:** Options are only exercised at maturity.
        * **Constant Volatility:** $\sigma$ is constant (no volatility smile/skew).
        * **Risk-Free Rate:** Uses the **10-Year Treasury (^TNX)** as the proxy.
        * **No Dividends:** This model does not account for dividend yield ($q$).
        * **Log-Normal Prices:** Stock returns are normally distributed.
        """)

    with col_b:
        st.markdown("### 3. Greek Definitions")
        st.markdown("""
        * **Delta ($\Delta$):** Sensitivity to **Spot Price**.
        * **Gamma ($\Gamma$):** Sensitivity to **Delta** (convexity).
        * **Vega ($\nu$):** Sensitivity to **Volatility** (1% change).
        * **Theta ($\Theta$):** Daily **Time Decay** (money lost per day).
        * **Rho ($\rho$):** Sensitivity to **Interest Rates**.
        """)

