# main.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# --- Set up the Streamlit page ---
st.set_page_config(page_title="Modern Portfolio Theory Optimiser", layout="wide")
st.title("📈 Modern Portfolio Theory Optimiser")
st.write("This application uses Monte Carlo simulation to find the optimal portfolios based on the principles of Modern Portfolio Theory.")

# --- Sidebar for User Inputs ---

st.sidebar.header("Portfolio Optimiser")
st.sidebar.markdown("""
**Created by:**
<a href="www.linkedin.com/in/kheelan-sarathee-319877261" target="_blank">
    <img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg" width="20" style="vertical-align:middle; margin-right: 5px;">
    Kheelan Sarathee
</a>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Portfolio Inputs")
    
    # Input for tickers
    tickers_input = st.text_area("Enter Stock Tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN,TSLA")
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", dt.date(2020, 1, 1))
    with col2:
        end_date = st.date_input("End Date", dt.date.today())
        
    # Other parameters
    risk_free_rate = st.number_input("Risk-Free Rate (e.g., 0.03 for 3%)", 0.0, 1.0, 0.03, 0.01)
    num_portfolios = st.slider("Number of Portfolios to Simulate", 100000, 100000, 800000)

    # Button to run the analysis
    run_button = st.button("Optimize Portfolio")

# --- Main functions from your notebook ---
def portfolio_return(weights, mean_returns):
    """Calculates the expected return of a portfolio."""
    return np.dot(weights, mean_returns)

def portfolio_volatility(weights, cov_matrix):
    """Calculates the volatility (std dev) of a portfolio."""
    port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
    return np.sqrt(port_var)

def sharpe_ratio(weights, mean_returns, cov_matrix, rf):
    """Calculates the Sharpe ratio of a portfolio."""
    ret = portfolio_return(weights, mean_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    if vol == 0:
        return 0
    return (ret - rf) / vol

def rand_weights(n_assets):
    """Generates random weights for a portfolio."""
    w = np.random.random(n_assets)
    return w / w.sum()

# --- Main application logic ---
if run_button and tickers:
    with st.spinner("Fetching data and running simulations... This may take a moment."):
        try:
            # --- Step 1: Download Data ---
            df = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
            if df.empty or 'Adj Close' not in df.columns:
                st.error("Could not download data for the given tickers. Please check the tickers and date range.")
                st.stop()
            
            adj_close = df['Adj Close'].dropna()

            # --- Step 2: Compute Returns and Covariance ---
            df_lr = np.log(adj_close / adj_close.shift(1)).dropna()
            mean_returns = df_lr.mean() * 252
            cov_matrix = df_lr.cov() * 252

            # --- Step 3: Run Monte Carlo Simulation ---
            num_assets = len(tickers)
            results = np.zeros((3, num_portfolios))
            all_weights = np.zeros((num_portfolios, num_assets))

            for i in range(num_portfolios):
                weights = rand_weights(num_assets)
                all_weights[i, :] = weights
                
                ret = portfolio_return(weights, mean_returns)
                vol = portfolio_volatility(weights, cov_matrix)
                sr = sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate)

                results[0, i] = ret
                results[1, i] = vol
                results[2, i] = sr
            
            portfolios_df = pd.DataFrame(results.T, columns=['Return', 'Volatility', 'SharpeRatio'])
            
            # --- Step 4: Find Key Portfolios ---
            max_sharpe_idx = portfolios_df['SharpeRatio'].idxmax()
            min_vol_idx = portfolios_df['Volatility'].idxmin()

            max_sharpe_portfolio = portfolios_df.loc[max_sharpe_idx]
            min_vol_portfolio = portfolios_df.loc[min_vol_idx]

            optimal_sharpe_weights = all_weights[max_sharpe_idx]
            min_vol_weights = all_weights[min_vol_idx]

            st.success("Optimization Complete!")
            
            # --- Step 5: Display Results ---
            st.header("Optimization Results")

            # Display the larger graph first
            st.subheader("Efficient Frontier")
            fig, ax = plt.subplots(figsize=(10, 6)) # Larger figure size
            scatter = ax.scatter(
                portfolios_df['Volatility'],
                portfolios_df['Return'],
                c=portfolios_df['SharpeRatio'],
                cmap='viridis', marker='o', s=10, alpha=0.5
            )
            plt.colorbar(scatter, label='Sharpe Ratio')
            ax.scatter(max_sharpe_portfolio['Volatility'], max_sharpe_portfolio['Return'], color='red', marker='*', s=150, edgecolors='black', label='Max Sharpe Ratio')
            ax.scatter(min_vol_portfolio['Volatility'], min_vol_portfolio['Return'], color='blue', marker='X', s=150, edgecolors='black', label='Min Volatility')
            ax.set_title('Efficient Frontier')
            ax.set_xlabel('Volatility')
            ax.set_ylabel('Return')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)

            # Create two columns below the graph for the portfolio data
            st.subheader("Optimal Portfolio Weights & Performance")
            data_col1, data_col2 = st.columns(2, gap="large")

            with data_col1:
                st.markdown("#### Maximum Sharpe Ratio Portfolio")
                max_sharpe_df = pd.DataFrame(optimal_sharpe_weights, index=tickers, columns=['Weight'])
                max_sharpe_df['Weight'] = max_sharpe_df['Weight'].map('{:.2%}'.format)
                st.dataframe(max_sharpe_df)
                st.write(f"**Return:** {max_sharpe_portfolio['Return']:.2%}")
                st.write(f"**Volatility:** {max_sharpe_portfolio['Volatility']:.2%}")
                st.write(f"**Sharpe Ratio:** {max_sharpe_portfolio['SharpeRatio']:.2f}")

            with data_col2:
                st.markdown("#### Minimum Volatility Portfolio")
                min_vol_df = pd.DataFrame(min_vol_weights, index=tickers, columns=['Weight'])
                min_vol_df['Weight'] = min_vol_df['Weight'].map('{:.2%}'.format)
                st.dataframe(min_vol_df)
                st.write(f"**Return:** {min_vol_portfolio['Return']:.2%}")
                st.write(f"**Volatility:** {min_vol_portfolio['Volatility']:.2%}")
                st.write(f"**Sharpe Ratio:** {min_vol_portfolio['SharpeRatio']:.2f}")

            # --- Additional Plots ---
            st.header("Additional Analysis")
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("Annualised Returns")
                fig_returns, ax_returns = plt.subplots()
                mean_returns.plot(kind='bar', ax=ax_returns)
                ax_returns.set_ylabel('Expected Annual Return')
                st.pyplot(fig_returns)

            with col4:
                st.subheader("Covariance Matrix Heatmap")
                fig_cov, ax_cov = plt.subplots()
                sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', ax=ax_cov, fmt=".2f")
                st.pyplot(fig_cov)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.info("Please ensure the tickers are valid and data is available for the selected period.")
