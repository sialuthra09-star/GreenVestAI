import pandas as pd

path = r"C:\Users\mluth\Downloads\iex_dam_15min (2).csv"

df = pd.read_csv(
    path,
    parse_dates=["date", "time_start_dt", "time_end_dt"]
)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# ============================================================
# STEP 2 — EXPLORATORY DATA ANALYSIS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assume df is already loaded from Step 1, e.g.:
# df = pd.read_csv(
#     r"C:\Users\mluth\Downloads\iex_dam_15min (2).csv",
#     parse_dates=["date", "time_start_dt", "time_end_dt"]
# )

# ------------------------------------------------------------
# 2.1 Basic information
# ------------------------------------------------------------

print("Dataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)


# ------------------------------------------------------------
# 2.2 Check missing values
# ------------------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 2.3 MCP descriptive statistics
# ------------------------------------------------------------

print("\n================ MCP STATISTICS ================")

mcp = df["mcp_rs_per_mwh"]
print(mcp.describe())


# ------------------------------------------------------------
# 2.4 Additional statistics
# ------------------------------------------------------------

print("\nMean MCP:", mcp.mean())
print("Median MCP:", mcp.median())
print("Standard deviation:", mcp.std())
print("Minimum MCP:", mcp.min())
print("Maximum MCP:", mcp.max())

print("\nMCP percentiles:")
print(mcp.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]))


# ------------------------------------------------------------
# 2.5 Check date range
# ------------------------------------------------------------

print("\n================ DATE RANGE ================")

print("Starting date/time:", df["time_start_dt"].min())
print("Ending date/time:", df["time_start_dt"].max())


# ------------------------------------------------------------
# 2.6 Check number of observations per day
# ------------------------------------------------------------

observations_per_day = df.groupby("date").size()

print("\n================ OBSERVATIONS PER DAY ================")

print(observations_per_day.describe())

print("\nObservations per day:")
print(observations_per_day)


# ------------------------------------------------------------
# 2.7 Plot MCP over time
# ------------------------------------------------------------

plt.figure(figsize=(14, 5))

plt.plot(
    df["time_start_dt"],
    mcp
)

plt.xlabel("Date and Time")
plt.ylabel("MCP (₹/MWh)")
plt.title("IEX DAM — 15-Minute Market Clearing Price")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 2.8 MCP distribution
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.hist(
    mcp,
    bins=50
)

plt.xlabel("MCP (₹/MWh)")
plt.ylabel("Frequency")
plt.title("Distribution of IEX DAM MCP")

plt.tight_layout()

plt.show()

# ============================================================
# STEP 3 — PRICE SPIKE ANALYSIS
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assume df is already loaded and mcp defined, e.g.:
# mcp = df["mcp_rs_per_mwh"]

# ------------------------------------------------------------
# 3.1 Create MCP variable
# ------------------------------------------------------------

mcp = df["mcp_rs_per_mwh"]


# ------------------------------------------------------------
# 3.2 Calculate spike thresholds
# ------------------------------------------------------------

# 95th percentile
p95 = mcp.quantile(0.95)

# 99th percentile
p99 = mcp.quantile(0.99)

print("================ PRICE SPIKE THRESHOLDS ================")

print("95th percentile:", p95)
print("99th percentile:", p99)


# ------------------------------------------------------------
# 3.3 Classify price spikes
# ------------------------------------------------------------

df["spike_95"] = df["mcp_rs_per_mwh"] > p95
df["spike_99"] = df["mcp_rs_per_mwh"] > p99


# ------------------------------------------------------------
# 3.4 Number and percentage of spikes
# ------------------------------------------------------------

number_spikes_95 = df["spike_95"].sum()
number_spikes_99 = df["spike_99"].sum()

percentage_spikes_95 = df["spike_95"].mean() * 100
percentage_spikes_99 = df["spike_99"].mean() * 100


print("\n================ SPIKE FREQUENCY ================")

print("95% threshold:")
print("Number of spikes:", number_spikes_95)
print("Percentage of observations:", percentage_spikes_95, "%")

print("\n99% threshold:")
print("Number of spikes:", number_spikes_99)
print("Percentage of observations:", percentage_spikes_99, "%")


# ------------------------------------------------------------
# 3.5 Extract the highest-price observations
# ------------------------------------------------------------

top_spikes = df.sort_values(
    "mcp_rs_per_mwh",
    ascending=False
).head(20)


print("\n================ TOP 20 HIGHEST PRICES ================")

print(
    top_spikes[
        ["date", "time_start_dt", "time_end_dt", "mcp_rs_per_mwh"]
    ].to_string(index=False)
)


# ------------------------------------------------------------
# 3.6 Calculate spike magnitude
# ------------------------------------------------------------

print("\n================ SPIKE MAGNITUDE ================")

print("Maximum MCP:", mcp.max())

print(
    "Maximum / Median ratio:",
    mcp.max() / mcp.median()
)

print(
    "Maximum / Mean ratio:",
    mcp.max() / mcp.mean()
)


# ------------------------------------------------------------
# 3.7 Plot price spikes
# ------------------------------------------------------------

plt.figure(figsize=(14, 5))

plt.plot(
    df["time_start_dt"],
    mcp,
    label="MCP"
)

plt.axhline(
    p95,
    linestyle="--",
    label="95th percentile"
)

plt.axhline(
    p99,
    linestyle="--",
    label="99th percentile"
)

plt.xlabel("Date and Time")
plt.ylabel("MCP (₹/MWh)")
plt.title("IEX DAM Price Spikes")

plt.legend()
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 3.8 Identify spike time blocks
# ------------------------------------------------------------

spike_observations = df[df["spike_95"]].copy()

print("\n================ SPIKE OBSERVATIONS ================")

print(
    spike_observations[
        ["date", "time_start_dt", "time_end_dt", "mcp_rs_per_mwh"]
    ].head(20).to_string(index=False)
)



price_col = "mcp_rs_per_mwh"

print("Maximum price:", df[price_col].max())

print("\nNumber of ₹10,000 observations:")
print((df[price_col] == 10000).sum())

print("\nPercentage of observations at ₹10,000:")
print((df[price_col] == 10000).mean() * 100)

print("\nTop 20 prices:")
print(
    df[
        ["date", "time_start_dt", "time_end_dt", price_col]
    ]
    .sort_values(price_col, ascending=False)
    .head(20)
    .to_string(index=False)
)


# ============================================================
# STEP 4 — PRICE REGIME CLASSIFICATION
# ============================================================

import pandas as pd
import numpy as np

price_col = "mcp_rs_per_mwh"

# Make sure data is sorted chronologically
df = df.sort_values("time_start_dt").reset_index(drop=True)

# ------------------------------------------------------------
# 4.1 Separate capped-price observations
# ------------------------------------------------------------

cap_price = 10000

df["regime"] = "Normal"


# ------------------------------------------------------------
# 4.2 Calculate thresholds using ONLY prices below the cap
# ------------------------------------------------------------

non_cap_mask = df[price_col] < cap_price
non_cap_prices = df.loc[non_cap_mask, price_col]

if non_cap_prices.empty:
    raise ValueError(
        "No observations below the cap price; cannot compute thresholds."
    )

low_threshold = non_cap_prices.quantile(0.33)
high_threshold = non_cap_prices.quantile(0.67)

print("LOW threshold:", low_threshold)
print("HIGH threshold:", high_threshold)


# ------------------------------------------------------------
# 4.3 Assign regimes
# ------------------------------------------------------------

df.loc[
    df[price_col] < low_threshold,
    "regime"
] = "Low"

df.loc[
    (df[price_col] >= low_threshold) &
    (df[price_col] < high_threshold),
    "regime"
] = "Normal"

df.loc[
    (df[price_col] >= high_threshold) &
    (df[price_col] < cap_price),
    "regime"
] = "High"

df.loc[
    df[price_col] == cap_price,
    "regime"
] = "Capped"


# ------------------------------------------------------------
# 4.4 Count observations in each regime
# ------------------------------------------------------------

print("\n================ REGIME COUNTS ================")

print(
    df["regime"].value_counts()
)


# ------------------------------------------------------------
# 4.5 Regime percentages
# ------------------------------------------------------------

print("\n================ REGIME PERCENTAGES ================")

regime_percentages = (
    df["regime"]
    .value_counts(normalize=True)
    .mul(100)
)

print(regime_percentages)


# ------------------------------------------------------------
# 4.6 Average MCP within each regime
# ------------------------------------------------------------

print("\n================ REGIME STATISTICS ================")

regime_stats = (
    df.groupby("regime")[price_col]
    .agg(
        ["count", "mean", "median", "std", "min", "max"]
    )
)

print(regime_stats)


# ============================================================
# STEP 5 — MARKOV TRANSITION MATRIX
# ============================================================

import pandas as pd
import numpy as np

# Ensure chronological order (if not already done)
df = df.sort_values("time_start_dt").reset_index(drop=True)

# ------------------------------------------------------------
# 5.1 Create the next-period regime
# ------------------------------------------------------------

df["next_regime"] = df["regime"].shift(-1)

# Drop the last row (it has NaN for next_regime)
df_markov = df.dropna(subset=["next_regime"]).copy()


# ------------------------------------------------------------
# 5.2 Define regime order
# ------------------------------------------------------------

states = ["Low", "Normal", "High", "Capped"]


# ------------------------------------------------------------
# 5.3 Count transitions
# ------------------------------------------------------------

transition_counts = pd.crosstab(
    df_markov["regime"],
    df_markov["next_regime"],
    rownames=["current"],
    colnames=["next"]
)

# Reindex to ensure all states appear in the same order
transition_counts = transition_counts.reindex(
    index=states,
    columns=states,
    fill_value=0
)

print("============== TRANSITION COUNTS ==============")
print(transition_counts)


# ------------------------------------------------------------
# 5.4 Convert counts into probabilities
# ------------------------------------------------------------

# Avoid division by zero for any state with no observations
row_sums = transition_counts.sum(axis=1)
transition_matrix = transition_counts.div(row_sums, axis=0).fillna(0)

print("\n============== MARKOV TRANSITION MATRIX ==============")
print(transition_matrix.round(4))


# ------------------------------------------------------------
# 5.5 Verify that each row sums to 1
# ------------------------------------------------------------

print("\n============== ROW SUMS ==============")
print(transition_matrix.sum(axis=1).round(4))


# ------------------------------------------------------------
# 5.6 Display probabilities in percentage form
# ------------------------------------------------------------

print("\n============== TRANSITION PROBABILITIES (%) ==============")
print((transition_matrix * 100).round(2))


# ============================================================
# STEP 6 — MONTE CARLO ELECTRICITY PRICE SIMULATION
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assume df, transition_matrix, and price_col are already defined
# from previous steps:
# price_col = "mcp_rs_per_mwh"
# states = ["Low", "Normal", "High", "Capped"]
# transition_matrix: DataFrame indexed and column-named by states

price_col = "mcp_rs_per_mwh"
states = ["Low", "Normal", "High", "Capped"]

# ------------------------------------------------------------
# 6.1 Settings
# ------------------------------------------------------------

n_simulations = 1000
n_periods = 96          # 96 × 15 minutes = 24 hours

# ------------------------------------------------------------
# 6.2 Starting regime
# ------------------------------------------------------------

# Start the simulation from the regime of the
# final observation in your dataset.

initial_state = df["regime"].iloc[-1]

print("Starting regime:", initial_state)


# ------------------------------------------------------------
# 6.3 Store simulated prices
# ------------------------------------------------------------

simulated_prices = np.zeros(
    (n_simulations, n_periods)
)

simulated_regimes = np.empty(
    (n_simulations, n_periods),
    dtype=object
)


# ------------------------------------------------------------
# 6.4 Create price distributions for each regime
# ------------------------------------------------------------

regime_prices = {}

for state in states:
    prices = df.loc[df["regime"] == state, price_col].values
    if len(prices) == 0:
        # Fallback: use global median if a regime has no data
        prices = [df[price_col].median()]
    regime_prices[state] = prices


# ------------------------------------------------------------
# 6.5 Monte Carlo simulation
# ------------------------------------------------------------

# Ensure transition_matrix is aligned to 'states'
transition_matrix = transition_matrix.reindex(
    index=states,
    columns=states,
    fill_value=0
)

# Normalize rows to sum to 1 (in case of any rounding / zero-row issues)
row_sums = transition_matrix.sum(axis=1)
transition_matrix = transition_matrix.div(row_sums, axis=0).fillna(0)

for simulation in range(n_simulations):

    current_state = initial_state

    for t in range(n_periods):

        # Store current regime
        simulated_regimes[simulation, t] = current_state

        # Generate price
        if current_state == "Capped":
            # Capped regime is exactly ₹10,000
            simulated_prices[simulation, t] = 10000
        else:
            # Randomly select an observed price from the current regime
            simulated_prices[simulation, t] = np.random.choice(
                regime_prices[current_state]
            )

        # Move to next regime
        probabilities = transition_matrix.loc[current_state].values

        # Guard against any row that might still sum to 0
        if probabilities.sum() == 0:
            # Stay in the same state if no transitions defined
            next_state = current_state
        else:
            next_state = np.random.choice(states, p=probabilities)

        current_state = next_state


# ------------------------------------------------------------
# 6.6 Convert simulated prices to DataFrame
# ------------------------------------------------------------

simulated_df = pd.DataFrame(
    simulated_prices,
    columns=[
        f"Period_{i+1}"
        for i in range(n_periods)
    ]
)

print("\nSimulation completed.")
print("Number of simulations:", n_simulations)
print("Periods per simulation:", n_periods)


# ------------------------------------------------------------
# 6.7 Calculate statistics across simulations
# ------------------------------------------------------------

mean_path = simulated_prices.mean(axis=0)

lower_5 = np.percentile(simulated_prices, 5, axis=0)
upper_95 = np.percentile(simulated_prices, 95, axis=0)


# ------------------------------------------------------------
# 6.8 Plot Monte Carlo scenarios
# ------------------------------------------------------------

plt.figure(figsize=(14, 6))

# Plot first 100 simulated paths
for i in range(100):
    plt.plot(
        range(n_periods),
        simulated_prices[i],
        alpha=0.15
    )

# Mean simulated path
plt.plot(
    range(n_periods),
    mean_path,
    linewidth=2,
    label="Mean simulated price"
)

plt.xlabel("15-Minute Period")
plt.ylabel("MCP (₹/MWh)")
plt.title("Monte Carlo Simulation of IEX Day-Ahead Electricity Prices")

plt.legend()
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 6.9 Summary statistics
# ------------------------------------------------------------

print("\n================ MONTE CARLO RESULTS ================")

print("Overall simulated mean price:", simulated_prices.mean())
print("Overall simulated median price:", np.median(simulated_prices))
print("Overall simulated minimum:", simulated_prices.min())
print("Overall simulated maximum:", simulated_prices.max())


# ------------------------------------------------------------
# 6.10 Probability of hitting the price cap
# ------------------------------------------------------------

cap_probability = (simulated_prices == 10000).mean() * 100

print(
    "\nProbability of simulated price being ₹10,000/MWh:",
    cap_probability,
    "%"
)


# ------------------------------------------------------------
# 6.11 Save simulation results
# ------------------------------------------------------------

simulated_df.to_csv(
    "IEX_Monte_Carlo_Simulations.csv",
    index=False
)

print(
    "\nSimulation results saved as 'IEX_Monte_Carlo_Simulations.csv'"
)


# ============================================================
# STEP 7 — MONTE CARLO MODEL VALIDATION
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("\n================ MODEL VALIDATION ================")

# Actual data statistics
actual_mean = df[price_col].mean()
actual_median = df[price_col].median()
actual_min = df[price_col].min()
actual_max = df[price_col].max()
actual_std = df[price_col].std()

actual_cap_probability = (
    (df[price_col] == 10000).mean() * 100
)

# Simulated statistics
sim_mean = simulated_prices.mean()
sim_median = np.median(simulated_prices)
sim_min = simulated_prices.min()
sim_max = simulated_prices.max()
sim_std = simulated_prices.std()

sim_cap_probability = (
    (simulated_prices == 10000).mean() * 100
)

print("\nACTUAL DATA")
print("-----------------------------")
print("Mean:", actual_mean)
print("Median:", actual_median)
print("Minimum:", actual_min)
print("Maximum:", actual_max)
print("Standard deviation:", actual_std)
print("Capped probability:", actual_cap_probability, "%")


print("\nSIMULATED DATA")
print("-----------------------------")
print("Mean:", sim_mean)
print("Median:", sim_median)
print("Minimum:", sim_min)
print("Maximum:", sim_max)
print("Standard deviation:", sim_std)
print("Capped probability:", sim_cap_probability, "%")


# Compare differences
print("\nDIFFERENCE (Simulated − Actual)")
print("-----------------------------")

print(
    "Mean difference:",
    sim_mean - actual_mean
)

print(
    "Median difference:",
    sim_median - actual_median
)

print(
    "Std dev difference:",
    sim_std - actual_std
)

print(
    "Capped probability difference:",
    sim_cap_probability - actual_cap_probability,
    "percentage points"
)


# Optional: simple visual comparison of distributions
plt.figure(figsize=(10, 5))

plt.hist(
    df[price_col],
    bins=50,
    alpha=0.6,
    label="Actual MCP",
    density=True
)

plt.hist(
    simulated_prices.ravel(),
    bins=50,
    alpha=0.6,
    label="Simulated MCP",
    density=True
)

plt.xlabel("MCP (₹/MWh)")
plt.ylabel("Density")
plt.title("Actual vs Simulated MCP Distribution")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# STEP 8 — ACTUAL VS SIMULATED PRICES
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# Actual prices
actual_prices = df[price_col].values

# Mean simulated price for each 15-minute period
mean_simulated = simulated_prices.mean(axis=0)

# --------------------------------------------------------
# 8.1 Full-series comparison
# --------------------------------------------------------

plt.figure(figsize=(14, 5))

plt.plot(
    actual_prices,
    label="Actual IEX Price",
    linewidth=1,
    alpha=0.8
)

plt.plot(
    mean_simulated,
    label="Mean Simulated Price",
    linewidth=2,
    color="red"
)

plt.xlabel("15-Minute Observation Index")
plt.ylabel("MCP (₹/MWh)")
plt.title("Actual vs Mean Simulated IEX Electricity Prices (Full Series)")

plt.legend()
plt.tight_layout()
plt.show()


# --------------------------------------------------------
# 8.2 Residuals: Actual − Mean Simulated
# --------------------------------------------------------

# Only possible if lengths match; if not, truncate to min length
n_actual = len(actual_prices)
n_sim = len(mean_simulated)
n_common = min(n_actual, n_sim)

residuals = actual_prices[:n_common] - mean_simulated[:n_common]

plt.figure(figsize=(14, 3))

plt.plot(
    residuals,
    linewidth=1,
    color="green"
)

plt.axhline(0, linestyle="--", color="black", linewidth=1)

plt.xlabel("15-Minute Observation Index")
plt.ylabel("Residual (Actual − Mean Simulated) [₹/MWh]")
plt.title("Residuals: Actual vs Mean Simulated MCP")

plt.tight_layout()
plt.show()


# --------------------------------------------------------
# 8.3 Single-day comparison (first full day)
# --------------------------------------------------------

# Assuming 96 observations per day
obs_per_day = 96

# Use the first complete day in your dataset
start_idx = 0
end_idx = obs_per_day

actual_day = actual_prices[start_idx:end_idx]
sim_day = mean_simulated[start_idx:end_idx]

plt.figure(figsize=(10, 5))

plt.plot(
    actual_day,
    label="Actual IEX Price (Day 1)",
    linewidth=1.5,
    marker="",
)

plt.plot(
    sim_day,
    label="Mean Simulated Price (Day 1)",
    linewidth=2,
    marker="",
    color="red"
)

plt.xlabel("15-Minute Period (within day)")
plt.ylabel("MCP (₹/MWh)")
plt.title("Actual vs Mean Simulated MCP — First 24 Hours")

plt.xticks(range(0, obs_per_day, 4))  # every 1 hour
plt.legend()
plt.tight_layout()
plt.show()
