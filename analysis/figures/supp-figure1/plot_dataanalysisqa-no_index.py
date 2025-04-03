from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


sns.set_theme("paper", "whitegrid")


curr_dir = Path(__file__).resolve().parent


output_dir = curr_dir / "figures"


input_dir = (
    curr_dir.parent.parent.parent
    / "benchmarking"
    / "dataanalysisqa"
    / "results"
    / "statistics-no_index"
    / "accuracy"
)
assert input_dir.exists(), input_dir


id_to_agentname = {
    "ada_gpt4o": "Advanced Data Analysis (GPT-4o)",
    "di_gpt4o": "Data Interpreter (GPT-4o)",
    "tablemage_gpt4o": "ChatDA (GPT-4o)",
    "openinterpreter_gpt4o": "Open Interpreter (GPT-4o)",
    "tablemage_llama3.3_70b": "ChatDA (Llama 3.3 70B)",
    "tablemage_gpt4o-tools_only": "ChatDA (GPT-4o; non-coding)",
}

# combine into one dataframe
dfs_to_concat_vertically = []

for summary_path in sorted(list(input_dir.glob("*.csv"))):
    summary_df = pd.read_csv(summary_path, index_col=0)
    summary_df = summary_df.reset_index(drop=False, names=["Agent"])
    target = summary_path.stem.split("-")[1]
    if target != "All":
        summary_df["Category"] = target
    else:
        summary_df["Category"] = "Overall"
    dfs_to_concat_vertically.append(summary_df)

summary_df = pd.concat(dfs_to_concat_vertically, axis=0)
summary_df["Agent"] = summary_df["Agent"].map(id_to_agentname)

# make a grouped bar plot, grouped by Category; include the 95% confidence interval error bars (mean and std columns), each bar is an agent
# use different colors for each agent

fig, ax = plt.subplots(figsize=(6, 3))
df = summary_df
df["ci"] = df["ci"]

# Plot grouped bar plot
categories = [
    "Overall",
    "Summary Statistics",
    "Statistical Testing",
    "Regression Analysis",
    "Data Transformation",
    "Persistence",
]
agents = [
    "ChatDA (GPT-4o)",
    "ChatDA (GPT-4o; non-coding)",
    "ChatDA (Llama 3.3 70B)",
    "Advanced Data Analysis (GPT-4o)",
    "Data Interpreter (GPT-4o)",
    "Open Interpreter (GPT-4o)",
]
agent_to_color = {
    "ChatDA (GPT-4o)": "#92278F",
    "ChatDA (GPT-4o; non-coding)": "#BB76BC",
    "ChatDA (Llama 3.3 70B)": "#E4C6E9",
    "Advanced Data Analysis (GPT-4o)": "#E46C0A",
    "Data Interpreter (GPT-4o)": "#375623",
    "Open Interpreter (GPT-4o)": "#1F4E79",
}


x = np.arange(len(categories))  # X-axis positions for categories
width = 0.13  # Width of each bar

fig, ax = plt.subplots(figsize=(13, 6))
df["Category"] = pd.Categorical(df["Category"], categories=categories, ordered=True)

df = df.sort_values("Category")  # Sort to enforce order


for i, agent in enumerate(agents):
    subset = df[df["Agent"] == agent]
    means = subset["mean"].values
    ci = subset["ci"].values
    color = agent_to_color[agent]

    ax.bar(
        x + (i - (len(agents) - 1) / 2) * width,  # Center the bars on tick positions
        means,
        width,
        label=agent,
        yerr=ci,
        capsize=5,
        alpha=1,
        color=color,
    )

# Formatting
ax.set_ylabel("Accuracy", fontsize=15)

# Set y tick label size
ax.tick_params(axis="y", labelsize=12)

# Ensure xticks are correctly aligned with categories
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=20, ha="right", fontsize=12)

# Put legend to the right of the plot
ax.legend(bbox_to_anchor=(1.01, 0.5), loc="center left", fontsize=12)

# Remove gridlines
ax.grid(False)

# Save figure
fig.tight_layout()
fig.savefig(output_dir / "dataanalysisqa-tools_only.png", dpi=300)
