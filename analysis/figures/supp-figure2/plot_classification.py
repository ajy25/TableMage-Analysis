from pathlib import Path
import pandas as pd
# set copy-on-write
pd.options.mode.chained_assignment = None
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches


sns.set_theme("paper", "whitegrid")


curr_dir = Path(__file__).resolve().parent
output_dir = curr_dir / "figures"

tasks_dict = {
    "361111": "classification",
    "361286": "classification",
    "361070": "classification",
    "361278": "classification",
    "361093": "regression",
    "361094": "regression",
    "361288": "regression",
    "361072": "regression",
    "361076": "regression",
    "361279": "regression",
    "361280": "regression",
}

statistics_df = pd.read_csv(
    curr_dir.parent.parent.parent
    / "benchmarking"
    / "ml"
    / "results"
    / "statistics"
    / "metrics.csv"
)
statistics_df["Task ID"] = statistics_df["Task ID"].astype(str)


id_to_agentname = {
    "ada_gpt4o": "Advanced Data Analysis (GPT-4o)",
    "di_gpt4o": "Data Interpreter (GPT-4o)",
    "tablemage_gpt4o": "ChatDA (GPT-4o)",
    "openinterpreter_gpt4o": "Open Interpreter (GPT-4o)",
    "tablemage_llama3.3_70b": "ChatDA (Llama 3.3 70B)",
    "tablemage_gpt4o-tools_only": "ChatDA (GPT-4o; non-coding)",
}

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


statistics_df = statistics_df[statistics_df["Task Type"] == "classification"]
tasks = statistics_df["Task ID"].unique().tolist()

raw_df = pd.read_csv(
    curr_dir.parent.parent.parent
    / "benchmarking"
    / "ml"
    / "results"
    / "statistics"
    / "raw_metrics.csv"
)
raw_df = raw_df[raw_df["Task Type"] == "classification"]
raw_df["Task ID"] = raw_df["Task ID"].astype(str)
print(raw_df.shape)

fig, axs = plt.subplots(nrows=len(tasks), ncols=1, figsize=(8, 9.1))
df = statistics_df
df["Method"] = df["Method"].map(id_to_agentname)

print(df)

agents = [
    "ChatDA (GPT-4o)",
    "ChatDA (GPT-4o; non-coding)",
    "ChatDA (Llama 3.3 70B)",
    "Advanced Data Analysis (GPT-4o)",
    "Data Interpreter (GPT-4o)",
    "Open Interpreter (GPT-4o)",
]


for i, task in enumerate(tasks):
    task_df = df[df["Task ID"] == task]
    print(task_df.shape)
    order = [agent for agent in agents if agent in task_df["Method"].values]
    ax = axs[i]
    task_df["Method"] = pd.Categorical(task_df["Method"], categories=agents, ordered=True)
    task_df = task_df.sort_values("Method")
    ax.bar(
        task_df["Method"],
        task_df["Mean"],
        yerr=task_df["Std"],
        color=[agent_to_color[agent] for agent in task_df["Method"]],
        capsize=5,
    )
    ax.set_ylim(bottom=0)
    ax.grid(False)

    # Formatting
    axs[i].set_title(f"Task {task}", fontsize=12)
    axs[i].set_ylabel("AUC", fontsize=12)

    # remove x ticklabels
    axs[i].set_xticklabels([])
    axs[i].set_xticks([])

    agent_order = task_df["Method"].to_list()
    task_df.set_index("Method", inplace=True)

    custom_labels = [
        f"{agent}: {task_df.loc[agent, "Mean"]:.3f} ± {task_df.loc[agent, "CI"]:.3f}" for agent in agent_order
    ]

    legend_handles = [
        mpatches.Patch(color=agent_to_color[agent], label=custom_label)
        for agent, custom_label in zip(agent_order, custom_labels)
    ]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.01, 0.5), loc="center left", fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / "classification.png", dpi=300)