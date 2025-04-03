from pathlib import Path
import pandas as pd
import numpy as np

curr_dir = Path(__file__).parent.resolve()

tasks_dict = {
    "361093": "regression",
    "361094": "regression",
    "361288": "regression",
    "361072": "regression",
    "361076": "regression",
    "361279": "regression",
    "361280": "regression",
}

task_order = ["361093", "361094", "361288", "361072", "361076", "361279", "361280"]

task_to_path = {
    "361093": curr_dir
    / "datasets"
    / "regression_mixed"
    / "361093_analcatdata_supreme.csv",
    "361094": curr_dir
    / "datasets"
    / "regression_mixed"
    / "361094_visualizing_soil.csv",
    "361288": curr_dir / "datasets" / "regression_mixed" / "361288_abalone.csv",
    "361072": curr_dir / "datasets" / "regression_numerical" / "361072_cpu_act.csv",
    "361076": curr_dir
    / "datasets"
    / "regression_numerical"
    / "361076_wine_quality.csv",
    "361279": curr_dir / "datasets" / "regression_numerical" / "361279_yprop_4_1.csv",
    "361280": curr_dir / "datasets" / "regression_numerical" / "361280_abalone.csv",
}

output = {
    "Task ID": [],
    "y_min": [],
    "y_max": [],
    "y_std": [],
}

for task_id in task_order:
    df = pd.read_csv(task_to_path[task_id])
    # target column is first column
    y = df.iloc[:, 0]
    output["Task ID"].append(task_id)
    output["y_min"].append(np.min(y))
    output["y_max"].append(np.max(y))
    output["y_std"].append(np.std(y))

output = pd.DataFrame(output)
output.to_csv(
    curr_dir / "results" / "statistics" / "task_target_min_max.csv", index=False
)
