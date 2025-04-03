from pathlib import Path
import pandas as pd
import numpy as np

curr_dir = Path(__file__).parent.resolve()

methods = [
    "ada_gpt4o",
    "di_gpt4o",
    "tablemage_gpt4o",
    "tablemage_gpt4o-tools_only",
    "tablemage_llama3.3_70b",
    "openinterpreter_gpt4o",
]

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

task_order = [
    "361111",
    "361286",
    "361070",
    "361278",
    "361093",
    "361094",
    "361288",
    "361072",
    "361076",
    "361279",
    "361280",
]


raw_output = {
    "Task ID": [],
    "Task Type": [],
    "Metric Value": [],
    "Method": [],
}


output = {
    "Task ID": [],
    "Task Type": [],
    "Mean": [],
    "Std": [],
    "CI": [],
    "Method": [],
    "Mean Normalized": [],
    "Std Normalized": [],
    "CI Normalized": [],
}


classification_output = {"Method": [], "Mean AUROC": [], "Std AUROC": []}

regression_output = {"Method": [], "Mean NRMSE": [], "Std NRMSE": []}

missing_val_output = {
    "Method": [],
    "Task ID": [],
    "Number of Missing Answers": [],
}

task_target_min_max_df = pd.read_csv(
    curr_dir / "results" / "statistics" / "task_target_min_max.csv"
)


for method in methods:
    print(method)

    for task_id in task_order:
        print(task_id)

        all_answer_paths = sorted(
            [
                path
                for path in (curr_dir / "results" / method / "formatted").glob("*.tsv")
                if path.stem.startswith("formatted-answers")
            ]
        )

        # get the metric
        formatted_answers = []
        for answer_path in all_answer_paths:
            answer_df = pd.read_csv(answer_path, sep="\t")
            answer_df["Dataset ID"] = answer_df["Dataset ID"].astype(str)
            formatted_answers.append(
                answer_df.loc[
                    answer_df["Dataset ID"] == task_id, "Formatted Answer"
                ].values[0]
            )

        raw_output["Task ID"].extend([task_id] * len(formatted_answers))
        raw_output["Task Type"].extend([tasks_dict[task_id]] * len(formatted_answers))
        raw_output["Metric Value"].extend(formatted_answers)
        raw_output["Method"].extend([method] * len(formatted_answers))

        # figure out number of missing values
        missing_val_output["Method"].append(method)
        missing_val_output["Task ID"].append(task_id)
        missing_val_output["Number of Missing Answers"].append(
            len([x for x in formatted_answers if str(x) == "nan"])
        )

        # needs to be robust to missing values, e.g. just ignore NaN values
        formatted_answers = [x for x in formatted_answers if str(x) != "nan"]

        formatted_answers_normalized = None
        if tasks_dict[task_id] == "regression":
            formatted_answers_normalized = [
                x
                / (
                    task_target_min_max_df.loc[
                        task_target_min_max_df["Task ID"] == int(task_id), "y_max"
                    ].values[0]
                    - task_target_min_max_df.loc[
                        task_target_min_max_df["Task ID"] == int(task_id), "y_min"
                    ].values[0]
                )
                for x in formatted_answers
            ]

        avg_metric = np.mean(formatted_answers)
        std_metric = np.std(formatted_answers)

        if tasks_dict[task_id] == "regression":
            avg_metric_normalized = np.mean(formatted_answers_normalized)
            std_metric_normalized = np.std(formatted_answers_normalized)
            output["Mean Normalized"].append(avg_metric_normalized)
            output["Std Normalized"].append(std_metric_normalized)
            output["CI Normalized"].append(
                1.96 * std_metric_normalized / np.sqrt(len(formatted_answers))
            )
        else:
            output["Mean Normalized"].append(np.nan)
            output["Std Normalized"].append(np.nan)
            output["CI Normalized"].append(np.nan)

        output["Task ID"].append(task_id)
        output["Mean"].append(avg_metric)
        output["Std"].append(std_metric)
        output["CI"].append(1.96 * std_metric / np.sqrt(len(formatted_answers)))
        output["Method"].append(method)
        output["Task Type"].append(tasks_dict[task_id])

        if tasks_dict[task_id] == "classification":
            classification_output["Method"].append(method)
            classification_output["Mean AUROC"].append(avg_metric)
            classification_output["Std AUROC"].append(std_metric)
        elif tasks_dict[task_id] == "regression":
            regression_output["Method"].append(method)
            regression_output["Mean NRMSE"].append(avg_metric_normalized)
            regression_output["Std NRMSE"].append(std_metric_normalized)


output_df = pd.DataFrame(output)
output_df.to_csv(curr_dir / "results" / "statistics" / "metrics.csv", index=False)


# make each method a column, with value mean
output_df = output_df.pivot(index="Task ID", columns="Method", values="Mean")
# add task type
output_df["Task Type"] = [tasks_dict[task_id] for task_id in output_df.index]

output_df.to_csv(curr_dir / "results" / "statistics" / "metrics_pivot.csv")


classification_output_df = pd.DataFrame(classification_output)
# average over the method
classification_output_df = (
    classification_output_df.groupby("Method").mean().reset_index()
)
classification_output_df.to_csv(
    curr_dir / "results" / "statistics" / "classification_metrics.csv", index=False
)


regression_output_df = pd.DataFrame(regression_output)
# average over the method
regression_output_df = regression_output_df.groupby("Method").mean().reset_index()
regression_output_df.to_csv(
    curr_dir / "results" / "statistics" / "regression_metrics.csv", index=False
)


raw_output_df = pd.DataFrame(raw_output)
raw_output_df.to_csv(
    curr_dir / "results" / "statistics" / "raw_metrics.csv", index=False
)


missing_val_output_df = pd.DataFrame(missing_val_output)
missing_val_output_df.to_csv(
    curr_dir / "results" / "statistics" / "missing_values.csv", index=False
)
