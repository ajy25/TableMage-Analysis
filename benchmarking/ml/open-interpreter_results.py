from pathlib import Path
import pandas as pd
from interpreter import interpreter


iternums = list(range(10))
subdir_stems_to_consider = [  # optionally comment out any of these
    "classification_mixed",
    "classification_numerical",
    "regression_mixed",
    "regression_numerical",
]

curr_dir = Path(__file__).resolve().parent

# read in the api key
with open(curr_dir / "api_key.txt") as f:
    api_key = f.read().strip()
interpreter.offline = True
interpreter.llm.model = "openai/gpt-4o-2024-08-06"
interpreter.llm.api_key = api_key
interpreter.llm.api_base = "https://api.openai.com/v1"
interpreter.system_message += """
Run shell commands with -y so the user doesn't have to confirm them.
"""
interpreter.auto_run = True

output_dir = curr_dir / "results" / "openinterpreter_gpt4o"
output_dir.mkdir(exist_ok=True)
datasets_dir = curr_dir / "datasets"
subdirs = sorted([x for x in datasets_dir.iterdir() if x.is_dir()])
subdirs = [x for x in subdirs if x.stem in subdir_stems_to_consider]


for iternum in iternums:
    for subdir in subdirs:
        print(f"Generating answers for {subdir.stem} dataset using TableMage...")

        if subdir.stem.split("_")[0] == "regression":
            metric = "RMSE"
            task = "regression"
        elif subdir.stem.split("_")[0] == "classification":
            metric = "AUROC"
            task = "classification"
        else:
            raise RuntimeError("Unknown dataset type")

        output_path = output_dir / f"{subdir.stem}-performances_{iternum}.csv"

        results_dict = {
            "file_name": [],
            "unformatted_answer": [],
        }

        for file in sorted(list(subdir.iterdir())):
            df = pd.read_csv(file)

            print(df.shape)

            target = df.columns[0]

            # for each dataset, reset the chat history
            interpreter.messages = []

            response = interpreter.chat(
                """\
The dataset is at {file_path}.
You must perform a 70/30 train/test split using sklearn’s train_test_split function, with random seed 42.

Predict the variable `{target}` with machine learning {task}. \
Please train the best possible model to accomplish this task. \
Report the test {metric} of the best possible model you can train. \
Only report the test {metric} value, rounded to 3 decimal points.
""".format(
                    file_path=file,
                    target=target,
                    task=task,
                    metric=metric,
                )
            )

            print(
                "\n\n"
                + "-" * 80
                + "\n"
                + f"TableMage's answer for {file.stem}:\n{response}"
                + "\n"
                + "-" * 80
                + "\n\n"
            )

            results_dict["file_name"].append(file.stem)
            results_dict["unformatted_answer"].append(response)

        results_df = pd.DataFrame(results_dict)
        results_df.to_csv(output_path, index=False)
