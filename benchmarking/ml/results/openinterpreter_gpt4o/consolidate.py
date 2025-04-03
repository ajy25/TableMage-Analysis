from pathlib import Path
import pandas as pd
import ast

curr_dir = Path(__file__).parent.resolve()

iternums = range(10)

subcategory_names = [
    "classification_mixed",
    "classification_numerical",
    "regression_mixed",
    "regression_numerical",
]


for iternum in iternums:
    output_df = {}

    for subcategory_name in subcategory_names:
        df_read_in = pd.read_csv(
            curr_dir / "orig" / f"{subcategory_name}-performances_{iternum}.csv",
        )
        # iterate through the rows
        for index, row in df_read_in.iterrows():
            dataset_id = row["file_name"].split("_")[0]
            answer = row["unformatted_answer"]
            # convert string (i.e., "[a, b, c]") to list
            answer = ast.literal_eval(answer)
            answer = str(answer[-1])
            output_df[dataset_id] = answer
            print(f"added {dataset_id}")

    df_output = pd.DataFrame.from_dict(output_df, orient="index")
    df_output.reset_index(inplace=True)
    df_output = df_output.rename(
        columns={"index": "Dataset ID", 0: "Unformatted Answer"}
    )
    df_output.to_csv(curr_dir / f"answers_{iternum}.csv", index=False)
