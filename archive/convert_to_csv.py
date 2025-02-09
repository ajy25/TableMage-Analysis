import pandas as pd

df: pd.DataFrame = pd.read_pickle("scl_df.pkl")
print(df.columns.to_list())

df = df.drop(
    columns=[
        "has_measurements",
        "patient_id",
        "scl_measurements",
        "avg_hf",
        "medial_hf",
        "lateral_hf",
        "Medial_rate",
        "Lateral_rate",
        "M-L",
    ]
)

df.to_csv("sclerosis_df.csv", index=False)
