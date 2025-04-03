import pandas as pd
from pathlib import Path

curr_dir = Path(__file__).parent.resolve()
all_answers_df = curr_dir / "all_answers.csv"


answer_ids = range(1, 11)

for answer_id in answer_ids:
    df = pd.read_csv(all_answers_df)
    df = df[["Dataset ID", f"{answer_id}"]]
    df = df.rename(columns={f"{answer_id}": "Unformatted Answer"})
    df.to_csv(curr_dir / f"answers_{answer_id-1}.csv", index=False)
