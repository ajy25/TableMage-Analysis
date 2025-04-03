import pandas as pd
from pathlib import Path

curr_dir = Path(__file__).parent.resolve()
all_answers_df = curr_dir / "all_answers.csv"

answer_ids = range(1, 11)

for answer_id in answer_ids:
    df = pd.read_csv(all_answers_df)
    if str(answer_id) not in df.columns.astype(str):
        print(answer_id, "not in df.columns")
        continue
    df = df[["Question ID", f"{answer_id}"]]
    df = df.rename(columns={f"{answer_id}": "Unformatted Answer"})
    print(
        "Saving answer",
        answer_id,
        "to separate file",
        curr_dir / f"answer_{answer_id-1}.csv",
    )
    df.to_csv(curr_dir / f"answers_{answer_id-1}.csv", index=False)
