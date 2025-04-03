from pathlib import Path
import pandas as pd
import time
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate


curr_dir = Path(__file__).resolve().parent


results_folder = curr_dir / "results" / "openinterpreter_gpt4o"  # change appropriately

iternums = list(range(10))


(results_folder / "formatted").mkdir(exist_ok=True)


for iternum in iternums:
    results_unformatted_df = pd.read_csv(results_folder / f"answers_{iternum}.csv")

    # read in the API key
    api_key = curr_dir / "api_key.txt"
    with open(api_key, "r") as f:
        api_key = f.read().strip()

    llm = OpenAI(model="gpt-4o-mini", temperature=0.0, api_key=api_key)

    initial_prompt_template = PromptTemplate(
        template="""\
Someone trained a machine learning model. \
They reported the test RMSE or test AUROC of their model. \
Your job is to extract the metric they reported. \
If no metric is reported, return "NaN". \

-- Example 1 --
Unformatted Answer: The best model I trained was a CatBoost model, which had a test RMSE of 0.243.
Model Response: 0.243

-- Example 2 --
Unformatted Answer: I trained several machine learning models, and random forest was best.
Model Response: NaN

-- Example 3 --
Unformatted Answer: 0.85
Model Response: 0.85

-- Your Turn --
Unformatted Answer: {answer}
Model Response: 
"""
    )

    dataset_ids = []
    formatted_answers = []

    for row in results_unformatted_df.iterrows():
        dataset_id = row[1]["Dataset ID"]
        dataset_id = int(dataset_id)
        answer = row[1]["Unformatted Answer"]
        print(f"Dataset ID: {dataset_id}")
        print(f"Answer: {answer}")

        output = llm.complete(initial_prompt_template.format(answer=answer))
        output = str(output)
        print(f"Formatted Output: {output}")

        dataset_ids.append(dataset_id)
        formatted_answers.append(output)

        time.sleep(0.5)

    formatted_answers_df = pd.DataFrame(
        {"Dataset ID": dataset_ids, "Formatted Answer": formatted_answers}
    )
    formatted_answers_df["Dataset ID"] = formatted_answers_df["Dataset ID"].astype(str)
    formatted_answers_df.to_csv(
        results_folder / "formatted" / f"formatted-answers_{iternum}.tsv",
        sep="\t",
        index=False,
    )
