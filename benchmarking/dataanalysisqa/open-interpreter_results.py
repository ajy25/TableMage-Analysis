from interpreter import interpreter
from pathlib import Path
import pandas as pd
import time

iternums = list(range(5, 10))
# set subset of datasets to generate answers for (must have already ran on all datasets)
# leave empty to generate answers for all datasets
subset = []

curr_dir = Path(__file__).parent

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
output_dir.mkdir(exist_ok=True, parents=True)


dataset_name_to_filestem = {
    "Titanic": "titanic.csv",
    "House Prices": "house_prices.csv",
    "Wine": "wine.csv",
    "Breast Cancer": "breast_cancer.csv",
    "Credit": "credit.csv",
    "Abalone": "abalone.csv",
    "Baseball": "baseball.csv",
    "Auto MPG": "autompg.csv",
    "Healthcare": "simulated_healthcare.csv",
    "Iris": "iris.csv",
}


rules = """\
I am providing you with a dataset located at path {file_path}. \
I will then give you instructions for analyzing the dataset.

Here are some rules: \
(1) Immediately split the dataset into 80/20 train/test sets using sklearn’s train_test_split function, with random seed 42. \
(2) If you are explicitly asked to transform the dataset (e.g., scaling, imputation, feature engineering), keep the changes for future questions. \
(3) When transforming data (e.g., feature scaling), always fit on the train dataset and transform the test dataset based on the train dataset. \
(4) When fitting models, always fit on the train dataset and predict on the test dataset. \
(5) For exploratory analysis (e.g., statistical testing, summary statistics), always consider the entire dataset. \
(6) Temporarily drop rows with missing values in variables of interest prior to each analysis step. \
(7) Return a sentence for each query describing your findings, round numeric answers to 3 decimal places. \
(8) Use significance level 0.05 for statistical tests. \
"""


for iternum in iternums:
    # read in the questions
    questions_df = pd.read_csv(curr_dir / "dataanalysisqa.tsv", sep="\t")
    results_df = None
    if len(subset) != 0 and len(subset) < len(dataset_name_to_filestem):
        # try to read in full output
        results_df = pd.read_csv(output_dir / f"answers_{iternum}.csv")
        dataset_name_to_filestem = {
            k: v for k, v in dataset_name_to_filestem.items() if k in subset
        }

    question_ids = []
    unformatted_answers = []

    for dataset_name in dataset_name_to_filestem.keys():
        print(f"Generating answers for {dataset_name} dataset using TableMage...")

        questions_subset_df = questions_df[questions_df["Dataset"] == dataset_name]
        n_questions = len(questions_subset_df)

        # for each dataset, reset the chat history
        interpreter.messages = []
        interpreter.chat(
            rules.format(
                file_path=str(
                    curr_dir / "datasets" / f"{dataset_name_to_filestem[dataset_name]}"
                )
            )
        )

        for i in range(1, n_questions + 1):
            question_id = questions_subset_df[
                questions_subset_df["Question Order"] == i
            ]["Question ID"].values[0]

            question_ids.append(question_id)

            question = questions_subset_df[questions_subset_df["Question Order"] == i][
                "Question"
            ].values[0]

            try:
                answer = interpreter.chat(question)
                unformatted_answers.append(answer)
            except Exception as e:
                answer = f"Error occurred: {e}. No answer generated."
                unformatted_answers.append(answer)

            print(
                f"\n\n\nQuestion ID: {i}\nQuestion: {question}\nAnswer: {answer}\n\n\n"
            )

            # wait to avoid rate limiting
            time.sleep(2.0)

    if results_df is not None:
        # if we're working with a subset, update only the rows that have been updated
        for i, question_id in enumerate(question_ids):
            results_df.loc[
                results_df["Question ID"] == question_id, "Unformatted Answer"
            ] = unformatted_answers[i]

    else:
        results_df = pd.DataFrame(
            {
                "Question ID": question_ids,
                "Unformatted Answer": unformatted_answers,
            }
        )

    results_df.to_csv(output_dir / f"answers_{iternum}.csv", index=False)
