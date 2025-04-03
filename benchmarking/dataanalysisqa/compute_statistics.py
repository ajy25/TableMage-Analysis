from pathlib import Path
import pandas as pd

curr_dir = Path(__file__).parent.resolve()

methods = [
    "ada_gpt4o",
    "di_gpt4o",
    "tablemage_gpt4o",
    "tablemage_llama3.3_70b",
    "openinterpreter_gpt4o",
    "tablemage_gpt4o-tools_only",
]

without_indexing = True

label_df = pd.read_csv(curr_dir / "results" / "labels.tsv", sep="\t")
questions_df = pd.read_csv(curr_dir / "dataanalysisqa.tsv", sep="\t")

categories = [
    "All",
    "Summary Statistics",
    "Statistical Testing",
    "Regression Analysis",
    "Persistence",
    "Data Transformation",
    "Indexing",
]
if without_indexing:
    categories = categories[:-1]


for category in categories:
    output_df = {}
    failure_df = {}

    for method in methods:
        all_answer_paths = sorted(
            [
                path
                for path in (curr_dir / "results" / method / "formatted").glob("*.tsv")
                if path.stem.startswith("formatted-answers")
            ]
        )

        # figure out how many questions were correct for each file
        accuracies = []
        failures = []
        for answer_path in all_answer_paths:
            answer_df = pd.read_csv(answer_path, sep="\t")
            merged_df = pd.merge(answer_df, label_df, on="Question ID")
            merged_df = pd.merge(merged_df, questions_df, on="Question ID")

            if without_indexing:
                merged_df = merged_df[~merged_df["Category"].str.contains("Indexing")]
            print(len(merged_df))

            if category != "All":
                # filter df, check if column "Category" contains value category
                merged_df = merged_df[merged_df["Category"].str.contains(category)]

            num_correct = (merged_df["Formatted Answer"] == merged_df["Label"]).sum()
            num_failed = merged_df["Formatted Answer"].str.contains("MISSING").sum()
            accuracies.append(num_correct / len(merged_df))
            failures.append(num_failed / len(merged_df))

        mean_correct = sum(accuracies) / len(accuracies)
        std_correct = pd.Series(accuracies).std()

        print(f"{method}: {mean_correct:.1f} ± {std_correct:.1f}")

        ci = 1.96 * std_correct / len(accuracies) ** 0.5

        output_df[method] = {"mean": mean_correct, "std": std_correct, "ci": ci}

        failure_df[method] = {
            "mean": sum(failures) / len(failures),
            "std": pd.Series(failures).std(),
            "ci": 1.96 * pd.Series(failures).std() / len(failures) ** 0.5,
        }

    output_df = pd.DataFrame(output_df).T

    failure_df = pd.DataFrame(failure_df).T
    if without_indexing:
        output_df.to_csv(
            curr_dir
            / "results"
            / "statistics-no_index"
            / "accuracy"
            / f"accuracy_summary-{category}-no_index.csv"
        )
    else:
        output_df.to_csv(
            curr_dir
            / "results"
            / "statistics"
            / "accuracy"
            / f"accuracy_summary-{category}.csv"
        )
        failure_df.to_csv(
            curr_dir
            / "results"
            / "statistics"
            / "failure"
            / f"failure_summary-{category}.csv"
        )
