import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics

correct_answers = {"q1": "Stück 2",
                   "q2": "Stück 2",
                   "q3": "Stück 1",
                   "q4": "Stück 1",
                   "q5": "Stück 2",
                   "q6": "Stück 1",
                   "q7": "Stück 1",
                   "q8": "Stück 1",
                   "q9": "Stück 2",
                   "q10": "Stück 1",
                   "q11": "Stück 2",
                   "q12": "Stück 1",
                   "q13": "Stück 1",
                   "q14": "Stück 2",
                   "q15": "Stück 1",
                   "q16": "Stück 2",
                   "q17": "Stück 1",
                   "q18": "Stück 1"}


def main():
    df = pd.read_csv("../resources/survey_results.csv")
    df.drop(df.columns[[0]], axis=1, inplace=True)

    for index, row in df.iterrows():
        for i, x in enumerate(row.iteritems()):
            if i < 18:
                if x[1] == correct_answers[x[0]]:
                    df.iat[index, i] = 1
                elif x[1][0] == "S":
                    df.iat[index, i] = 0
                else:
                    df.iat[index, i] = -1

    results_per_level = [[] for _ in range(5)]
    results_per_difficulty = [[] for _ in range(3)]
    quality_per_level = [[] for _ in range(5)]
    confidence_per_level = [[] for _ in range(5)]

    for index, row in df.iterrows():
        level = row.iat[-1]
        for i in range(18):
            if row.iat[i] != -1:
                results_per_level[level - 1].append(row.iat[i])
                results_per_difficulty[i % 3].append(row.iat[i])
        quality_per_level[level - 1].append(row.iat[-2])
        confidence_per_level[level - 1].append(row.iat[-3])

    results_overall = list(itertools.chain.from_iterable(results_per_level))
    quality_overall = list(itertools.chain.from_iterable(quality_per_level))
    confidence_overall = list(itertools.chain.from_iterable(confidence_per_level))

    for i in range(5):
        rpd = results_per_level[i]
        qpl = quality_per_level[i]
        cpl = confidence_per_level[i]
        print(
            f"Level: {i + 1}, Proportion: {sum(rpd) / len(rpd):.2f}, Qual Mean: {statistics.mean(qpl):2f}, Qual Std: {statistics.stdev(qpl):2f}, "
            f"Conf Mean: {statistics.mean(cpl):2f}, Conf Std: {statistics.stdev(cpl):2f} (Samples {len(cpl)})")

    print()

    print(
        f"Overall, Proportion: {sum(results_overall) / len(results_overall):.2f}, Qual Mean: {statistics.mean(quality_overall):2f}, Qual Std: {statistics.stdev(quality_overall):2f}, "
        f"Conf Mean: {statistics.mean(confidence_overall):2f}, Conf Std: {statistics.stdev(confidence_overall):2f} (Samples {len(confidence_overall)})")

    print()

    for i in range(3):
        rpd = results_per_difficulty[i]
        print(f"Difficulty: {i}, Proportion: {sum(rpd) / len(rpd):.2f} (Samples {len(rpd)})")

    correctly_classified = [0 for _ in range(18)]
    incorrectly_classified = [0 for _ in range(18)]
    invalid = [0 for _ in range(18)]

    for index, row in df.iterrows():
        for i in range(18):
            if row.iat[i] == 1:
                correctly_classified[i] = correctly_classified[i] + 1
            if row.iat[i] == 0:
                incorrectly_classified[i] = incorrectly_classified[i] + 1
            if row.iat[i] == -1:
                invalid[i] = invalid[i] + 1

    labels = [f"Sample {x + 1}" for x in range(18)]
    width = 0.25

    for pair in [(0, 9, "lead"), (9, 18, "acmp")]:
        x = np.arange(start=0, stop=len(labels[pair[0]:pair[1]]), step=1)
        fig, ax = plt.subplots()
        fig.set_figwidth(12)
        ax.bar(x - width, correctly_classified[pair[0]:pair[1]], width, label="Correctly classified", hatch="\\",
               edgecolor="black",
               color='0.8')
        ax.bar(x, incorrectly_classified[pair[0]:pair[1]], width, label="Incorrectly classified", hatch="/",
               edgecolor="black",
               color='0.5')
        ax.bar(x + width, invalid[pair[0]:pair[1]], width, label="Invalid answer", hatch="||", edgecolor="black",
               color='0.2')
        ax.set_xticks(x, labels[pair[0]:pair[1]])
        ax.set_ylabel("Amount of Answers")
        ax.set_ylim([0, 55])
        ax.set_yticks(range(0, 55, 10))
        ax.legend()
        fig.tight_layout()
        plt.savefig(f"../out/survey_bar/survey_result_bar_{pair[2]}.pdf")


main()
