import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sCoda import Sequence


def test_print_confusion_matrix_paul_paper():
    array = [[53, 31, 0],
             [13, 56, 15],
             [0, 5, 58]]
    df_cm = pd.DataFrame(array, index=["EASY", "MEDIUM", "HARD"],
                         columns=["EASY", "MEDIUM", "HARD"])
    sn.heatmap(df_cm, annot=True, cmap="gray_r")

    plt.xlabel("Evaluated Class", fontsize=16, labelpad=15)
    plt.ylabel("True Class", fontsize=16, labelpad=15)

    plt.tight_layout()
    plt.savefig("matrix.svg", format="svg")


def test_print_activation_functions():
    plt = plot_function(threshold_function, [-6, 6], None, "Input Value", "Output Value")
    plt.savefig("../out/neural_networks/threshold.pdf")

    plt = plot_function(sigmoid_function, [-6, 6], None, "Input Value", "Output Value")
    plt.savefig("../out/neural_networks/sigmoid.pdf")

    plt = plot_function(relu_function, [-2, 2], None, "Input Value", "Output Value")
    plt.savefig("../out/neural_networks/relu.pdf")


def test_plot_validation_graphs():
    xlim = [0, 51]
    ylim = [0.8, 3]

    plt = plot_csv("../out/paul_validation_graphs/validation_lead.csv", ["Step", "Value"], xlim, ylim,
                   x_label="Checkpoint", y_label="Loss")
    plt.savefig("../out/paul_validation_graphs/validation_lead.pdf")

    plt = plot_csv("../out/paul_validation_graphs/validation_acmp.csv", ["Step", "Value"], xlim, ylim,
                   x_label="Checkpoint", y_label="Loss")
    plt.savefig("../out/paul_validation_graphs/validation_acmp.pdf")


def test_plot_confusion_matrix():
    df = pd.read_pickle("../out/paul_confusion_matrix/df_lead_1.pkl")
    plot_heatmap(df, "actual", "output", "Generated Difficulty", "Specified Difficulty").savefig(
        "../out/paul_confusion_matrix/confusion_lead_1bar.pdf")

    df = pd.read_pickle("../out/paul_confusion_matrix/df_acmp_1.pkl")
    plot_heatmap(df, "actual", "output", "Generated Difficulty", "Specified Difficulty").savefig(
        "../out/paul_confusion_matrix/confusion_acmp_1bar.pdf")


def test_plot_pianorolls():
    for d in [1, 4, 7]:
        for i in [1, 2, 3]:
            lead_seq = Sequence.from_midi_file(f"../out/results_pianorolls/l_d{d}_{i}.mid")[0]
            acmp_seq = Sequence.from_midi_file(f"../out/results_pianorolls/a_d{d}_{i}.mid")[0]

            print(lead_seq.rel.guess_key_signature())

            Sequence.pianorolls([lead_seq, acmp_seq], x_label="Time in Ticks", y_label="Notes").savefig(f"../out/results_pianorolls/d{d}_{i}.pdf")


def plot_function(func, x_limits, y_limits, x_label, y_label, x_steps=10000):
    x = np.linspace(x_limits[0], x_limits[1], x_steps)

    fig = plt.figure()
    plt.plot(x, wrap_func(func, x), color="black")

    plt.xlim(x_limits)
    plt.ylim(y_limits)

    plt.xlabel(x_label, fontsize=16, labelpad=15)
    plt.ylabel(y_label, fontsize=16, labelpad=15)

    return fig


def plot_csv(file_path, columns, x_limits, y_limits, x_label, y_label):
    fig = plt.figure()

    df = pd.read_csv(file_path, usecols=columns)
    plt.plot(df.Step, df.Value, color="black")

    plt.xlim(x_limits)
    plt.ylim(y_limits)

    plt.xlabel(x_label, fontsize=16, labelpad=15)
    plt.ylabel(y_label, fontsize=16, labelpad=15)

    plt.tight_layout()

    return fig


def plot_heatmap(dataframe, name_truth, name_prediction, xlabel, ylabel):
    fig = plt.figure()

    confusion_matrix = pd.crosstab(dataframe[name_truth], dataframe[name_prediction])

    sn.heatmap(confusion_matrix, annot=True, cmap="Greys")

    plt.xlabel(xlabel, fontsize=16, labelpad=15)
    plt.ylabel(ylabel, fontsize=16, labelpad=15)
    plt.tight_layout()

    return fig


def wrap_func(func, values):
    results = []
    for i in values:
        results.append(func(i))
    return results


def threshold_function(x):
    if x <= 0:
        return 0
    else:
        return 1


def sigmoid_function(x):
    return 1 / (1 + np.power(np.e, -x))


def relu_function(x):
    return max(0, x)
