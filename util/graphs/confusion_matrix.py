import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn


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
