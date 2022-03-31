import matplotlib.pyplot as plt
import numpy as np
from sCoda import Bar


def get_message_lengths_and_difficulties(input_bars: [([Bar], [Bar])]):
    messages = []
    difficulties = []

    for bar_tuple in input_bars:
        bars = bar_tuple[0]
        bars.extend(bar_tuple[1])

        for bar in bars:
            amount_messages = len(bar._sequence._get_rel().messages)
            messages.append(amount_messages)
            difficulties.append(bar._difficulty)

    plt.hist(messages, np.linspace(0, 1024, 256))
    plt.show()

    plt.hist(difficulties, np.linspace(0, 1, 100))
    plt.show()
