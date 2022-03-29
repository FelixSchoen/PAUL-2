import matplotlib.pyplot as plt
import numpy as np
from sCoda import Bar


def get_message_lengths_and_difficulties(bars: [([Bar], [Bar])]):
    messages = []
    difficulties = []

    ungute_bars = 0
    overall = 0

    for bar_tuple in bars:
        bars = bar_tuple[0]
        bars.extend(bar_tuple[1])

        for bar in bars:
            overall += 1
            amount_messages = len(bar._sequence._get_rel().messages)
            messages.append(amount_messages)
            difficulties.append(bar._difficulty)
            if bar._difficulty is None:
                ungute_bars += 1

    print(f"Ungute bars {ungute_bars}, overall bars {overall}")

    plt.hist(messages, np.linspace(0, 1024, 256))
    plt.show()

    plt.hist(difficulties, np.linspace(0, 1, 100))
    plt.show()
