import matplotlib.pyplot as plt
import numpy as np
from sCoda import Bar


def get_message_lengths_and_difficulties(input_bars: [([Bar], [Bar])]):
    messages = []
    difficulties = []

    for bar_tuple in input_bars:
        local_messages = []

        for bar in bar_tuple[0]:
            amount_messages = len(bar._sequence._get_rel().messages)
            local_messages.append(amount_messages)
            difficulties.append(bar._difficulty)

        messages.append(sum(local_messages))
        local_messages = []

        for bar in bar_tuple[1]:
            amount_messages = len(bar._sequence._get_rel().messages)
            local_messages.append(amount_messages)
            difficulties.append(bar._difficulty)

        messages.append(sum(local_messages))

    plt.hist(messages, bins=np.arange(min(messages), max(messages) + 5, 5))
    plt.show()

    plt.hist(difficulties, np.linspace(0, 1, 100))
    plt.show()
