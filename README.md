# PAUL-2

This repository contains the practical implementation of our work PAUL-2.
PAUL-2 is a [transformer](https://arxiv.org/abs/1706.03762) based algorithmic composer utilising the enhancements made by the [Music Transformer](https://arxiv.org/abs/1809.04281).

PAUL-2 is capable of composing two-track piano pieces.
The distinguishing feature of the composer is its ability to compose pieces based on a ''difficulty'' parameter, defining how difficulty an output piece should be to play for a human pianist.

We refer to the [full thesis](https://repositum.tuwien.at/handle/20.500.12708/139690) for more information on PAUL-2.

In 2023 we published a [paper](https://doi.org/10.1007/978-3-031-47546-7_19) on our work at the 22nd International conference of the Italian Association for Artificial Intelligence (AIxIA).

## Requirements

- [s-coda](https://github.com/FelixSchoen/S-Coda) 1.0
- tensorflow 2.8
- numpy 1.22
- pandas 1.4
- mido 1.2