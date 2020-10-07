from collections import OrderedDict
from crayons import cyan, white, red
from tqdm import tqdm

import numpy as np


class LossLogger:
    def __init__(self):
        # Build state dict
        self.losses = OrderedDict({"Epoch": 0})

        self.are_headers_logged = False

    def heading(self, text):
        pass

    def add_loss(self, key, value):
        if key not in self.losses:
            self.losses[key] = [value.item()]
        else:
            self.losses[key].append(value.item())

    def flush(self):
        if not self.are_headers_logged:
            self.are_headers_logged = True
            self.loss_format = "{:>15}" * len(self.losses)
            tqdm.write(
                str(white(self.loss_format.format(*self.losses.keys()), bold=True))
            )

        tqdm.write(
            self.loss_format.format(
                f'{self.losses["Epoch"]}',
                *[
                    f"{np.mean(np.array(v)):.3f}"
                    for _, v in list(self.losses.items())[1:]
                ],
            )
        )

        for k in self.losses.keys():
            if k == "Epoch":
                self.losses[k] += 1
                continue

            self.losses[k] = []
