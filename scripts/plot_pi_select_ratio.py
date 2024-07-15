import os
from os import path
from collections import defaultdict
from typing import Literal


def parse_line(line: str):
    _si = line.index("[")
    _ei = line.index("]")
    _content = line[_si + 1 : _ei]
    _content = list(map(float, _content.split(",")))

    return _content


def parse_file(file_name: str):
    ratios = {"eval": {}, "train": []}
    i = 0
    _i_has_incremented = True
    with open(file_name, "r") as f:
        for line in f:
            if "pi_selected_ratio" not in line:
                continue
            # ratios.append(line)
            _rlt = parse_line(line)
            if "eval" in line:
                if i not in ratios["eval"]:
                    ratios["eval"][i] = []
                ratios["eval"][i].append(_rlt)
                _i_has_incremented = True
            else:
                if _i_has_incremented:
                    i += 1
                    _i_has_incremented = False
                assert "train" in line
                ratios["train"].append(_rlt)
    print(f"in {file_name}, ratios['train'] has len {len(ratios['train'])}")
    return ratios


TASKS = [
    "walk",
    "pole",
    "reach",
    "balance_simple",
    "crawl",
    "cube",
    "sit_hard",
    "maze",
    "insert_normal",
]


def plot(rlts, variant: Literal["train", "eval"]):
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import random
    from tqdm import tqdm
    from os import path
    import os

    np.random.seed(1111)
    random.seed(1111)

    f, axes = plt.subplots(3, 3, figsize=(8, 8))
    f.subplots_adjust(hspace=0.45, wspace=0.3)
    sns.set_theme()
    for ti, task in tqdm(enumerate(TASKS), desc="processing file"):
        _rlts = rlts[task][variant]
        _skip = 10 if variant == "train" else 8

        if variant == "eval":

            _rlts = [_step for _eval in _rlts.values() for _step in _eval]

        def _get_j_column_of_rlts(j: int):
            return [r[j] for i, r in enumerate(_rlts) if i % _skip == 0]

        df = pd.DataFrame(columns=["x", "j", "ratio"])
        _li = 0
        for j in [0, 1, 2, 5, 7]:
            _vals = _get_j_column_of_rlts(j)
            for x, v in enumerate(_vals):
                df.loc[_li] = [x, j, (v - 0) / (24 / 64 - 0)]
                _li += 1
        sns.lineplot(
            data=df,
            x="x",
            y="ratio",
            hue="j",
            palette=["#d4a373", "#faedcd", "#fefae0", "#e9edc9", "#ccd5ae"],
            ax=axes[ti // 3, ti % 3],
        )
        axes[ti // 3, ti % 3].title.set_text(task)
        axes[ti // 3, ti % 3].get_legend().remove()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    f.legend(
        [plt.plot([], marker="", ls="")[0]] + handles,
        ["mpc iteration: "] + labels,
        loc="upper center",
        ncol=9,
        # title='mpc iteration',
        bbox_to_anchor=(1.87, 1.42),
        bbox_transform=axes[0, 0].transAxes,
        # bbox_inches="tight",
    )
    plt.tight_layout(pad=1.01, rect=(0, 0, 1, 0.96))
    plt.savefig(f"./tmp_ratio_{variant}.png")


if __name__ == "__main__":
    file_pi_selected_ratios = {}
    for r, ds, fs in os.walk("./logdir"):
        for f in fs:
            if f.endswith(".log"):
                _si = f.index("-")
                _ei = f.index("-", _si + 1)
                print(f"ready to parse {r}/{f}")
                _rlt = parse_file(f"{r}/{f}")
                file_pi_selected_ratios[f[_si + 1 : _ei]] = _rlt
    plot(file_pi_selected_ratios, "train")
    plot(file_pi_selected_ratios, "eval")
