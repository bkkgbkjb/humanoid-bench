import os
from os import path
from collections import defaultdict
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm
from pathlib import Path
import json


def parse_file(file_name: str):
    ea = event_accumulator.EventAccumulator(
        file_name,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()
    eval_returns = ea.Scalars("eval/episode_reward")
    return eval_returns


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


def plot(rlts):
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import random
    from tqdm import tqdm
    from os import path
    import os

    with open("./logs/main_results.json", "r") as json_file:
        main_rlt = json.load(json_file)

    f, axes = plt.subplots(3, 3, figsize=(12, 8))
    # f.tight_layout()
    f.subplots_adjust(hspace=0.45, wspace=0.35)
    for ti, task in enumerate(tqdm(TASKS, desc="processing file")):

        np.random.seed(1111)
        random.seed(1111)

        if not path.exists(f"./tmp/return_csv/{task}.csv"):
            _rlts = rlts[task]
            Path("./tmp/return_csv").mkdir(exist_ok=True, parents=True)
            df = pd.DataFrame(
                {
                    "return": [r.value for r in _rlts],
                    "x": [_x / len(_rlts) * 2 for _x in list(range(len(_rlts)))],
                    "algo": "ours",
                    "seed": "11111",
                }
            )
            for _algos in ["SAC", "DreamerV3", "TD-MPC2"]:
                for _seed in ["0", "1", "2"]:
                    df = pd.concat(
                        [
                            df,
                            pd.DataFrame(
                                {
                                    "return": main_rlt[task][_algos][f"seed_{_seed}"][
                                        "return"
                                    ],
                                    "x": main_rlt[task][_algos][f"seed_{_seed}"][
                                        "million_steps"
                                    ],
                                    "algo": _algos,
                                    "seed": _seed,
                                }
                            ),
                        ],
                        ignore_index=True,
                    )
            df.to_csv(f"./tmp/return_csv/{task}.csv", index=False)
        else:
            df = pd.read_csv(f"./tmp/return_csv/{task}.csv")

        sns.set_theme()
        sns.lineplot(
            data=df,
            x="x",
            y="return",
            ax=axes[ti // 3, ti % 3],
            hue="algo",
            palette=["#26547c", "#ef476f", "#fb8500", "#06d6a0"],
        )
        axes[ti // 3, ti % 3].title.set_text(task)
        axes[ti // 3, ti % 3].get_legend().remove()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    f.legend(
        handles,
        labels,
        loc="upper center",
        ncol=9,
        bbox_to_anchor=(1.87, 1.42),
        bbox_transform=axes[0, 0].transAxes,
        # bbox_inches="tight",
    )
    plt.tight_layout(pad=1.01, rect=(0, 0, 1, 0.96))
    plt.savefig("./tmp_return.png")


if __name__ == "__main__":
    file_eval_returns = {}
    if not path.exists("./tmp/return_csv"):
        with tqdm(total=len(TASKS), desc="process tasks") as pbar:
            for r, ds, fs in os.walk("./outputs"):
                for f in fs:
                    if "events" in f:
                        assert sum(list(map(lambda t: t in r, TASKS))) == 1
                        _task = TASKS[list(map(lambda t: t in r, TASKS)).index(True)]
                        _rlt = parse_file(f"{r}/{f}")
                        file_eval_returns[_task] = _rlt
                        pbar.update(1)

    plot(file_eval_returns)
