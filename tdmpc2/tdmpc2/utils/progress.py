from tqdm import tqdm as _tqdm
import os
from typing import Optional
from tdmpc2.utils.time import get_current_datetime_str
from datetime import timedelta

NO_TQDM = int(os.environ.get("NO_TQDM", "0"))
# PRINT_INTERVAL = int(os.environ.get("PRINT_INTERVAL", "1"))


def format_timedelta(td: timedelta) -> str:
    total_seconds = int(td.total_seconds())
    days, remainder = divmod(total_seconds, 86400)  # 86400 seconds in a day
    hours, remainder = divmod(remainder, 3600)  # 3600 seconds in an hour
    minutes, seconds = divmod(remainder, 60)  # 60 seconds in a minute

    return f"{days}d{hours}h{minutes}m{seconds}s"


def comp_end_time(total: int, comped: int, chunked, elapsed_time: timedelta):
    return format_timedelta((total - comped) / chunked * elapsed_time)


def tqdm(iter, desc: str):
    if not NO_TQDM:
        return _tqdm(iter, desc=desc)

    class CLITqdm:
        def __init__(self, iter, desc: str) -> None:
            self._iter = iter
            self._desc = desc
            self._print_interval = NO_TQDM
            self._i = 0
            self._total = len(iter)

        def __iter__(self):
            _sts, self._st = get_current_datetime_str(no_micro=True)
            self._sst = self._st
            print(
                f"----CLI-tqdm: {self._desc} started at {_sts}",
                flush=True,
            )
            for ele in self._iter:
                if self._i > 0 and self._i % self._print_interval == 0:
                    _ets, self._et = get_current_datetime_str(no_micro=True)
                    print(
                        f"----CLI-tqdm: {self._i}th of {self._desc} started at {_ets}, remaining time: {comp_end_time(self._total, self._i,self._print_interval, elapsed_time=self._et-self._st)}",
                        flush=True,
                    )
                    self._st = self._et
                yield ele
                self._i += 1
            _ets, self._et = get_current_datetime_str(no_micro=True)
            print(
                f"----CLI-tqdm: {self._desc} ended at {_ets}, in total {format_timedelta(self._et - self._sst)} time passed",
                flush=True,
            )

    return CLITqdm(iter, desc=desc)


class TqdmBar:
    def __init__(self, total: int, desc: str):
        if not NO_TQDM:
            self._tqdm = _tqdm(total=total, desc=desc)
        else:
            self._tqdm = None
        self._desc = desc
        self._print_interval = NO_TQDM
        self._i = 0
        self._total = total

    def __enter__(self):
        if self._tqdm:
            self._tqdm.__enter__()
            return self._tqdm

        _sts, self._st = get_current_datetime_str(no_micro=True)
        self._sst = self._st
        print(
            f"----CLI-tqdm bar: {self._desc} started at {_sts}",
            flush=True,
        )
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        if self._tqdm:
            self._tqdm.__exit__(exception_type, exception_value, exception_traceback)
            return

        _ets, self._et = get_current_datetime_str(no_micro=True)
        print(
            f"----CLI-tqdm bar: {self._desc} ended at {_ets}, in total {format_timedelta(self._et - self._sst)} time passed",
            flush=True,
        )

    def update(self, *args, **kwargs):
        if self._tqdm:
            self._tqdm.update(*args, **kwargs)
            return

        if self._i > 0 and self._i % self._print_interval == 0:
            _ets, self._et = get_current_datetime_str(no_micro=True)
            print(
                f"----CLI-tqdm bar: {self._i}th of {self._desc} end at {_ets}, remaining time: {comp_end_time(self._total, self._i, self._print_interval, elapsed_time=(self._et-self._st))}",
                flush=True,
            )
            self._st = self._et
        self._i += 1


if __name__ == "__main__":
    n = 0
    for i in tqdm(range(int(1e7)), desc="training..."):
        n += i * i
    print(n)

    n = 0
    with TqdmBar(total=int(1e7), desc="training in bar...") as pbar:
        for i in range(int(1e7)):
            n += i * i
            pbar.update(1)
    print(n)
