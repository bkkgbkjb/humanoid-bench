from datetime import datetime
from typing import Optional, Tuple


def get_current_datetime_str(no_micro: Optional[bool] = False) -> Tuple[str, datetime]:
    _time = datetime.now()
    return (
        _time.strftime("%m-%d:%H:%M:%S:%f" if not no_micro else "%m-%d:%H:%M:%S"),
        _time,
    )


def get_current_ms() -> Tuple[str, datetime]:
    _time = datetime.now()
    return _time.strftime("%f"), _time


start_time_str, start_time = get_current_datetime_str()


class Timeit:
    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self):
        self.start_cpu = datetime.now()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed_time_cpu = datetime.now() - self.start_cpu
        print(
            f"Timeit: {self.name.capitalize()} costs: {self.elapsed_time_cpu.total_seconds()}s"
        )
