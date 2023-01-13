from typing import TypeVar, Iterable, Callable, Sequence
from collections import defaultdict

K, V = TypeVar("K"), TypeVar("V")


def bin_key_value_list(key_value_pairs: Iterable[tuple[K, V]]) -> dict[K, list[V]]:
    binned_dict = defaultdict(list)
    for k, v in key_value_pairs:
        binned_dict[k].append(v)
    return binned_dict


def filter_dict(
    data_dict: dict[K, V], rejection_criteria: Sequence[Callable[[V], bool]]
) -> dict[K, V]:
    def filter_func(data: V) -> bool:
        return any(c(data) for c in rejection_criteria)

    return {k: v for k, v in data_dict.items() if not filter_func(v)}
