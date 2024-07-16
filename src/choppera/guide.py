# SPDX-FileCopyrightText: 2024-present Gregory Tucker <gregory.tucker@ess.eu>
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from numpy import ndarray
from typing import Callable, Tuple, List, Collection



def interpolate(positions: ndarray, values: ndarray, at: float) -> float:
    from numpy import searchsorted
    second = searchsorted(positions, at)
    t = (at - positions[second - 1]) / (positions[second] - positions[second - 1])
    return values[second - 1] + t * (values[second] - values[second - 1])


@dataclass
class Guide:
    name: str
    velocity: Tuple[float, ...]
    shortest: Tuple[float, ...]
    longest: Tuple[float, ...]
    nominal: Tuple[float, ...] = (-1.,)

    # this would be better as __post_init__ but that doesn't get called?
    def __init__(self, name, velocity, shortest, longest, nominal=(-1.,)):
        self.name = name
        self.velocity = velocity
        self.shortest = shortest
        self.longest = longest
        self.nominal = nominal
        assert len(self.velocity) == len(self.shortest) and len(self.shortest) == len(self.longest)
        if any([s > l for s, l in zip(self.shortest, self.longest)]):
            raise RuntimeError("shortest path lengths should not be longer than longest path lengths")
        if self.nominal[0] < 0.:
            self.nominal = tuple([(s + l) / 2 for s, l in zip(self.shortest, self.longest)])
        if not all([s <= n <= l for s, n, l in zip(self.shortest, self.nominal, self.longest)]):
            raise RuntimeError("nominal path length should be bounded by shortest and longest paths")
        for i in range(len(self.velocity) - 1):
            assert self.velocity[i] < self.velocity[i + 1]

    def __str__(self):
        return f"Guide[{self.name}]"

    def propagate(self, distribution: List[Tuple[ndarray, ndarray, ndarray]]) -> List[Tuple[ndarray, ndarray, ndarray]]:
        from numpy import array, allclose, vstack, min, max
        out = []
        short, long, v = array(self.shortest), array(self.longest), array(self.velocity)
        short /= v
        long /= v
        for vel, early, late in distribution:
            assert allclose(v, vel)
            times = vstack((early + short, early + long, late + short, late + long))
            out.append((vel, min(times, axis=0), max(times, axis=0)))
        return out

    def tinv_transforms(self, pre=0., post=0.):
        # These are, strictly speaking, fully dependent on v. However, that seems too difficult for now
        return pre + min(self.shortest), post + max(self.longest)

    def phase_length(self, target_velocity):
        from numpy import array
        return interpolate(array(self.velocity), array(self.nominal), target_velocity)
