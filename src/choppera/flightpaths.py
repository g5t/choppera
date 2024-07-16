# SPDX-FileCopyrightText: 2024-present Gregory Tucker <gregory.tucker@ess.eu>
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import List, Tuple
from numpy import ndarray
from polystar import Polygon


def interpolate(positions: ndarray, values: ndarray, at: float) -> float:
    from numpy import searchsorted
    second = searchsorted(positions, at)
    t = (at - positions[second - 1]) / (positions[second] - positions[second - 1])
    return values[second - 1] + t * (values[second] - values[second - 1])


@dataclass
class FlightPath:
    name: str
    velocity: Tuple[float, ...] = (1e-9, 1e9)
    nominal: Tuple[float, ...] = (-1., )

    def __post_init__(self):
        from numpy import std
        if len(self.velocity) < 2 or std(self.velocity) == 0.:
            raise RuntimeError("Multiple unique velocities required")
        if self.nominal[0] < 0.:
            self.nominal = tuple([0 for _ in self.velocity])
        assert len(self.velocity) == len(self.nominal)

    def __len__(self):
        return len(self.velocity)

    def __str__(self):
        return f"[{self.name}]"

    @property
    def shortest(self):
        return self.nominal

    @property
    def longest(self):
        return self.nominal

    def tinv_transforms(self, pre=0., post=0.):
        # These are, strictly speaking, fully dependent on velocity
        # However, implementing that is beyond our ability/need at the moment
        return pre + min(self.shortest), post + max(self.longest)

    def td_length(self):
        # Hopefully there's only one nominal length ...
        return sum(self.nominal) / len(self.nominal)

    def phase_length(self, target_velocity):
        from numpy import array
        return interpolate(array(self.velocity), array(self.nominal), target_velocity)

    def inv_velocity_limits(self):
        return 1/max(self.velocity), 1/min(self.velocity)

    def tinv_polygon(self, times=None):
        # from numpy import array
        # from nsimplex import Border
        # if times is None:
        #     times = [0, 1e9]
        # iv = self.inv_velocity_limits()
        # points = [[times[0], iv[1]], [times[0], iv[0]], [times[1], iv[0]], [times[1], iv[1]]]
        # return Polygon(Border(array(points)), [])
        return []

    def tinv_overlap(self, other: Polygon, times=None):
        from nsimplex.polygon import intersection
        if times is None:
            times = other.min(), other.max()
        limits = self.tinv_polygon(times=times)
        return intersection(other, limits)


@dataclass
class Guide(FlightPath):
    # The fact that Python lets you replace a base class property with a value amazes me
    shortest: Tuple[float, ...] = (-1., )
    longest: Tuple[float, ...] = (-1., )

    def __str__(self):
        return f"Guide[{self.name}]"

    def __post_init__(self):
        if self.shortest[0] < 0.:
            self.shortest = tuple([0. for _ in self.velocity])
        if self.longest[0] < 0.:
            self.longest = tuple([0. for _ in self.velocity])
        if self.nominal[0] < 0.:
            self.nominal = tuple([(s + l) / 2 for s, l in zip(self.shortest, self.longest)])
        assert len(self) == len(self.nominal) and len(self) == len(self.longest) and len(self) == len(self.shortest)


# neutron_mass = 1.674 927 498 04 x 10-27 kg
# planck constant = 6.626 070 15  x 10-34 J Hz-1
NEUTRON_MASS_OVER_PLANCK_CONSTANT = 1.67492749804e-27 / 6.62607015e-34 / 1e10  # s / m / Å


@dataclass
class AnalyzerArm(FlightPath):
    d_spacing: float = 0.  # ångstrom
    angle: float = 0.  # radian
    mosaic: float = 0.  # radian

    def __str__(self):
        return f"AnalyzerArm[{self.name}]"

    def relative_lambda_uncertainty(self):
        from numpy import cos, sin
        return cos(self.angle) / sin(self.angle) * self.mosaic

    def inv_velocity_limits(self):
        from numpy import sin
        wavelength = 2 * self.d_spacing * sin(self.angle)
        delta_wavelength = wavelength * self.relative_lambda_uncertainty()
        wavelength_range = wavelength - delta_wavelength / 2, wavelength + delta_wavelength /2
        return tuple([x * NEUTRON_MASS_OVER_PLANCK_CONSTANT for x in wavelength_range])
