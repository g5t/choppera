# SPDX-FileCopyrightText: 2024-present Gregory Tucker <gregory.tucker@ess.eu>
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
from scipp import DataArray, Variable
from polystar import Polygon

from .phase import Phase
from .utils import interpolate


class BasePath:
    name: str
    data: DataArray
    dim: str

    def __eq__(self, other):
        from scipp import allclose
        mine = self.data.coords
        theirs = other.data.coords
        for key in mine:
            if key not in theirs or not allclose(mine[key], theirs[key]):
                return False
        return self.name == other.name

    def __hash__(self):
        coords = [str(v.values) for v in self.data.coords.values()]
        return hash((self.name, *coords, self.dim))

    def __init__(self, name: str, velocity: Variable, **kwarg_variables):
        from scipp import sort, Variable
        from numpy import argsort, std
        self.name = name
        assert isinstance(velocity, Variable), "Velocity must be a Variable"
        assert len(velocity) >= 2 and std(velocity.values) > 0, "Multiple unique velocities required"
        assert velocity.ndim == 1, "Velocity must be 1-dimensional"

        index = Variable(values=argsort(velocity.values), dims=velocity.dims)
        coords = {'velocity': velocity}
        coords.update(kwarg_variables)
        self.dim = velocity.dims[0]
        self.data = sort(DataArray(data=index, coords=coords), 'velocity')

    def __str__(self):
        return f"[{self.name}]"

    def __len__(self):
        return self.data.sizes[self.dim]

    @property
    def velocity(self):
        return self.data.coords['velocity'].to(unit='m/s')

    @property
    def nominal(self):
        if 'nominal' in self.data.coords:
            return self.data.coords['nominal'].to(unit='m')
        raise NotImplementedError

    @property
    def shortest(self):
        if 'shortest' in self.data.coords:
            return self.data.coords['shortest'].to(unit='m')
        return self.nominal

    @property
    def longest(self):
        if 'longest' in self.data.coords:
            return self.data.coords['longest'].to(unit='m')
        return self.nominal

    def tinv_transforms(self, pre: Variable | None, post: Variable | None) -> Tuple[Variable, Variable]:
        from scipp import min, max, scalar
        if pre is None:
            pre = scalar(0., unit='m')
        if post is None:
            post = scalar(0., unit='m')
        # These are, strictly speaking, fully dependent on velocity
        # However, implementing that is beyond our ability/need at the moment
        return pre + min(self.shortest), post + max(self.longest)

    def td_length(self):
        from scipp import sum
        return sum(self.nominal) / len(self.nominal)

    def phase_length(self, target_velocity) -> Variable:
        from scipp import scalar
        return scalar(interpolate(self.velocity.values, self.nominal.values, target_velocity), unit=self.velocity.unit)

    def inv_velocity_limits(self):
        from scipp import min, max
        return 1/max(self.velocity), 1/min(self.velocity)

    def tinv_polygon(self, times: list[Variable] | None = None) -> Polygon:
        from scipp import min, max
        if times is None:
            times = Variable(values=[0, 1e9], unit='s', dims=['time'])
        t_min, t_max = [t.to(unit='s').value for t in (min(times), max(times))]
        iv_min, iv_max = [iv.to(unit='s/m').value for iv in self.inv_velocity_limits()]
        points = [[t_min, iv_max], [t_min, iv_min], [t_max, iv_min], [t_max, iv_max]]
        return Polygon(points)

    def tinv_overlap(self, other: Polygon, times: list[Variable] | None = None) -> Polygon:
        if times is None:
            times = other.min(), other.max()
        limits = self.tinv_polygon(times=times)
        return other.intersection(limits)

    def propagate(self, phases: list[Phase]) -> list[Phase]:
        from scipp import concat, min, max
        from uuid import uuid4
        out = []
        v = self.velocity
        short, long = [length/v for length in (self.shortest, self.longest)]
        dim = str(uuid4())
        for phase in phases:
            assert phase.velocity == v
            times = concat((phase.left + short, phase.left + long, phase.right + short, phase.right + long), dim=dim)
            out.append(Phase(v, min(times, dim=dim), max(times, dim=dim)))
        return out


class FlightPath(BasePath):

    def __init__(self, name: str, velocity: Variable, nominal: Variable):
        from scipp import ones
        if nominal.ndim != velocity.ndim:
            if nominal.ndim == 0:
                nominal = nominal * ones(dims=velocity.dims, shape=velocity.shape, dtype='float64')
            else:
                raise RuntimeError("Non-scalar nominal distances must match dimensionality (and shape) of velocities")
        super().__init__(name, velocity=velocity, nominal=nominal)


class Guide(BasePath):
    def __init__(self, name: str, velocity: Variable, nominal: Variable, shortest: Variable, longest: Variable):
        super().__init__(name, velocity=velocity, nominal=nominal, shortest=shortest, longest=longest)

    def __str__(self):
        return f"Guide[{self.name}]"


#
# # neutron_mass = 1.674 927 498 04 x 10-27 kg
# # planck constant = 6.626 070 15  x 10-34 J Hz-1
# NEUTRON_MASS_OVER_PLANCK_CONSTANT = 1.67492749804e-27 / 6.62607015e-34 / 1e10  # s / m / Å


class AnalyzerArm(FlightPath):
    d_spacing: Variable
    angle: Variable
    mosaic: Variable

    def __eq__(self, other):
        from scipp import allclose
        d_spacing = allclose(self.d_spacing, other.d_spacing)
        angle = allclose(self.angle, other.angle)
        mosaic = allclose(self.mosaic, other.mosaic)
        return super().__eq__(other) and d_spacing and angle and mosaic

    def __hash__(self):
        arrays = [str(v.values) for v in (self.d_spacing, self.angle, self.mosaic)]
        return hash((super().__hash__(), *arrays))

    def __init__(self, name: str, velocity: Variable, d_spacing: Variable, angle: Variable, mosaic: Variable):
        super().__init__(name, velocity=velocity, nominal=d_spacing)
        self.d_spacing = d_spacing.to(unit='angstrom')
        self.angle = angle.to(unit='radian')
        self.mosaic = mosaic.to(unit='radian')

    def __str__(self):
        return f"AnalyzerArm[{self.name}]"

    def inv_velocity_limits(self):
        # TODO update this to account for higher-order harmonics -> return a list of ranges
        from scipp import cos, sin
        from scipp.constants import Planck, neutron_mass
        wavelength = 2 * self.d_spacing * sin(self.angle)
        delta_wavelength = wavelength * cos(self.angle) / sin(self.angle) * self.mosaic.value
        wavelength_range = wavelength - delta_wavelength / 2, wavelength + delta_wavelength / 2
        return tuple((x * neutron_mass / Planck).to(unit='s/m') for x in wavelength_range)
