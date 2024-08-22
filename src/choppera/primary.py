# SPDX-FileCopyrightText: 2024-present Gregory Tucker <gregory.tucker@ess.eu>
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import List, Tuple

from scipp import DataArray, Variable
from polystar import Polygon

from .flightpaths import FlightPath, Guide
from .chopper import DiscChopper
from .phase import Phase


class PulsedSource:
    frequency: Variable
    data: DataArray

    def __eq__(self, other):
        from scipp import allclose
        data_close = all(allclose(v, other.data.coords[k]) for k, v in self.data.coords.items())
        return data_close and allclose(self.frequency, other.frequency)

    def __hash__(self):
        coords = [str(v.values) for v in self.data.coords.values()]
        return hash((str(self.frequency.values), *coords))

    def __init__(self, frequency: Variable,
                 duration: Variable | None = None,
                 delay: Variable | None = None,
                 velocities: Variable | None = None):
        from numpy import argsort
        from scipp import sort
        if velocities is None:
            velocities = Variable(values=[0., 1e12], unit='m/s', dims=['velocity'])
        if delay is None:
            delay = Variable(values=0. * velocities.values, unit='s', dims=velocities.dims)
        elif not delay.ndim:
            delay = Variable(values=delay.value + 0 * velocities.values, unit=delay.unit, dims=velocities.dims)
        if duration is None:
            duration = Variable(values=1. + 0 * velocities.values, unit='s', dims=velocities.dims)
        elif not duration.ndim:
            duration = Variable(values=duration.value + 0 * velocities.values, unit=duration.unit, dims=velocities.dims)

        self.frequency = frequency
        # if the delay, duration and velocities do not have consistent shapes, the following will raise an error
        index = Variable(values=argsort(velocities.values), unit='1', dims=velocities.dims)
        data = DataArray(data=index, coords={
            'delay': delay.to(unit='s'),
            'duration': duration.to(unit='s'),
            'velocities': velocities.to(unit='m/s')
        })
        # sort the data by the velocities
        self.data = sort(data, 'velocities')

    @property
    def delay(self):
        return self.data.coords['delay']

    @property
    def duration(self):
        return self.data.coords['duration']

    @property
    def velocity(self):
        return self.data.coords['velocities']

    @property
    def slowest(self):
        from scipp import min
        return min(self.data.coords['velocities'])

    @property
    def fastest(self):
        from scipp import max
        return max(self.data.coords['velocities'])

    def early_late(self) -> Phase:
        edge = self.data.coords['velocities']
        early = self.data.coords['delay']
        late = early + self.data.coords['duration']
        return Phase(edge, early, late)

    def tinv_polygon(self) -> Polygon:
        from numpy import array
        phase = self.early_late()
        left = [(t, 1 / v) for t, v in zip(phase.left.values, phase.velocity.values)]
        right = [(t, 1 / v) for t, v in zip(phase.right.values, phase.velocity.values)]
        return Polygon(array(list(reversed(left)) + right))

    def arrival_time(self, target: float, centred=False) -> float:
        from numpy import flatnonzero
        de, dr, v = (self.data.coords[x].values for x in ('delay', 'duration', 'velocities'))
        indexes = flatnonzero((target - v) >= 0)
        if len(indexes) < 1:
            raise RuntimeError("The requested velocity is out of range")
        index = indexes[-1]
        diff = (target - v[index]) / (v[index + 1] - v[index])
        delay = (1 - diff) * de[index] + diff * de[index + 1]
        duration = (1 - diff) * dr[index] + diff * dr[index + 1]
        return delay + duration / 2 if centred else delay


@dataclass
class PrimarySpectrometer:
    source: PulsedSource
    pairs: List[Tuple[FlightPath, DiscChopper]]
    sample: FlightPath  # The path to the sample position from the last chopper (allowed to be nothing or guide)

    def __eq__(self, other):
        pairs_match = all([fa == fb and da == db for (fa, da), (fb, db) in zip(self.pairs, other.pairs)])
        return self.source == other.source and pairs_match and self.sample == other.sample

    def __hash__(self):
        return hash((self.source, tuple(self.pairs), self.sample))

    def __init__(self, source: PulsedSource, pairs: List[Tuple[FlightPath, DiscChopper]], sample: FlightPath):
        from scipp import allclose
        # As a limitation to make everything easy/possible, ensure that all guides use the *same* velocity vectors
        v = pairs[0][0].velocity
        for g, _ in pairs:
            assert allclose(g.velocity, v)
        self.pairs = pairs
        self.source = source
        self.sample = sample

    def setup_phases(self, target_velocity, centred=False):
        from scipp import scalar
        cumulative = scalar(0., unit='m')
        for guide, chopper in self.pairs:
            cumulative += guide.phase_length(target_velocity)
            zero = self.source.arrival_time(target_velocity, centred=centred)
            chopper.setup_phase(cumulative, target_velocity, zero_offset=zero, centred=centred)

    def set_phase_angles(self, phase_angles):
        from scipp import to_unit
        for (guide, chopper), phase_angle in zip(self.pairs, phase_angles):
            chopper.phase = to_unit(phase_angle, 'rad').value

    def set_frequencies(self, frequencies):
        from scipp import to_unit
        for (guide, chopper), frequency in zip(self.pairs, frequencies):
            chopper.frequency = to_unit(frequency, 'Hz').value

    def set_delays(self, delays):
        from scipp import to_unit
        for (guide, chopper), delay in zip(self.pairs, delays):
            chopper.set_delay(to_unit(delay, 'sec').value)

    def propagate(self):
        # find the cumulative sum (time, velocity) distributions for each
        vt = [self.source.early_late()]
        for guide, chopper in self.pairs:
            vt = guide.propagate(vt)
            vt = chopper.propagate(vt)
        vt = self.sample.propagate(vt)
        return vt

    def project_all_on_source(self):
        from scipp import scalar, min, max
        from .utils import skew_smear
        regions = [[self.source.tinv_polygon()]]
        slowest, fastest = self.source.slowest, self.source.fastest
        short, long = scalar(0., unit='m'), scalar(0., unit='m')
        for guide, chopper in self.pairs:
            short, long = guide.tinv_transforms(pre=short, post=long)
            delay = min(short / fastest + self.source.delay)
            duration = max(long / slowest + self.source.delay + self.source.duration)
            at_chopper = chopper.tinv_polygons(delay, duration, slowest, fastest)
            at_source = [skew_smear(window, -long, -short) for window in at_chopper]
            regions.append(at_source)
        return regions

    def project_transmitted_on_source(self):
        regions = self.project_all_on_source()
        remaining = regions[0]
        layers = [remaining]
        for idx in range(1, len(regions)):
            remaining = [pg for w in regions[idx] for pgs in [r.intersection(w) for r in remaining] for pg in pgs if pg.area]
            layers.append(remaining)
        return remaining, layers

    def project_transmitted_on_sample(self):
        from scipp import scalar
        from .utils import skew_smear
        at_source, layers = self.project_transmitted_on_source()
        short, long = scalar(0., unit='m'), scalar(0., unit='m')

        def forward_project(shrt, lng, on):
            return [skew_smear(x, shrt, lng) for x in on]

        for guide, chopper in self.pairs:
            short, long = guide.tinv_transforms(pre=short, post=long)
        short, long = self.sample.tinv_transforms(pre=short, post=long)

        at_sample = forward_project(short, long, at_source)
        s_layers = [forward_project(short, long, layer) for layer in layers]
        return at_sample, s_layers

    def project_on_source_alternate(self):
        from scipp import scalar, min, max
        from .utils import skew_smear
        regions = [self.source.tinv_polygon()]
        slowest, fastest = self.source.slowest, self.source.fastest
        short, long = scalar(0., unit='m'), scalar(0., unit='m')
        for guide, chopper in self.pairs:
            short, long = guide.tinv_transforms(pre=short, post=long)
            delay = min(short / fastest + self.source.delay)
            duration = max(long / slowest + self.source.delay + self.source.duration)
            at_chopper = chopper.tinv_polygons(delay, duration, slowest, fastest)
            at_source = [skew_smear(w, -long, -short) for w in at_chopper]
            regions = [z for w in at_source for z in [r.intersection(w) for r in regions] if z.area]
        return regions

    def project_on_sample_alternate(self):
        from scipp import scalar, min, max
        from numpy import min as np_min
        from .utils import skew_smear
        regions = [self.source.tinv_polygon()]
        slowest, fastest = self.source.slowest, self.source.fastest
        tot_short, tot_long = scalar(0., unit='m'), scalar(0., unit='m')
        for guide, chopper in self.pairs:
            # just this guide, velocity independent (for now)
            short, long = guide.tinv_transforms(pre=scalar(0., unit='m'), post=scalar(0., unit='m'))
            tot_short += short
            tot_long += long
            delay = min(tot_short / fastest + self.source.delay)
            duration = max(tot_long / slowest + self.source.delay + self.source.duration)
            moved = [skew_smear(x, short, long) for x in regions]
            regions = chopper.tinv_overlap(moved, delay, duration, slowest, fastest)
        short, long = self.sample.tinv_transforms(pre=scalar(0., unit='m'), post=scalar(0., unit='m'))
        on_sample = [skew_smear(x, short, long) for x in regions]
        return list(sorted(on_sample, key=lambda x: np_min(x.vertices[:, 0])))

    def forward_time_distance_diagram(self):
        from polystar import Polygon
        from scipp import scalar, min, max
        from .utils import skew_smear

        def td_poly(low: Polygon, up: Polygon, a: Variable, b: Variable):
            from numpy import min as n_min, max as n_max
            low_min = n_min(low.vertices[:, 0])
            low_max = n_max(low.vertices[:, 0])
            up_min = n_min(up.vertices[:, 0])
            up_max = n_max(up.vertices[:, 0])
            return Polygon([[low_min, a.value], [low_max, a.value], [up_max, b.value], [up_min, b.value]])

        first = [self.source.tinv_polygon()]
        slowest, fastest = self.source.slowest, self.source.fastest
        tot_short, tot_long = scalar(0., unit='m'), scalar(0., unit='m')
        parts = []
        zero = scalar(0., unit='m')
        for guide, chopper in self.pairs:
            short, long = guide.tinv_transforms(pre=scalar(0., unit='m'), post=scalar(0., unit='m'))
            tot_short += short
            tot_long += long
            delay = min(tot_short / fastest + self.source.delay)
            duration = max(tot_long / slowest + self.source.delay + self.source.duration)
            second = [skew_smear(x, short, long) for x in first]
            d = guide.td_length()
            parts.append([td_poly(low, up, zero, zero+d) for low, up in zip(first, second)])
            zero += d
            first = chopper.tinv_overlap(second, delay, duration, slowest, fastest)
        short, long = self.sample.tinv_transforms(pre=scalar(0., unit='m'), post=scalar(0., unit='m'))
        second = [skew_smear(x, short, long) for x in first]
        d = self.sample.td_length()
        parts.append([td_poly(low, up, zero, zero + d) for low, up in zip(first, second)])
        return parts

    def time_distance_openings(self, minimum_time: Variable | None = None, maximum_time: Variable | None = None):
        from scipp import scalar
        from numpy import nan
        if minimum_time is None:
            minimum_time = scalar(0., unit='s')
        if maximum_time is None:
            maximum_time = 1 / self.source.frequency
        zero = scalar(0., unit='m')
        x, y = [], []
        for guide, chopper in self.pairs:
            zero += guide.td_length()
            windows = chopper.windows_time(earliest=minimum_time, latest=maximum_time, sort=True)
            last = minimum_time
            for edge in range(windows.sizes['edges']):
                times =  windows['edges', edge]
                t_open, t_close = times['window', 0], times['window', 1]
                if last >= t_open:
                    x.extend([last.value, t_open.value, t_open.value, t_close.value])
                    y.extend([zero.value, zero.value, nan, nan])
                last = t_close
            if last < maximum_time:
                x.extend([last.value, maximum_time.value, maximum_time.value])
                y.extend([zero.value, zero.value, nan])
        return x, y

    def time_distance_frames(self,
                             minimum_time: Variable | None = None,
                             maximum_time: Variable | None = None,
                             offset=0.,
                             extra_times=None
                             ):
        from scipp import scalar, min, sum, concat
        from numpy import nan
        if minimum_time is None:
            minimum_time = scalar(0., unit='s')
        if maximum_time is None:
            maximum_time = 1 / self.source.frequency
        length = sum(concat([g.td_length() for g, _ in self.pairs], dim='guides'))
        x, y = [], []
        zero = min(self.source.delay + offset)
        while zero < maximum_time:
            if zero > minimum_time:
                x.extend([zero.value, zero.value, zero.value, zero.value])
                y.extend([nan, 0, length.value, nan])
            zero += 1 / self.source.frequency
        if extra_times:
            for zero in extra_times:
                x.extend([zero.value, zero.value, zero.value, zero.value])
                y.extend([nan, 0, length.value, nan])
        return x, y

