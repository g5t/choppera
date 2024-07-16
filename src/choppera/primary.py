# SPDX-FileCopyrightText: 2024-present Gregory Tucker <gregory.tucker@ess.eu>
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field
from typing import List, Tuple
from numpy import ndarray, array

from .flightpaths import FlightPath, Guide
from .chopper import DiscChopper

@dataclass
class PulsedSource:
    frequency: float
    duration: ndarray = field(default=array((1.,)))
    delay: ndarray = field(default=array((0.,)))
    velocities: ndarray = field(default=array((0., 1e12)))

    def __post_init__(self):
        from numpy import argsort
        if len(self.velocities) < 2:
            raise RuntimeError("At least two velocities required!")
        if len(self.delay) != len(self.velocities) and len(self.delay) > 1:
            raise RuntimeError("Delay times must match velocities!")
        if len(self.duration) != len(self.velocities) and len(self.duration) > 1:
            raise RuntimeError("Duration times must match velocities!")
        if len(self.delay) < 2:
            self.delay = self.delay[0] + 0 * self.velocities
        if len(self.duration) < 2:
            self.duration = self.duration[0] + 0 * self.velocities
        index = argsort(self.velocities)
        self.velocities = self.velocities[index]
        self.delay = self.delay[index]
        self.duration = self.duration[index]

    @property
    def slowest(self):
        from numpy import min
        return min(self.velocities)

    @property
    def fastest(self):
        from numpy import max
        return max(self.velocities)

    def early_late(self) -> Tuple[ndarray, ndarray, ndarray]:
        edge = self.velocities
        early = self.delay
        late = early + self.duration
        return edge, early, late

    def tinv_polygon(self):
       # from nsimplex import Polygon, Border
       # from numpy import array
       # vel, early, late = self.early_late()
       # left = [(t, 1 / v) for t, v in zip(early, vel)]
       # right = [(t, 1 / v) for t, v in zip(late, vel)]
       # return Polygon(Border(array(list(reversed(left)) + right)), [])
       return []

    def arrival_time(self, target: float, centred=False):
        # This was correct when there was only one delay one one duration; now we need to interpolate:
        # zero = self.delay + self.duration / 2 if centred else self.delay
        from numpy import flatnonzero
        indexes = flatnonzero((target - self.velocities) >= 0)
        if len(indexes) < 1:
            raise RuntimeError("The requested velocity is out of range")
        index = indexes[-1]
        diff = (target - self.velocities[index]) / (self.velocities[index+1] - self.velocities[index])
        delay = (1-diff) * self.delay[index] + diff * self.delay[index+1]
        duration = (1-diff) * self.duration[index] + diff * self.duration[index+1]
        return delay + duration / 2 if centred else delay


@dataclass
class PrimarySpectrometer:
    source: PulsedSource
    pairs: List[Tuple[FlightPath, DiscChopper]]
    sample: FlightPath  # The final flight path to the sample position from the last chopper (allowed to be nothing or guide)

    def __init__(self, source: PulsedSource, pairs: List[Tuple[FlightPath, DiscChopper]], sample: FlightPath):
        from numpy import allclose
        # As a limitation to make everything easy/possible, ensure that all guides use the *same* velocity vectors
        v = pairs[0][0].velocity
        for g, _ in pairs:
            assert allclose(g.velocity, v)
        self.pairs = pairs
        self.source = source
        self.sample = sample

    def setup_phases(self, target_velocity, centred=False):
        cumulative = 0.
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
        vt = [self.source.early_late(self.pairs[0][0].velocity)]
        for guide, chopper in self.pairs:
            vt = guide.propagate(vt)
            vt = chopper.propagate(vt)
        vt = self.sample.propagate(vt)
        return vt

    def project_all_on_source(self):
        from numpy import min, max
        regions = [[self.source.tinv_polygon()]]
        slowest, fastest = self.source.slowest, self.source.fastest
        short, long = 0., 0.
        for guide, chopper in self.pairs:
            short, long = guide.tinv_transforms(pre=short, post=long)
            delay = min(short / fastest + self.source.delay)
            duration = max(long / slowest + self.source.delay + self.source.duration)
            at_chopper = chopper.tinv_polygons(delay, duration, slowest, fastest)
            at_source = [window.skew_smear(-long, -short) for window in at_chopper]
            regions.append(at_source)
        return regions

    def project_transmitted_on_source(self):
        # import matplotlib.pyplot as pp
        # from nsimplex.plot import plot_polygons
        from nsimplex.polygon import intersection
        regions = self.project_all_on_source()
        remaining = regions[0]
        layers = [remaining]
        # fig, ax = pp.subplots(1, 1)
        # plot_polygons(ax, remaining, alpha=0.2)
        for idx in range(1, len(regions)):
            # for w in regions[idx]:
            #     intersections = [intersection(r, w) for r in remaining]
            # plot_polygons(ax, regions[idx], alpha=0.2)
            remaining = [z for w in regions[idx] for z in [intersection(r, w) for r in remaining] if not z.isempty]
            # plot_polygons(ax, remaining, color='red', alpha=0.2)
            layers.append(remaining)
        return remaining, layers

    def project_transmitted_on_sample(self):
        at_source, layers = self.project_transmitted_on_source()
        short, long = 0., 0.

        def forward_project(s, l, on):
            return [x.skew_smear(s, l) for x in on]

        for guide, chopper in self.pairs:
            short, long = guide.tinv_transforms(pre=short, post=long)
        short, long = self.sample.tinv_transforms(pre=short, post=long)

        at_sample = forward_project(short, long, at_source)
        s_layers = [forward_project(short, long, l) for l in layers]
        return at_sample, s_layers

    def project_on_source_alternate(self):
        from nsimplex.polygon import intersection
        from numpy import min, max
        regions = [self.source.tinv_polygon()]
        slowest, fastest = self.source.slowest, self.source.fastest
        short, long = 0., 0.
        for guide, chopper in self.pairs:
            short, long = guide.tinv_transforms(pre=short, post=long)
            delay = min(short / fastest + self.source.delay)
            duration = max(long / slowest + self.source.delay + self.source.duration)
            at_chopper = chopper.tinv_polygons(delay, duration, slowest, fastest)
            at_source = [w.skew_smear(-long, -short) for w in at_chopper]
            regions = [z for w in at_source for z in [intersection(r, w) for r in regions] if not z.isempty]
        return regions

    def project_on_sample_alternate(self):
        from numpy import min, max
        regions = [self.source.tinv_polygon()]
        slowest, fastest = self.source.slowest, self.source.fastest
        tot_short, tot_long = 0., 0.
        for guide, chopper in self.pairs:
            short, long = guide.tinv_transforms()  # just this guide, velocity independent (for now)
            tot_short += short
            tot_long += long
            delay = min(tot_short / fastest + self.source.delay)
            duration = max(tot_long / slowest + self.source.delay + self.source.duration)
            moved = [x.skew_smear(short, long) for x in regions]
            regions = chopper.tinv_overlap(moved, delay, duration, slowest, fastest)
        short, long = self.sample.tinv_transforms()
        on_sample = [x.skew_smear(short, long) for x in regions]
        return list(sorted(on_sample, key=lambda x: x.min()))

    def forward_time_distance_diagram(self):
       # from numpy import array, min, max
       # from nsimplex import Border, Polygon
       # def td_poly(low, up, a, b):
       #     verts = array([[low.min(), a], [low.max(), a], [up.max(), b], [up.min(), b]])
       #     border = Border(verts)
       #     return Polygon(border, [])

       # first = [self.source.tinv_polygon()]
       # slowest, fastest = self.source.slowest, self.source.fastest
       # tot_short, tot_long = 0., 0.
       # parts = []
       # zero = 0.
       # for guide, chopper in self.pairs:
       #     short, long = guide.tinv_transforms()
       #     tot_short += short
       #     tot_long += long
       #     delay = min(tot_short / fastest + self.source.delay)
       #     duration = max(tot_long / slowest + self.source.delay + self.source.duration)
       #     second = [x.skew_smear(short, long) for x in first]
       #     d = guide.td_length()
       #     parts.append([td_poly(l, u, zero, zero+d) for l, u in zip(first, second)])
       #     zero += d
       #     first = chopper.tinv_overlap(second, delay, duration, slowest, fastest)
       # short, long = self.sample.tinv_transforms()
       # second = [x.skew_smear(short, long) for x in first]
       # d = self.sample.td_length()
       # parts.append([td_poly(l, u, zero, zero + d) for l, u in zip(first, second)])
       # return parts
       return []

    def time_distance_openings(self, minimum_time=0., maximum_time=None):
        from numpy import nan
        if maximum_time is None:
            maximum_time = 1 / self.source.frequency
        zero = 0.
        x, y = [], []
        for guide, chopper in self.pairs:
            zero += guide.td_length()
            windows = chopper.windows_time(delay=minimum_time, duration=maximum_time-minimum_time, sort=True)
            last = minimum_time
            for t_open, t_close in windows:
                if t_open < last:
                    pass
                else:
                    x.extend([last, t_open, t_open, t_close])
                    y.extend([zero, zero, nan, nan])
                last = t_close
            if last < maximum_time:
                x.extend([last, maximum_time, maximum_time])
                y.extend([zero, zero, nan])
        return x, y

    def time_distance_frames(self, minimum_time=0., maximum_time=None, offset=0., extra_times=None):
        from numpy import nan, min
        if maximum_time is None:
            maximum_time = 1 / self.source.frequency
        length = sum([g.td_length() for g, _ in self.pairs])
        x, y = [], []
        zero = min(self.source.delay + offset)
        while zero < maximum_time:
            if zero > minimum_time:
                x.extend([zero, zero, zero, zero])
                y.extend([nan, 0, length, nan])
            zero += 1 / self.source.frequency
        if extra_times:
            for zero in extra_times:
                x.extend([zero, zero, zero, zero])
                y.extend([nan, 0, length, nan])
        return x, y
