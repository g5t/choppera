# SPDX-FileCopyrightText: 2024-present Gregory Tucker <gregory.tucker@ess.eu>
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import Tuple, List
from numpy import ndarray, pi as npi
from scipp import Variable, DataArray, scalar
from polystar import Polygon

from .phase import Phase

# Since scipp.constants.pi dimensionless and not radians, we might as well define tau:
tau = scalar(2 * npi, unit='radian')


class Aperture:
    _half: Variable
    height: Variable
    offset: Variable
    first: Variable
    second: Variable

    def __eq__(self, other):
        from scipp import allclose
        mine = self.width, self.height, self.offset, self.first, self.second
        theirs = other.width, other.height, other.offset, other.first, other.second
        return all(allclose(m, t) for m, t in zip(mine, theirs))

    def __hash__(self):
        arrays = [str(x.values) for x in (self.width, self.height, self.offset, self.first, self.second)]
        return hash(tuple(arrays))

    def __init__(self, width: Variable | None = None, height: Variable | None = None, offset: Variable | None = None):
        from scipp import atan2, scalar
        if width is None:
            width = scalar(0., unit='m')
        if height is None:
            height = scalar(0., unit='m')
        if offset is None:
            offset = scalar(0., unit='m')
        self._half = 0 * width
        self.width = width
        self.height = height
        self.offset = offset
        self.first = atan2(y=self.width / 2, x=self.offset)
        self.second = atan2(y=self.width / 2, x=self.offset + self.height)

    @property
    def width(self):
        return 2 * self.half

    @width.setter
    def width(self, w):
        self.half = w / 2

    @property
    def half(self):
        return self._half

    @half.setter
    def half(self, h):
        assert h >= 0 * h
        self._half = h

    def half_step(self, density=10):
        from numpy import linspace, tan, hstack
        from scipp import array
        # step from (0, 0) to (first, half * height)
        # through (second, 1/2 {height**2 + 2*height*offset} * tan(second))
        # triangle intersects top of aperture:
        f = self.first.to(unit='radian').value
        s = self.second.to(unit='radian').value
        h = self.height.to(unit='m').value
        o = self.offset.to(unit='m').value
        z = self.half.to(unit='m').value
        x1 = linspace(0, s, num=density)
        p1 = 0.5 * (h ** 2 + 2 * h * o) * tan(x1)
        x2 = linspace(s + (f - s) / (density - 1), f, num=density - 1)
        tx2 = tan(x2) / 2
        p2 = z * (h + o) - (o ** 2) * tx2 - (z ** 2) / tx2
        angles = array(values=hstack((x1, x2)), unit='radian', dims=['angle'])
        opening = array(values=hstack((p1, p2)), unit='m^2', dims=['angle'])
        return angles, opening

    def open(self, density=10):
        from numpy import hstack
        # step from (0, 0) to (first-second, (height**2 + 2*height*offset)*tan(second)) to (first, (half * height))
        # to (first+second, half*height + (height**2 + 2*height*offset)*tan(second))
        # to (2*first, width*height)
        h_xi, h_step = self.half_step(density)
        f = self.first.to(unit='radian').value
        x = hstack((list(reversed(f - h_xi.values[:-1])), h_xi.values + f))
        h = (self.half * self.height).to(unit='m^2').value
        a = (self.height * self.width).to(unit='m^2').value
        y = hstack((list(reversed(h - h_step.values[:-1])), h_step.values + h)) / a
        return x, y

    def close(self, density=10):
        x, y = self.open(density)
        return x, 1 - y


def list_append_at(points: List[Tuple[float, List[float]]], at: float) -> List[Tuple[float, List[float]]]:
    from numpy import max, argwhere
    less = [p[0] < at for p in points]
    if any(less) and any([p[0] > at for p in points]):
        idx = max(argwhere(less))
        x = (at - points[idx][0]) / (points[idx + 1][0] - points[idx][0])
        y = points[idx][1][-1] + x * (points[idx + 1][1][0] - points[idx][1][-1])
        point = at, [y, ]
        points.append(point)
    return points


def vector_append_at(xy: ndarray, ats: List[float]) -> ndarray:
    from numpy import max, argwhere, any, hstack
    for at in ats:
        if any(xy[0] < at) and any(xy[0] > at):
            idx = max(argwhere(xy[0] < at))
            ratio = (at - xy[0, idx]) / (xy[0, idx + 1] - xy[0, idx])
            value = ratio * xy[1, idx + 1] + (1 - ratio) * xy[1, idx]
            xy = hstack((xy[:, :idx], [[at], [value]], xy[:, idx:]))
    return xy


class Chopper:
    """A chopper is a device which periodically blocks or opens a pathway in a neutron instrument

    Parameters
    ----------
    name : str
        The name of the chopper
    frequency : Variable, optional
        The frequency of the chopper in Hz
    phase_to : Tuple[str, int], optional
        The name of the frequency to which the chopper is synchronized, and the harmonic overtone
    phase : Variable, optional
        The phase of the chopper in radians, zero corresponds to the zero-angle of the chopper passing a reference
        point (here forced to be the center of the beam aperture) when the harmonic overtone of the reference frequency
        completes a cycle.
        This could alternatively be the angle relative to the arrival time of a specific wavelength of neutrons
        at the chopper.
    aperture : Aperture, optional
        The physical aperture of the chopper, limiting the profile of the beam at the chopper position
    """
    name: str
    _frequency: Variable
    _to: Tuple[str, int]
    _phase: Variable
    _aperture: Aperture

    def __eq__(self, other):
        from scipp import allclose
        mine = self.frequency, self.phase
        theirs = other.frequency, other.phase
        sc = all(allclose(m, t) for m, t in zip(mine, theirs))
        return self.name == other.name and self.aperture == other.aperture and sc

    def __hash__(self):
        arrays = self._frequency.values, self._to, self._phase.values
        return hash((self.name, *[str(x) for x in arrays], self._aperture))

    def __init__(self,
                 name: str,
                 frequency: Variable | None = None,
                 phase_to: tuple[str, int] | None = None,
                 phase: Variable | None = None,
                 aperture: Aperture | None = None):
        from scipp import scalar
        if frequency is None:
            frequency = scalar(0., unit='Hz')
        if phase_to is None:
            phase_to = ('', 0)
        if phase is None:
            phase = scalar(0., unit='radian')
        if aperture is None:
            aperture = Aperture()
        self.name = name
        self._frequency = frequency.to(unit='Hz')
        self._phase = phase.to(unit='radian')
        self._aperture = Aperture()
        self.frequency = frequency
        self._to = phase_to
        self.phase = phase.to(unit='radian')
        self.aperture = aperture

    def __str__(self):
        return f"Chopper[{self.name}]"

    @property
    def aperture(self):
        return self._aperture

    @property
    def frequency(self):
        return self._frequency

    @property
    def period(self):
        from scipp import abs
        return abs(1 / self.frequency)

    @property
    def phase(self):
        return self._phase

    @aperture.setter
    def aperture(self, a: Aperture):
        self._aperture = a

    @frequency.setter
    def frequency(self, f: Variable):
        assert f != 0. * f, "Chopper frequency must be non-zero"
        self._frequency = f.to(unit='Hz', dtype='float64')

    @phase.setter
    def phase(self, p: Variable):
        self._phase = p.to(unit='radian', dtype='float64') % tau

    def fully_open_delay(self, window=None):
        from scipp import scalar
        return scalar(0., unit='radian')

    def centered_delay(self, window=None):
        from scipp import scalar
        return scalar(0., unit='radian')

    def partly_open_delay(self, window=None):
        from scipp import scalar
        return scalar(0., unit='radian')

    def setup_phase(self,
                    flight_length: Variable,
                    target_velocity: Variable,
                    target_window: int | None = None,
                    zero_offset: Variable | None = None,
                    centred: bool = False):
        """Set the phase of the chopper, given the flight length and target velocity

        Parameters
        ----------
        flight_length : Variable
            The distance the neutrons travel from the source to the chopper
        target_velocity : Variable
            The velocity of the neutrons at the chopper
        target_window : int, optional
            The window number to target
        zero_offset : Variable, optional
            The offset in time of the zero-angle of the chopper
        centred : bool, optional
            Whether to center the phase on the window or the window leading edge
        """
        if zero_offset is None:
            zero_offset = scalar(0., unit='s')

        arrival = flight_length / target_velocity + zero_offset
        delay = self.centered_delay(window=target_window) if centred else self.fully_open_delay(window=target_window)
        # TODO check that this is correct for negative frequency
        self.phase = (arrival % self.period) * tau * self.frequency - delay

    def set_delay(self, to: Variable, target_window: int | None = None, centred: bool = False):
        delay = self.centered_delay(window=target_window) if centred else self.partly_open_delay(window=target_window)
        # TODO check that this is correct for negative frequency
        self.phase = (to.to(unit='s', dtype='float64') % self.period) * tau * self.frequency - delay


class DiscChopper(Chopper):
    radius: Variable
    _windows: Variable
    _discs: int = 1

    def __eq__(self, other):
        from scipp import allclose
        mine = self.radius, self.windows
        theirs = other.radius, other.windows
        return super().__eq__(other) and all(allclose(m, t) for m, t in zip(mine, theirs)) and self.discs == other.discs

    def __hash__(self):
        arrays = self.radius.values, self._windows.values
        return hash((super().__hash__(), *[str(x) for x in arrays], self._discs))

    def __str__(self):
        return f"DiscChopper[{self.name}]"

    def __init__(self,
                 name: str,
                 radius: Variable,
                 frequency: Variable | None = None,
                 phase_to: Tuple[str, int] | None = None,
                 phase: Variable | None = None,
                 aperture: Aperture | None = None,
                 windows: Variable | None = None,
                 discs=1
                 ):
        super().__init__(name, frequency, phase_to, phase, aperture)
        assert radius > 0 * radius
        self.radius = radius.to(unit='m', dtype='float64')
        self._windows = Variable(values=[[0, 0]], dims=['slot', 'window'], unit='radian', dtype='float64')
        if windows is not None:
            self.windows = windows
        self.discs = discs

    @property
    def discs(self):
        return self._discs

    @discs.setter
    def discs(self, value: int):
        assert 0 < value < 3
        self._discs = value

    @property
    def angular_velocity(self):
        #  a double disc chopper effectively has a twice-faster edge velocity
        return tau * self.frequency * self.discs

    @property
    def windows(self):
        return self._windows

    @windows.setter
    def windows(self, wins: Variable):
        from scipp import any, abs, array, DataArray, sort
        assert wins.ndim == 2, "Windows must be a 2D array"
        assert 'slot' in wins.sizes and wins.sizes['slot'] > 0, "Windows must have a slot dimension"
        assert 'window' in wins.sizes and wins.sizes['window'] == 2, "Windows must have two columns for window edges"
        assert wins.unit == 'radian', "Windows must be in radians"

        # minimum_angle_half_width = self.aperture.second if self.discs == 2 else self.aperture.first
        minimum_angle_half_width = self.aperture.first
        half_widths = abs(wins['window', 1] - wins['window', 0]) / 2.0
        if any(half_widths < minimum_angle_half_width):
            raise RuntimeWarning(f"Window narrower than aperture. Case is not handled correctly.")

        centers = ((wins['window', 0] + wins['window', 1]) / 2.0) % tau
        # centers and halves sorted by centers
        windows = DataArray(array(values=[-1, 1], dims=['window']) * half_widths + centers, coords={'center': centers})
        windows = sort(windows, 'center').data
        self._windows = windows

    def nonzero(self):
        """Return the probability of the chopper being open at critical angles

        The values are the path defining the edge of the probability distribution for the chopper being open.
        At each critical angle the probability switches from zero to finite (or the reverse) so every critical
        angle is represented twice, with opening edges first zero then one and closing edges first one then zero.

        Returns
        -------
        DataArray
            data: the probability at each critical angle (0 or 1)
            coords: {'psi': the critical angles}
        """
        from scipp import concat, DataArray, sort
        #  The starting/ending opening angles of a single disc are defined by the aperture width
        #  For a double disc it is the angle where the two opening edges meet, which *should* be zero
        zero = scalar(0., unit='radian', dtype='float64')
        critical_angle = zero if self.discs == 2 else self.aperture.first

        d0, d1 = 'switch', 'psi'
        psi = Variable(values=[-1, 1], dtype='float64', dims=[d1]) * critical_angle
        prob = Variable(values=[[0, 1], [1, 0]], dtype='float64', dims=[d0, d1])

        def open_close(window):
            # Reverse the window edges before adding psi and the phase
            x = Variable(values=-(window.values[::-1]), dims=[d1], unit=window.unit) + psi + self.phase
            y = prob
            return DataArray(data=y, coords={'psi': x})

        # find the critical angles for each window, always ordered [(open, close), ...]
        oc = [open_close(self.windows['slot', x]) for x in range(self.windows.sizes['slot'])]
        # combine them all and ensure they're sorted by angle (if there is overlap we have bigger problems)
        xy = sort(concat(oc, d1) if len(oc) > 1 else oc[0], 'psi')
        # transpose to have the switch dimension first, then flatten such that 'psi' gets duplicated
        return xy.transpose([d1, d0]).flatten(to='psi')

    def nonzero_time(self):
        psi_prob = self.nonzero()
        # convert from angle [radian (0, 2pi)] to time and add it as a new coordinate:
        psi_prob.coords['time'] = psi_prob.coords['psi'] * self.period / tau
        return psi_prob

    def windows_time(self, earliest=None, latest=None, sort=False):
        from numpy import roll, abs as np_abs, any as np_any, array as np_array
        from scipp import arange, concat, min, max
        t = self.period
        if earliest is None:
            earliest = scalar(0., unit=t.unit)
        if latest is None or latest.to(unit=t.unit) < t:
            latest = t
        time_prob = self.nonzero_time()
        # select the windows where the probability is non-zero:
        windows = time_prob[time_prob.data > scalar(0., unit=time_prob.unit)]
        # since we selected the 'nonzero' times, these are all probability=1 and only the coordinates are needed
        # reshape them first to (N, 2) windows and limits, keep only the _times_ coordinate
        windows = windows.fold('psi', sizes={'window': -1, 'edges': 2}).coords['time']

        if latest < earliest:
            earliest, latest = latest, earliest

        n_before, n_after = 0, 0
        min_win, max_win = [min(windows), max(windows)]
        # Extend lower time as long as doing so will not add only windows that ends before the earliest time
        while min_win - n_before * t > earliest < max_win - (n_before + 1) * t:
            n_before += 1
        # Extend higher time as long as doing so will not add only windows that starts after the latest time
        while max_win + n_after * t < latest > min_win + (n_after + 1) * t:
            n_after += 1
        if n_before or n_after:
            offsets = arange(start=-n_before, stop=n_after + 1, dim='offset') * t
            windows = (offsets + windows).flatten(to='x').fold('x', sizes={'window': -1, 'edges': 2})


        # If the phase/windows are such that 2 pi is *in* a window, there are likely successive windows which
        # end and start at the same time. Combine them at this point:
        # over_2pi = roll(windows[:, 1], 1) == windows[:, 0]
        over_2pi = np_abs(roll(windows['edges', 1].values, 1) - windows['edges', 0].values) < 1e-15
        if np_any(over_2pi):
            before_2pi = Variable(values=roll(over_2pi, -1), dims=['window'])
            over_2pi = Variable(values=over_2pi, dims=['window'])
            over = concat((windows[before_2pi]['edges', 0], windows[over_2pi]['edges', 1]), dim='edges')
            not_over = windows[~(over_2pi | before_2pi)]
            windows = concat((not_over, over), dim='window')
            if sort:
                windows.values = np_array(sorted(windows.values, key=lambda x: x[0]))
        return windows

    def tinv_polygons(self, delay=None, duration=None, minimum_velocity=1e-9, maximum_velocity=1e9):
        from numpy import array, vstack, repeat
        from scipp import Variable
        if isinstance(minimum_velocity, Variable):
            minimum_velocity = minimum_velocity.to(unit='m/s').value
        if isinstance(maximum_velocity, Variable):
            maximum_velocity = maximum_velocity.to(unit='m/s').value
        v = 1 / array([minimum_velocity, maximum_velocity, maximum_velocity, minimum_velocity])
        t = self.windows_time(delay, duration).to(unit='s').transpose(dims=['window', 'edges']).values
        return [Polygon(vstack((repeat(w, 2), v)).T) for w in t]

    def propagate(self, phases: list[Phase]) -> list[Phase]:
        """Use the chopper openings to divide the provided phase space into new phases"""
        out = []
        for phase in phases:
            t_min, t_max = phase.min(), phase.max()
            for window in self.windows_time(t_min, t_max - t_min):
                open_at, close_at = window['edges', 0], window['edges', 1]
                sub = phase.copy()
                sub.mask(sub.left < open_at)
                sub.mask(sub.right > close_at)
                sub.shift('left', (sub.left < open_at) & (open_at < sub.right), open_at)
                sub.shift('right', (sub.left < close_at) & (close_at < sub.right), close_at)
                if sub:
                    out.append(sub.copy())  # remove any masked edges at this point
        return out

    def fully_open_delay(self, slot=None):
        if slot is None:
            slot = 0
        return self.aperture.first - self.windows['slot': slot]['window': 1]

    def centered_delay(self, slot=None):
        if slot is None:
            slot = 0
        # The delay is the average window edge angle
        return -(self.windows['slot': slot]['window': 1] + self.windows['slot': slot]['window': 0]) / 2

    def partly_open_delay(self, slot=None):
        if slot is None:
            slot = 0
        return -self.aperture.first - self.windows['slot': slot]['window': 1]

    def tinv_overlap(self,
                     bases: List[Polygon],
                     delay: Variable,
                     duration: Variable,
                     minimum_velocity: Variable | None = None,
                     maximum_velocity: Variable | None = None):
        from numpy import array, vstack, repeat
        if minimum_velocity is None:
            minimum_velocity = scalar(1e-9, unit='m/s')
        if maximum_velocity is None:
            maximum_velocity = scalar(1e9, unit='m/s')
        if isinstance(minimum_velocity, Variable):
            minimum_velocity = minimum_velocity.to(unit='m/s').value
        if isinstance(maximum_velocity, Variable):
            maximum_velocity = maximum_velocity.to(unit='m/s').value
        v = 1 / array([minimum_velocity, maximum_velocity, maximum_velocity, minimum_velocity])
        times = self.windows_time(delay, duration).transpose(['window', 'edges'])
        print(times)
        print(times.values)
        windows = [Polygon(vstack((repeat(w, 2), v)).T) for w in times.values]
        regions = []
        for base in bases:
            regions.extend([p for z in [base.intersection(w) for w in windows] for p in z if p.area])
        return regions
