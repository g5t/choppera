# SPDX-FileCopyrightText: 2024-present Gregory Tucker <gregory.tucker@ess.eu>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to read and understand YAML configuration files"""
from __future__ import annotations

from strictyaml import ScalarValidator, Map, Seq, Str, Optional, Int, Float
from strictyaml.validators import Validator, SeqValidator
from dataclasses import dataclass


# TODO more-robust unit handling via scipp


def parse_value(x: str, expected_unit: str = None, expected_length: int = -1):
    power = 0
    if expected_unit is None:
        value = x
    else:
        x_split = x.split()
        if len(x_split) > 2:
            # multiple comma-separated values? (hopefully as a literal tuple)
            x_split = ' '.join(x_split[:-1]), x_split[-1]
        if len(x_split) != 2:
            raise RuntimeError(f"{x} does not match '[value] [unit]' expected format")
        value, unit = x_split[0], x_split[1]
        if expected_unit is None:
            expected_unit = unit
        if unit != expected_unit:
            # attempt to support *very limited* power conversions
            pack = {'m': {'cm': 2, 'mm': 3, 'um': 6, 'µm': 6, 'nm': 9, 'å': 10, 'angstrom': 10},
                    'mm': {'m': -3, 'cm': -1, 'um': 3, 'µm': 3, 'nm': 6, 'å': 7, 'angstrom': 7},
                    's': {'ms': 3, 'µs': 6, 'us': 6},
                    'ms': {'s': -3, 'µs': 3, 'us': 3},
                    'µs': {'s': -6, 'ms': -3}
                    }
            if expected_unit not in pack:
                raise RuntimeError("No conversion for {unit} to {expected_unit")
            known = pack[expected_unit]
            if not unit.lower() in known:
                raise RuntimeError("Unknown {unit} for conversion to {expected_unit}")
            power = known[unit.lower()]
    # This is a not-safe thing to do if this code should ever run as part of a service, e.g. on a server
    # A malicious user could create an entry like '__import__("pathlib").Path().absolute()' to do nefarious things
    #
    # Running on your own machine as your own user, you can't do anything special; so just ensure
    # the input you provide doesn't attempt anything bad
    value = eval(value, {})
    if expected_length > -1:
        if isinstance(value, tuple) and len(value) != expected_length:
            raise RuntimeError(f"Expected length {expected_length} item but got {len(value)=}")
        elif not isinstance(value, tuple) and expected_length > 0:
            raise RuntimeError(f"Expected length{expected_length} item but got a scalar")
    power = 10 ** power
    if isinstance(value, tuple):
        value = tuple([v / power for v in value])
    else:
        value /= power
    return value


class List(ScalarValidator):
    def validate_scalar(self, chunk):
        return parse_value(chunk.contents)

    def to_yaml(self, data):
        return f"{data}"


class ScippVariable(ScalarValidator):
    def __init__(self, unit: str, dims: list[str] | None = None, elements: int | None = None):
        self.unit = unit
        if dims is None:
            from uuid import uuid4
            dims = [str(uuid4())]
        self.dims = dims
        self.elements = elements

    def validate_scalar(self, chunk):
        from scipp import scalar, Variable
        value = parse_value(chunk.contents, self.unit)
        if isinstance(value, tuple):
            if self.elements is not None:
                assert len(value) == self.elements, f"Expected {self.elements} elements but got {len(value)}"
            return Variable(values=list(value), unit=self.unit, dims=self.dims)
        if self.elements is not None:
            assert self.elements == 1, f"Expected {self.elements} elements but got a scalar"
        return scalar(value, unit=self.unit)

    def to_yaml(self, data):
        from scipp import Variable
        if isinstance(data, Variable):
            data = data.to(unit=self.unit)
            if data.ndim == 0:
                data = data.value
            elif len(data) == 1:
                data = data.values[0]
            elif data.ndim == 1:
                data = tuple(data.values)
            else:
                raise RuntimeError(f"Cannot convert {data} to a scalar")
        return f"{data} {self.unit}"


class PairStrInt(ScalarValidator):
    def __repr__(self):
        return f"Pair(Str, Int)"

    def validate_scalar(self, chunk):
        pair = chunk.contents.split(',')  # do something fancy with parenthetical groups?
        assert len(pair) == 2, "the Pair must be provided two comma separated values"
        return pair[0], int(pair[1])

    def to_yaml(self, data):
        return f"{data[0]}, {data[1]}"


SOURCE_SCHEMA = Map({
    'name': Str(),
    'frequency': ScippVariable('Hz'),
    'duration': ScippVariable('s', ['wavelength']),
    'velocities': ScippVariable('m/s', ['wavelength']),
    'emission_delay': ScippVariable('s', ['wavelength']),
})

FREQUENCY_SCHEMA = Map({
    'name': Str(),
    Optional('harmonics'): List(),
    Optional('highest_harmonic'): Int(),
    Optional('ratio'): PairStrInt(),
    Optional('value'): Int(),
})

SEGMENT_SCHEMA = Map({
    'name': Str(),
    'length': ScippVariable('m'),
    Optional('guide'): Map({
        'velocities': ScippVariable('m/s', ['wavelength_limit'], 2),
        Optional('short'): ScippVariable('m', ['wavelength_limit'], 2),
        Optional('long'): ScippVariable('m', ['wavelength_limit'], 2),
    }),
})

CHOPPER_SCHEMA = Map({
    'name': Str(),
    'position': ScippVariable('m'),
    'opening': ScippVariable('degrees'),
    'radius': ScippVariable('mm'),
    Optional('discs'): Int(),
    Optional('slots'): Int(),
    Optional('frequency'): Map({'name': Str(), Optional('multiplier'): Int()}),
    'aperture': Map({
        'width': ScippVariable('mm'),
        'height': ScippVariable('mm'),
        Optional('offset'): ScippVariable('mm')
    })
})

SAMPLE_SCHEMA = Map({'position': ScippVariable('m')})

PRIMARY_SCHEMA = Map({
    'frequencies': Seq(FREQUENCY_SCHEMA),
    'path_segments': Seq(SEGMENT_SCHEMA),
    'choppers': Seq(CHOPPER_SCHEMA),
    'sample': SAMPLE_SCHEMA,
})

SCHEMA = Map({'name': Str(), 'source': SOURCE_SCHEMA, 'primary_spectrometer': PRIMARY_SCHEMA})


def parse(contents: str):
    from strictyaml import load
    return load(contents, SCHEMA).data


def load(filename):
    from pathlib import Path
    text = Path(filename).read_text()
    return parse(text)


def load_flight_path(path, velocities):
    from scipp import Variable
    from .flightpaths import FlightPath, Guide
    length = path['length']
    nominal = length * Variable(values=[1, 1], dims=['wavelength_limit'])
    if 'guide' in path:
        g = path['guide']
        shortest = g.get('short', nominal)
        longest = g.get('long', nominal)
        return Guide(name=path['name'], velocity=g['velocities'], shortest=shortest, longest=longest, nominal=nominal)
    elif 'bragg' in path:
        raise NotImplementedError("Not implemented yet ...")
    else:
        return FlightPath(name=path['name'], velocity=velocities, nominal=nominal)


def load_chopper(vals, harmonics):
    from scipp import arange as s_range, array, scalar
    from numpy import pi, arange
    from .chopper import Aperture, DiscChopper
    phase = scalar(0., unit='radian')
    h = vals['aperture']['height']
    offset = vals['aperture'].get('offset', vals['radius'] - h)
    aperture = Aperture(vals['aperture']['width'], h, offset)
    theta = vals['opening'].to(unit='radian')
    slots = vals.get('slots', 1)
    slot_at = s_range(dim='slot', start=0, stop=slots, unit='radian') * 2 * pi / slots
    window_width = array(values=[-0.5, 0.5], dims=['window']) * theta
    windows = window_width + slot_at

    freq_dict = vals.get('frequency', {})
    phase_to = freq_dict.get('name', 'Source'), freq_dict.get('multiplier', 1)
    frequency = harmonics[phase_to[0]] * phase_to[1]  # don't worry too much about this value yet

    return DiscChopper(name=vals['name'], radius=vals['radius'], frequency=frequency, phase_to=phase_to,
                       phase=phase, aperture=aperture, windows=windows, discs=vals.get('discs', 1))


def prime_factors(n):
    """Returns all the prime factors of a positive integer"""
    factors = []
    d = 2
    while n > 1:
        while n % d == 0:
            factors.append(d)
            n /= d
        d = d + 1
        if d * d > n:
            if n > 1: factors.append(n)
            break
    return factors


def divisors(n):
    return [d for d in range(n + 1) if n % d == 0]


def load_frequencies(vals, base_frequency):
    from numpy import arange, ndarray
    from .frequencies import IndependentHarmonics, DependentHarmonics, Frequencies
    # build up the *Harmonics objects before creating the composite Frequency object

    values = {}
    objects = {}
    names = []
    for val in vals:
        name = val.get('name', 'UNDEFINED')
        names.append(name)
        value = val.get('value', 1)
        if 'harmonics' in val:
            harmonics = val['harmonics']
        else:
            highest = val.get('highest_harmonic', 1)
            harmonics = list(arange(highest) + 1)
        if value not in harmonics:
            value = min(harmonics)

        if 'ratio' in val:
            to, ratio = val['ratio']
            assert to in objects, f"the frequency {to} must be defined before {name}"
            allowed = {m: [d for d in divisors(m) if d in harmonics] for m in objects[to].allowed * ratio}
            obj = DependentHarmonics(name, objects[to], allowed)
        else:
            obj = IndependentHarmonics(name, harmonics)

        values[name] = value
        objects[name] = obj

    if 'Source' not in names:
        values['Source'] = 1
        objects['Source'] = IndependentHarmonics('Source', [1])
        names.append('Source')

    harmonics = [values[x] for x in names]
    composite = Frequencies(base_frequency, [objects[x] for x in names], harmonics)

    # use __setitem__ to verify that all harmonics are allowed
    for name, harmonic in zip(names, harmonics):
        composite[name] = harmonic

    return composite


def load_primary_spectrometer(filename):
    data = load(filename)
    return parse_primary_spectrometer(data)


def parse_pulsed_source(source):
    from .primary import PulsedSource
    delay = source['emission_delay']
    duration = source['duration']
    velocities = source['velocities']
    ps = PulsedSource(frequency=source['frequency'], delay=delay, duration=duration, velocities=velocities)
    return ps


def parse_primary_spectrometer(data):
    from scipp import concat
    from .primary import PulsedSource, PrimarySpectrometer
    #
    ps = parse_pulsed_source(data['source'])
    velocities = concat((ps.slowest, ps.fastest), dim='velocity')
    #
    primary = data['primary_spectrometer']
    #
    frequencies = load_frequencies(primary['frequencies'], ps.frequency)
    #
    paths = primary['path_segments']
    choppers = primary['choppers']
    assert len(paths) == len(choppers) + 1  # [Source] === Chopper === Chopper === Chopper --- [Sample]
    pairs = []
    for path, chopper in zip(paths[:-1], choppers):
        pairs.append((load_flight_path(path, velocities), load_chopper(chopper, frequencies)))
    #
    # There is a key data['primary_spectrometer']['sample'], which has information about the sample,
    # but we only need/want the flight path information here
    sample = load_flight_path(paths[-1], velocities)
    #
    return PrimarySpectrometer(ps, pairs, sample)
