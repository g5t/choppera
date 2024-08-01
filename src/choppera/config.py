# SPDX-FileCopyrightText: 2024-present Gregory Tucker <gregory.tucker@ess.eu>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to read and understand YAML configuration files"""
from __future__ import annotations

from strictyaml import ScalarValidator, Map, Seq, Str, Optional, Int, Float
from strictyaml.validators import Validator, SeqValidator
from dataclasses import dataclass


def separate_value_unit(x: str):
    """Separate a value and its unit using scipp.Unit parsing

    Expected valid inputs are of one of three forms:
    1. A constant and a unit
        CONSTANT UNIT
    2. A tuple of constants and a unit
        (CONSTANT, CONSTANT, ..., CONSTANT) UNIT
    3. A list of constants and a unit
        [CONSTANT, CONSTANT, ..., CONSTANT] UNIT

    As the unit is determined greedily and scipp.Units are allowed to have scaling factors,
    the value and unit may not be as expected if the final ')' or ']' are missing from the line
    or otherwise a constant directly precedes the unit, e.g.,
        CONSTANT CONSTANT UNIT

    """
    from scipp import Unit, UnitError
    # split on spaces
    x_split = x.split()
    # try to find a valid unit in the string (greedily)
    loop = True
    index = 1  # there (should) always be a number of some sort before the unit
    unit = None
    while loop and index < len(x_split):
        try:
            unit = Unit(' '.join(x_split[index:]))
            loop = False
        except UnitError:
            index += 1
    return ' '.join(x_split[:index]), unit


def is_simple_tree(tree):
    """Recursively check if the abstract syntax tree is built out of simple (self-contained) elements

    Allowed contents are lists, tuples, and numeric constants.

    This is a more-restrictive pre-check for the AST module literal_eval function, which currently only allows,
    "strings, bytes, numbers, tuples, lists, dicts, sets, booleans, None and Ellipsis"
    """
    from ast import Expression, Tuple, List, Constant
    # since the parse _was_ done in 'eval' mode, the root of the tree is an Expression
    # (If 'expr' was used it might be a Module with one or more Expr in a list at tree.body)
    if isinstance(tree, Expression):
        tree = tree.body
    if isinstance(tree, (Tuple, List)):
        return all(is_simple_tree(x) for x in tree.elts)
    if isinstance(tree, Constant):
        return isinstance(tree.value, (int, float))
    return False


def parse_value_ast(x: str,
                    expected_unit: str | None = None,
                    expected_shape: list[int] | None = None,
                    dims: list[str] | None = None,
                    dtype: str = 'float64'
                    ):
    """Use the python Abstract Syntax Tree to more-safely load values"""
    from ast import parse, literal_eval
    from scipp import scalar, Variable, UnitError
    from numpy import array as n_array
    value, unit = separate_value_unit(x)
    tree = parse(value, mode='eval')
    if is_simple_tree(tree):
        value = literal_eval(tree)

    if isinstance(value, str):
        return value

    check_shape = False
    if isinstance(value, (float, int)):
        if dims is not None and len(dims) > 0:
            value = Variable(values=[value], unit=unit, dims=dims, dtype=dtype)
            check_shape = True
        else:
            value = scalar(value, unit=unit, dtype=dtype)
            if expected_shape is not None and len(expected_shape) > 0:
                raise RuntimeError(f"Expected {len(expected_shape)}-D data but found scalar value")
    elif isinstance(value, (tuple, list)):
        if dims is None:
            dims = [f'dim{n}' for n in range(n_array(value).ndim)]
        value = Variable(values=value, unit=unit, dims=dims, dtype=dtype)
        check_shape = True
    else:
        raise RuntimeError("Value to parse should be a string, a numeric constant, or a tuple or list of constants")
    if expected_unit is not None and value.unit != expected_unit:
        try:
            value = value.to(unit=expected_unit)
        except UnitError:
            found = value.unit if value.unit else "None"
            raise RuntimeError(f"Expected unit {expected_unit} does not match {found} which is not convertible to it")

    if check_shape and expected_shape is not None:
        for shape, dim in zip(expected_shape, dims):
            if value.sizes[dim] != shape:
                raise RuntimeError(f"Expected {shape} elements along {dim} but found {value.sizes[dim]}")
    return value


class List(ScalarValidator):
    def validate_scalar(self, chunk):
        value = parse_value_ast(chunk.contents)
        return tuple(value.values) if value.ndim else value.value

    def to_yaml(self, data):
        return f"{data}"


class ScippScalar(ScalarValidator):
    def __init__(self, unit: str):
        self.unit = unit

    def validate_scalar(self, chunk):
        return parse_value_ast(chunk.contents, expected_unit=self.unit)

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
                raise RuntimeError(f"Cannot convert {data} to a YAML scalar")
        return f"{data} {self.unit}"


class ScippVariable(ScippScalar):
    def __init__(self, unit: str, dims: list[str], shape: list[int] | None = None):
        super().__init__(unit)
        self.dims = dims
        self.shape = shape

    def validate_scalar(self, chunk):
        return parse_value_ast(chunk.contents, expected_unit=self.unit, expected_shape=self.shape, dims=self.dims)


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
    'frequency': ScippScalar('Hz'),
    'duration': ScippScalar('s'),
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
    'length': ScippScalar('m'),
    Optional('guide'): Map({
        'velocities': ScippVariable('m/s', ['wavelength_limit'], [2]),
        Optional('short'): ScippVariable('m', ['wavelength_limit'], [2]),
        Optional('long'): ScippVariable('m', ['wavelength_limit'], [2]),
    }),
})

CHOPPER_SCHEMA = Map({
    'name': Str(),
    'position': ScippScalar('m'),
    'opening': ScippScalar('degrees'),
    'radius': ScippScalar('mm'),
    Optional('discs'): Int(),
    Optional('slots'): Int(),
    Optional('frequency'): Map({'name': Str(), Optional('multiplier'): Int()}),
    'aperture': Map({
        'width': ScippScalar('mm'),
        'height': ScippScalar('mm'),
        Optional('offset'): ScippScalar('mm')
    })
})

SAMPLE_SCHEMA = Map({'position': ScippScalar('m')})

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
