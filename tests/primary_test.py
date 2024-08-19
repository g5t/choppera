from scipp.testing.assertions import assert_allclose, assert_identical

from functools import lru_cache
from choppera.primary import PrimarySpectrometer


def source_velocity():
    from scipp import array
    return array(values=[100, 1e9], unit='m/s', dims=['wavelength'])


def source_frequency_duration_delay_velocities(velocity=None):
    from scipp import scalar, array
    frequency = scalar(14., unit='Hz', dtype='float64')
    duration = scalar(2.7e-3, unit='s', dtype='float64')
    delay = array(values=[1e-7, 0], unit='s', dims=['wavelength'])
    return frequency, duration, delay, source_velocity() if velocity is None else velocity


def make_source(velocity=None):
    from choppera.primary import PulsedSource
    return PulsedSource(*source_frequency_duration_delay_velocities(velocity))


def test_source_creation():
    from scipp import array, allclose
    frequency, duration, delay, velocities = source_frequency_duration_delay_velocities()
    source = make_source()
    assert_allclose(source.frequency, frequency)
    assert_allclose(source.duration, duration * array(values=[1, 1], dims=['wavelength']))
    assert_allclose(source.delay, delay)
    assert_allclose(source.data.coords['velocities'], velocities)


def test_source_equivalence():
    source = make_source()
    source2 = make_source()
    assert source == source2


def make_pairs(velocity=None):
    from .chopper_test import make_chopper
    from .flightpaths_test import make_flightpath
    return [(make_flightpath(velocity), make_chopper()) for _ in range(3)]


def make_sample(velocity=None):
    from .flightpaths_test import make_flightpath
    return make_flightpath(velocity)


def make_primary():
    velocity = source_velocity()
    source = make_source(velocity)
    pairs = make_pairs(velocity)
    sample = make_sample(velocity)
    return PrimarySpectrometer(source, pairs, sample)


def test_primary_creation():
    velocity = source_velocity()
    source = make_source(velocity)
    pairs = make_pairs(velocity)
    sample = make_sample(velocity)
    primary = make_primary()
    assert primary.source == source
    assert primary.pairs == pairs
    assert primary.sample == sample


def test_primary_equivalence():
    primary = make_primary()
    primary2 = make_primary()
    assert primary == primary2


@lru_cache
def primary_function(primary: PrimarySpectrometer):
    return 1


def test_primary_argument_function_caching():
    primary = make_primary()
    primary2 = make_primary()
    assert primary_function(primary) == 1
    assert primary_function(primary) == 1
    assert primary_function(primary) == 1
    assert primary_function(primary2) == 1

    cache = primary_function.cache_info()
    assert cache.hits == 3
    assert cache.misses == 1