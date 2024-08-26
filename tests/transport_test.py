from scipp.testing.assertions import assert_allclose, assert_identical
from scipp import array, scalar
from polystar import Polygon


"""
Construct and test an unrealistic primary spectrometer to test phase space transport.

The source of the primary spectrometer is pulsed, with a repetition period of 12 seconds.
The source emits neutrons for 1 second, with a pulse-delay of 1 second.
Neutrons are emitted with velocities over a large range, but only those between 1/7 and 1/3 m/s
are able to pass the single chopper in the spectrometer.

The chopper is a disc chopper with a radius of 1M m and a aperture 1 m wide and 1 m tall.
The single slit in the chopper is 90 degrees wide, and the chopper rotates at 1/12 Hz.
It is phased to _open_ at 5 seconds (and close at 8 seconds). 
The chopper is located 1 m from the source.

The sample is located 1 m from the chopper.

Note
----
The aperture size of the chopper must be small compared to its radius to ensure that the
opening time is small compared to the open time. The chosen ratio of 1e-6 may still be too large
for this test case, which relies on small deviations for numeric equality checks.
"""


def neutron_velocity():
    return array(values=[1/7, 1/3], unit='m/s', dims=['wavelength'])


def source_frequency_duration_delay_velocities(velocity=None):
    frequency = scalar(1/12, unit='Hz', dtype='float64')
    duration = scalar(1., unit='s', dtype='float64')
    delay = array(values=[1, 1.], unit='s', dims=['wavelength'])
    return frequency, duration, delay, neutron_velocity() if velocity is None else velocity


def make_source(velocity=None):
    from choppera.primary import PulsedSource
    return PulsedSource(*source_frequency_duration_delay_velocities(velocity))


def make_flightpath(which: str, velocity=None):
    from choppera.flightpaths import FlightPath
    velocity = neutron_velocity() if velocity is None else velocity
    return FlightPath(which, velocity, array(values=[1, 1], unit='m', dims=['wavelength']))


def chopper_params():
    from scipp import scalar, Variable
    from numpy import pi, array
    frequency = scalar(-1/12, unit='Hz', dtype='float64')
    phase_to = ('Primary', 1)
    phase = scalar(6.5/12 * 2 * pi, unit='radian', dtype='float64')
    radius = scalar(1e6, unit='m', dtype='float64')
    angles = array([[-45, 45]], dtype='float64')
    windows = Variable(values=angles, unit='degree', dims=['slot', 'window']).to(unit='radian')
    discs = 1
    return frequency, phase_to, phase, radius, windows, discs


def make_aperture():
    from choppera.chopper import Aperture
    width = scalar(1, unit='m', dtype='float64')
    height = scalar(1, unit='m', dtype='float64')
    offset = scalar(1e6 - 1, unit='m', dtype='float64')
    return Aperture(width, height, offset)


def make_disc_chopper(which: str):
    from choppera.chopper import DiscChopper
    frequency, phase_to, phase, radius, windows, discs = chopper_params()
    aperture = make_aperture()
    return DiscChopper(which, radius, frequency, phase_to, phase, aperture, windows, discs)


def make_pairs(velocity=None):
    return [(make_flightpath('Source to Chopper', velocity), make_disc_chopper('Chopper'))]


def make_primary():
    from choppera.primary import PrimarySpectrometer
    velocity = neutron_velocity()
    source = make_source(velocity)
    pairs = make_pairs(velocity)
    sample = make_flightpath('Chopper to Sample')
    return PrimarySpectrometer(source, pairs, sample)


def test_primary_source_phase_space_creation():
    primary = make_primary()
    # The source emits neutrons for 1 second, with a pulse-delay of 1 second.
    # The source velocities should be restricted to (1/7, 1/3) m/s.
    t_min = 1  # second
    t_max = 2  # second
    inverse_v_min = 3  # second/m
    inverse_v_max = 7  # second/m
    vertices = [(t_min, inverse_v_min), (t_max, inverse_v_min), (t_max, inverse_v_max), (t_min, inverse_v_max)]
    expected = Polygon(vertices)

    source_poly = primary.source.tinv_polygon()

    assert source_poly == expected


def test_primary_chopper_phase_space_selection():
    """
    The disc chopper selects all velocities within a time window of approximately 3 seconds, centered at 6.5 seconds.
    """
    from numpy import allclose
    primary = make_primary()
    t_min = 6.5 - 3/2
    t_max = 6.5 + 3/2
    inverse_v_min = 3  # second/m
    inverse_v_max = 7  # second/m
    vertices = [(t_min, inverse_v_max), (t_min, inverse_v_min), (t_max, inverse_v_min), (t_max, inverse_v_max)]
    expected = Polygon(vertices)

    chopper_poly = primary.pairs[0][1].tinv_polygons(minimum_velocity=1/inverse_v_max, maximum_velocity=1/inverse_v_min)
    assert len(chopper_poly) == 1
    assert allclose(chopper_poly[0].area, expected.area)
    assert allclose(chopper_poly[0].vertices, expected.vertices)


def test_primary_project_phase_space_on_source():
    from numpy import allclose
    primary = make_primary()
    regions = primary.project_all_on_source()
    # returned regions are [[source], [chopper]]
    assert len(regions) == 2
    assert len(regions[0]) == 1
    assert len(regions[1]) == 1

    source_t_min = 1  # second
    source_t_max = 2  # second
    inverse_v_min = 3  # second/m
    inverse_v_max = 7  # second/m
    expected_0 = Polygon([(source_t_min, inverse_v_max),
                          (source_t_min, inverse_v_min),
                          (source_t_max, inverse_v_min),
                          (source_t_max, inverse_v_max)])
    assert regions[0][0] == expected_0

    # The chopper polygon has been skewed by its distance
    chopper_t_min = 6.5 - 3 / 2
    chopper_t_max = 6.5 + 3 / 2
    path_length = 1  # m
    expected_1 = Polygon([
        (chopper_t_min - path_length * inverse_v_max, inverse_v_max),
        (chopper_t_min - path_length * inverse_v_min, inverse_v_min),
        (chopper_t_max - path_length * inverse_v_min, inverse_v_min),
        (chopper_t_max - path_length * inverse_v_max, inverse_v_max)
    ])
    assert allclose(regions[1][0].area, expected_1.area)
    assert allclose(regions[1][0].vertices[regions[1][0].border], expected_1.vertices[expected_1.border])


def test_primary_transmitted_phase_space_on_source():
    from numpy import allclose, argmin, sum, abs, sqrt
    primary = make_primary()
    expected = Polygon([(1, 7), (1, 4), (2, 3), (2, 6)], [0, 1, 2, 3])
    space, individuals = primary.project_transmitted_on_source()
    assert len(space) == 1
    space = space[0]
    assert allclose(expected.area, space.area)
    # The opening/closing time of the chopper makes small deviations from the expected case
    # The area is appproximately right, but there are more vertices as a result of the difference.
    # assert allclose(expected.vertices[expected.border], space.vertices[space.border])

    # TODO find a working test to compare the two polygons
    #   This doesn't work because the vertices of space include small deviations from those in expected.
    #   _AND_ polystar says it is not convex, _AND_ the intersection is errantly empty.
    #   Weirdly two identical Polygons do not give an intersection.
    # intersection = space.intersection(expected)
    # assert len(intersection) == 1
    # assert allclose(intersection[0].area, space.area)

    closest = [argmin(sum(abs(expected.vertices - v), axis=1)) for v in space.vertices]
    difference = expected.vertices[closest] - space.vertices
    distance = sqrt(sum(difference * difference, axis=1))
    for x in distance:
        assert x < 1e-6


def test_primary_transmitted_phase_space_on_sample():
    from numpy import allclose
    primary = make_primary()
    expected = Polygon([(15, 7), (9, 4), (8, 3), (14, 6)], [0, 1, 2, 3])
    space, individuals = primary.project_transmitted_on_sample()
    assert len(space) == 1
    space = space[0]
    assert allclose(expected.area, space.area)
    # The opening/closing time of the chopper makes small deviations from the expected case
    assert allclose(expected.vertices[expected.border], space.vertices[space.border])


def test_primary_transmitted_phase_space_on_sample_alt():
    from numpy import allclose
    primary = make_primary()
    expected = Polygon([(15, 7), (9, 4), (8, 3), (14, 6)], [0, 1, 2, 3])
    space = primary.project_on_sample_alternate()
    assert len(space) == 1
    space = space[0]
    assert allclose(expected.area, space.area)
    # The opening/closing time of the chopper makes small deviations from the expected case
    assert allclose(expected.vertices[expected.border], space.vertices[space.border])
