from scipp.testing.assertions import assert_allclose, assert_identical
from scipp import array, scalar
from polystar import Polygon


"""
Construct and test a slightly realistic primary spectrometer to test phase space transport.

The source of the primary spectrometer is pulsed, with a repetition period of 100 milliseconds.
The source emits neutrons for 10 milliseconds, with no pulse-delay.
Neutrons are emitted with velocities over a large range, but only those between ~75 and 1000 m/s
are worth considering, and only velocities above ~333 m/s make it past the first three choppers.

All choppers are disc choppers with a radius of 1M m and a aperture 1 m wide and 1 m tall.
All choppers rotate at 10 Hz, to match the source frequency.
The choppers have properties
|distance|opening time|closing time|angle   |
|--------|------------|------------|--------|
| 10.0 m | -20 ms     | 30 ms      | 180    |
| 10.0 m |  20 ms     | 70 ms      | 180    |
| 15.0 m |  20 ms     | 50 ms      | 3 * 36 |
| 20.0 m | -50 ms     | 40 ms      | 9 * 36 |

The sample is located at the 30 m mark.
"""


def neutron_velocity():
    from scipp import scalar
    v = scalar(10., unit='m', dtype='float64') / array(values=[130, 10.], unit='ms', dims=['wavelength'])
    return v.to(unit='m/s')


def source_frequency_duration_delay_velocities(velocity=None):
    frequency = (1 / scalar(100, unit='ms', dtype='float64')).to(unit='Hz')
    duration = scalar(10., unit='ms', dtype='float64')
    delay = array(values=[0, 0.], unit='s', dims=['wavelength'])
    return frequency, duration, delay, neutron_velocity() if velocity is None else velocity


def make_source(velocity=None):
    from choppera.primary import PulsedSource
    return PulsedSource(*source_frequency_duration_delay_velocities(velocity))


def make_flightpath(which: str, distance: float, velocity=None):
    from choppera.flightpaths import FlightPath
    velocity = neutron_velocity() if velocity is None else velocity
    return FlightPath(which, velocity, array(values=[distance, distance], unit='m', dims=['wavelength']))


def chopper_params(opening, closing, angle):
    from scipp import scalar, Variable
    from numpy import pi, array
    frequency = (1 / scalar(100, unit='ms', dtype='float64')).to(unit='Hz')
    phase_to = ('Primary', 1)
    centre = scalar((opening + closing) / 2, unit='ms', dtype='float64')
    phase = (centre * frequency * scalar(2 * pi, unit='radian')).to(unit='radian')
    radius = scalar(1e6, unit='m', dtype='float64')
    angles = array([[-angle / 2, angle / 2]], dtype='float64')
    windows = Variable(values=angles, unit='degree', dims=['slot', 'window']).to(unit='radian')
    discs = 1
    return frequency, phase_to, phase, radius, windows, discs


def make_aperture():
    from choppera.chopper import Aperture
    width = scalar(1, unit='m', dtype='float64')
    height = scalar(1, unit='m', dtype='float64')
    offset = scalar(1e6 - 1, unit='m', dtype='float64')
    return Aperture(width, height, offset)


def make_disc_chopper(which: str, opening, closing, angle):
    from choppera.chopper import DiscChopper
    frequency, phase_to, phase, radius, windows, discs = chopper_params(opening, closing, angle)
    aperture = make_aperture()
    return DiscChopper(which, radius, frequency, phase_to, phase, aperture, windows, discs)


def make_pairs(velocity=None):
    return [
        (make_flightpath('Source to Chopper 1', 10., velocity), make_disc_chopper('Chopper 1', -20, 30, 180)),
        (make_flightpath('Chopper 1 to Chopper 2', 0, velocity), make_disc_chopper('Chopper 2', 20, 70, 180)),
        (make_flightpath('Chopper 2 to Chopper 3', 15.00-10, velocity), make_disc_chopper('Chopper 3', 20, 50, 3 * 36)),
        (make_flightpath('Chopper 3 to Chopper 4', 20.00-15, velocity), make_disc_chopper('Chopper 4', -50, 40, 9 * 36))
    ]


def make_primary():
    from choppera.primary import PrimarySpectrometer
    velocity = neutron_velocity()
    source = make_source(velocity)
    pairs = make_pairs(velocity)
    sample = make_flightpath('Chopper to Sample', 30-20., velocity)
    return PrimarySpectrometer(source, pairs, sample)


def test_primary_source_phase_space_creation():
    primary = make_primary()
    # The source emits neutrons for 10 millisecond, with no pulse-delay.
    # The source velocities should be restricted to (1/7, 1/3) m/s.
    t_min = 0  # second
    t_max = 0.01  # second
    inverse_v_min = 0.001  # second/m
    inverse_v_max = 0.013  # second/m
    vertices = [(t_min, inverse_v_min), (t_max, inverse_v_min), (t_max, inverse_v_max), (t_min, inverse_v_max)]
    expected = Polygon(vertices)

    source_poly = primary.source.tinv_polygon()

    assert source_poly == expected


def test_primary_chopper_phase_space_selection():
    from numpy import allclose
    primary = make_primary()
    t0_ms = [[5, 105], [45], [35], [-5, 95]]
    tw_ms = [[50, 50], [50], [30], [90, 90]]
    inverse_v_min = 0.001  # second/m
    inverse_v_max = 0.013  # second/m
    for t0, tw, pair in zip(t0_ms, tw_ms, primary.pairs):
        chopper_poly = pair[1].tinv_polygons(minimum_velocity=1/inverse_v_max, maximum_velocity=1/inverse_v_min)
        assert len(chopper_poly) == len(t0)
        for centre, width, poly in zip(t0, tw, chopper_poly):
            t_min = (centre - width / 2) / 1000  # second
            t_max =(centre + width / 2) / 1000  # second
            vertices = [(t_min, inverse_v_max), (t_min, inverse_v_min), (t_max, inverse_v_min), (t_max, inverse_v_max)]
            expected = Polygon(vertices)
            assert allclose(poly.area, expected.area)
            assert allclose(poly.vertices[poly.border], expected.vertices[expected.border])



def test_primary_project_phase_space_on_source():
    from numpy import allclose
    primary = make_primary()
    regions = primary.project_all_on_source()  # returned regions are [[source], [ch0], [ch1], [ch2], [ch3]]

    t0_ms = [[5, 105], [45, 145], [35, 135], [-5, 95, 195, 295]]
    tw_ms = [50, 50, 30, 90]
    dists = [10, 10, 15, 20]

    assert len(regions) == len(t0_ms) + 1
    for region, t0 in zip(regions[1:], t0_ms):
        assert len(region) == len(t0)

    source_t_min = 0  # second
    source_t_max = 10/1000  # second
    inverse_v_min = 0.001  # second/m
    inverse_v_max = 0.013  # second/m
    expected_0 = Polygon([(source_t_min, inverse_v_max),
                          (source_t_min, inverse_v_min),
                          (source_t_max, inverse_v_min),
                          (source_t_max, inverse_v_max)])
    assert regions[0][0] == expected_0

    # The chopper polygons have been skewed by their distance

    for t0, tw, dist, region in zip(t0_ms, tw_ms, dists, regions[1:]):
        for centre, poly in zip(t0, region):
            chopper_t_min = (centre - tw / 2) / 1000
            chopper_t_max = (centre + tw / 2) / 1000
            expected = Polygon([
                (chopper_t_min - dist * inverse_v_max, inverse_v_max),
                (chopper_t_min - dist * inverse_v_min, inverse_v_min),
                (chopper_t_max - dist * inverse_v_min, inverse_v_min),
                (chopper_t_max - dist * inverse_v_max, inverse_v_max)
            ])
            assert allclose(poly.area, expected.area)
            assert allclose(poly.vertices[poly.border], expected.vertices[expected.border])


def test_primary_transmitted_phase_space_on_source():
    from numpy import allclose
    primary = make_primary()
    space, individuals = primary.project_transmitted_on_source()
    lower = Polygon([(0.00, 0.0020), (0.01, 0.0010), (0.01, 0.0015)], [0, 1, 2])
    upper = Polygon([(0.00, 0.0030), (0.00, 0.0025), (0.01, 0.0020)], [0, 1, 2])
    assert len(space) == 2
    assert allclose(lower.area, space[0].area)
    assert allclose(upper.area, space[1].area)
    # The opening/closing time of the chopper makes small deviations from the expected case
    # The area is approximately right, but there are more vertices as a result of the difference.

    for expected, poly in zip([lower, upper], space):
        assert allclose(expected.area, poly.area)
        intersection = poly.intersection(expected)
        assert len(intersection) == 1
        assert allclose(intersection[0].area, expected.area)
        assert intersection[0] == expected

#
#
# def test_primary_transmitted_phase_space_on_sample():
#     from numpy import allclose
#     primary = make_primary()
#     expected = Polygon([(15, 7), (9, 4), (8, 3), (14, 6)], [0, 1, 2, 3])
#     space, individuals = primary.project_transmitted_on_sample()
#     assert len(space) == 1
#     space = space[0]
#     assert allclose(expected.area, space.area)
#     # The opening/closing time of the chopper makes small deviations from the expected case
#     assert allclose(expected.vertices[expected.border], space.vertices[space.border])
#
#
# def test_primary_transmitted_phase_space_on_sample_alt():
#     from numpy import allclose
#     primary = make_primary()
#     expected = Polygon([(15, 7), (9, 4), (8, 3), (14, 6)], [0, 1, 2, 3])
#     space = primary.project_on_sample_alternate()
#     assert len(space) == 1
#     space = space[0]
#     assert allclose(expected.area, space.area)
#     # The opening/closing time of the chopper makes small deviations from the expected case
#     assert allclose(expected.vertices[expected.border], space.vertices[space.border])
