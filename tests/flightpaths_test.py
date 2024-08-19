from scipp.testing.assertions import assert_allclose, assert_identical


def name_velocity_nominal(velocity=None):
    from scipp import array
    if velocity is None:
        velocity = array(values=[100, 1e9], unit='m/s', dims=['wavelength'])
    nominal = array(values=[3.14, 3.14], unit='m', dims=['wavelength'])
    return 'Path', velocity, nominal


def make_flightpath(velocity=None):
    from choppera.flightpaths import FlightPath
    return FlightPath(*name_velocity_nominal(velocity))


def test_flightpath_creation():
    from scipp import allclose
    name, velocity, nominal = name_velocity_nominal()
    flightpath = make_flightpath()
    assert flightpath.name == name
    assert_allclose(flightpath.velocity, velocity)
    assert_allclose(flightpath.nominal, nominal)


def test_flightpath_equivalence():
    flightpath = make_flightpath()
    flightpath2 = make_flightpath()
    assert flightpath == flightpath2
