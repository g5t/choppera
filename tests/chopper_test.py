from scipp.testing.assertions import assert_allclose, assert_identical

def width_height_offset():
    from scipp import scalar
    width = scalar(30, unit='mm', dtype='float64')
    height = scalar(48, unit='mm', dtype='float64')
    offset = scalar(302, unit='mm', dtype='float64')
    return width, height, offset


def name_frequency_phase_to_phase():
    from scipp import scalar
    frequency = scalar(50, unit='Hz', dtype='float64')
    phase_to = ('Primary', 1)
    phase = scalar(1., unit='radian', dtype='float64')
    return 'Chopper', frequency, phase_to, phase


def radius_angles_windows_discs():
    from numpy import array
    from scipp import scalar, Variable
    radius = scalar(350, unit='mm', dtype='float64').to(unit='m')
    angles = array([[-10, 10], [20, 30]], dtype='float64')
    windows = Variable(values=angles, unit='degree', dims=['slot', 'window']).to(unit='radian')
    return radius, angles, windows, 2


def make_aperture():
    from choppera.chopper import Aperture
    width, height, offset = width_height_offset()
    return Aperture(width, height, offset)


def make_chopper():
    from choppera.chopper import Chopper
    name, frequency, phase_to, phase = name_frequency_phase_to_phase()
    aperture = make_aperture()
    return Chopper(name, frequency, phase_to, phase, aperture)


def make_disc_chopper():
    from choppera.chopper import DiscChopper
    name, frequency, phase_to, phase = name_frequency_phase_to_phase()
    aperture = make_aperture()
    radius, angles, windows, discs = radius_angles_windows_discs()
    return DiscChopper(name, radius, frequency, phase_to, phase, aperture, windows, discs)


def test_aperture_creation():
    width, height, offset = width_height_offset()
    aperture = make_aperture()
    assert_identical(aperture.width, width)
    assert_identical(aperture.height, height)
    assert_identical(aperture.offset, offset)


def test_aperture_equality():
    aperture = make_aperture()
    aperture2 = make_aperture()
    assert aperture == aperture2


def test_chopper_creation():
    name, frequency, phase_to, phase = name_frequency_phase_to_phase()
    aperture = make_aperture()
    chopper = make_chopper()
    assert chopper.name == name
    assert_identical(chopper.frequency, frequency)
    assert_identical(chopper.phase, phase)
    assert chopper.aperture == aperture
    assert_identical(chopper.period, 1/frequency)


def test_chopper_equality():
    chop_a = make_chopper()
    chop_b = make_chopper()
    assert chop_a == chop_b


def test_disc_chopper_creation():
    from scipp import allclose
    name, frequency, phase_to, phase = name_frequency_phase_to_phase()
    aperture = make_aperture()
    radius, angles, windows, discs = radius_angles_windows_discs()
    chop_a = make_disc_chopper()
    assert chop_a.name == name
    assert_identical(chop_a.frequency, frequency)
    assert_identical(chop_a.phase, phase)
    assert chop_a.aperture == aperture
    assert_identical(chop_a.period, 1/frequency)
    assert_identical(chop_a.radius, radius)
    assert chop_a.discs == discs
    assert allclose(chop_a.windows, windows)


def test_disc_chopper_equality():
    chop_a = make_disc_chopper()
    chop_b = make_disc_chopper()
    assert chop_a == chop_b
