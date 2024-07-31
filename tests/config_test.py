def mock_config():
    mock = """name: MOCK
# some comment that doesn't get parsed

source:
  name: The moderator
  frequency: 50 Hz
  duration: 0.1 µs
  velocities: (100, 1e9) m/s
  emission_delay: (0.1, 0) µs

primary_spectrometer:
  frequencies:
    - name: Main
      highest_harmonic: 15
  path_segments:
    - name: Moderator to Chopper
      length: 3.14 m
      guide:
        velocities: (100, 1e9) m/s
    - name: Chopper to Sample
      length: 4 m
  choppers:
    - name: Chopper
      position: 3.14 m
      opening: 10 degrees
      radius: 350 mm
      discs: 2
      frequency:
        name: Main
      aperture:
        width: 30 mm
        height: 48 mm
        offset: 302 mm
  sample:
    position: 7.14 m
"""
    return mock


def test_config_is_importable():
    import choppera.config


def test_mock_config():
    from scipp import scalar, array, allclose
    from choppera.config import parse
    mock = mock_config()
    parsed = parse(mock)
    assert parsed['name'] == 'MOCK'

    assert 'source' in parsed
    source = parsed['source']
    assert source['name'] == 'The moderator'
    assert allclose(source['frequency'], scalar(50., unit='Hz'))
    assert allclose(source['duration'], scalar(1e-7, unit='s'))
    assert allclose(source['velocities'], array(values=[100, 1e9], unit='m/s', dims=['wavelength']))
    assert allclose(source['emission_delay'], array(values=[1e-7, 0], unit='s', dims=['wavelength']))

    assert 'primary_spectrometer' in parsed
    spec = parsed['primary_spectrometer']
    assert spec['frequencies'][0]['name'] == 'Main'
    assert spec['frequencies'][0]['highest_harmonic'] == 15

    assert 'path_segments' in spec
    assert len(spec['path_segments']) == 2

    p0 = spec['path_segments'][0]
    assert p0['name'] == 'Moderator to Chopper'
    assert allclose(p0['length'], scalar(3.14, unit='m'))
    assert allclose(p0['guide']['velocities'], array(values=[100, 1e9], unit='m/s', dims=['wavelength_limit']))

    p1 = spec['path_segments'][1]
    assert p1['name'] == 'Chopper to Sample'
    assert allclose(p1['length'], scalar(4.0, unit='m'))

    assert 'choppers' in spec
    assert len(spec['choppers']) == 1

    c0 = spec['choppers'][0]
    assert c0['name'] == 'Chopper'
    assert allclose(c0['position'], scalar(3.14, unit='m'))
    assert allclose(c0['opening'], scalar(10.0, unit='degrees'))
    assert allclose(c0['radius'], scalar(350., unit='mm'))
    assert c0['discs'] == 2
    assert c0['frequency']['name'] == 'Main'
    assert allclose(c0['aperture']['width'], scalar(30., unit='mm'))
    assert allclose(c0['aperture']['height'], scalar(48., unit='mm'))
    assert allclose(c0['aperture']['offset'], scalar(302., unit='mm'))

    assert allclose(spec['sample']['position'], scalar(7.14, unit='m'))


def test_pulsed_source():
    from scipp import scalar, array, allclose
    from choppera.config import parse, parse_pulsed_source
    mock = parse(mock_config())['source']
    source = parse_pulsed_source(mock)
    assert source.frequency == scalar(50.0, unit='Hz')
    assert source.slowest == scalar(100.0, unit='m/s')
    assert source.fastest == scalar(1e9, unit='m/s')
    phase = source.early_late()
    assert allclose(phase.velocity, array(values=[100, 1e9], unit='m/s', dims=['wavelength']))
    assert allclose(phase.left, array(values=[1e-7, 0], unit='s', dims=['wavelength']))
    assert allclose(phase.right, array(values=[2e-7, 1e-7], unit='s', dims=['wavelength']))


def test_primary_spectrometer():
    from choppera.config import parse, parse_primary_spectrometer
    mock = parse(mock_config())
    spec = parse_primary_spectrometer(mock)
    # TODO extend this beyond just verifying that it does not throw an error