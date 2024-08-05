from scipp import Variable
from scippnexus import Group


def guess_group_geometry(group: Group):
    guesses = 'OFF_GEOMETRY', 'OFF', 'GEOMETRY'
    children = 'vertices', 'faces', 'winding_order'
    for x in guesses + tuple(y.lower() for y in guesses):
        if x in group and all(y in group[x] for y in children):
            return x
    raise RuntimeError(f'Specify which of {list(group)} contains OFF geometry fields {children}')


def entry_exit_size(group: Group, geom: str | None = None):
    """ Find the extent of the guide at its entry and exit planes, using the dot product to find the two extremes

    Note
    ____
    This method only works for guides that are symmetric about the z-axis (no curvature) but has the advantage of
    not relying on the entrance and exit faces being represented in the OFF structure.
    """
    from scipp import dot, vector, min, max
    if geom is None:
        geom = guess_group_geometry(group)

    def width_height(v):
        v_x = dot(v, vector(value=[1., 0, 0], unit='1'))
        v_y = dot(v, vector(value=[0, 1., 0], unit='1'))
        return max(v_x) - min(v_x), max(v_y) - min(v_y)

    vertices = group[f'{geom}/vertices'][...]
    v_z = dot(vertices, vector(value=[0, 0, 1.], unit='1'))
    z_min, z_max = min(v_z), max(v_z)
    return width_height(vertices[v_z == z_min]), width_height(vertices[v_z == z_max])


def nexus_off_to_polyhedron(group: Group, geom: str | None = None):
    from numpy import hstack
    from polystar import Polyhedron
    if geom is None:
        geom = guess_group_geometry(group)
    face_starts = group[f'{geom}/faces'][...].values  # immediately use the numpy array since we're going to index
    winding_order = group[f'{geom}/winding_order'][...].values
    face_ends = hstack((face_starts[1:], len(winding_order)))
    faces = [winding_order[start:stop] for start, stop in zip(face_starts, face_ends)]
    vertices = group[f'{geom}/vertices'][...].values

    poly = Polyhedron(vertices, faces)
    return poly


def entry_exit_size_curved(group: Group):
    """If a guide is curved, the simpler approach may not suffice, but this _REQUIRES_ that the faces are represented

    Note
    ----
    If the group represents an Elliptic Guide (an EllipticGuideGravity, or similar) translated to NeXus by moreniius
    the input and output faces _ARE NOT PRESENT_ in the OFF, so this method will fail.
    """
    from numpy import argmax
    from scipp import dot, cross, vector, min, max, sum, concat, sqrt
    assert 'NX_class' in group.attrs and group.attrs['NX_class'] == 'NXguide'
    geom = guess_group_geometry(group)
    poly = nexus_off_to_polyhedron(group, geom)
    v = group[f'{geom}/vertices'][...]
    y = vector(value=[0, 1., 0], unit='1')
    z = vector(value=[0, 0, 1.], unit='1')

    faces = [v[f] for f in poly.faces]
    centers = concat([sum(f) / len(f) for f in faces], 'face')
    c_z = dot(centers, z)
    z_min, z_max = min(c_z), max(c_z)

    c_z_min = c_z == z_min
    c_z_max = c_z == z_max
    if sum(c_z_min).value != 1 or sum(c_z_max).value != 1:
        print(f"Non-singular minimum/maximum face along z")
    closest = faces[argmax(c_z_min.values)]
    farthest = faces[argmax(c_z_max.values)]

    def width_height(face):
        v0 = face[1] - face[0]
        v1 = face[2] - face[1]
        n = cross(v0, v1)
        n /= sqrt(dot(n, n))
        x = cross(y, n)
        x /= sqrt(dot(x, x))
        v_x = dot(face, x)
        v_y = dot(face, y)
        return max(v_x) - min(v_x), max(v_y) - min(v_y)

    return width_height(closest), width_height(farthest)


def determine_name(inst: Group, name: str | None, options: list, type_name: str):
    if name is not None and name in inst:
        return name
    found = {x for x in inst if type_name in x.lower()}
    for option in options:
        found.update(set(inst[option]))
    if len(found) != 1:
        raise RuntimeError(f"Could not determine unique {type_name} name, provide correct input")
    return list(found)[0]


def named_index(name: str):
    return int(name.split('_')[0])


def path_length(positions: Variable):
    from scipp import dot, sqrt, sum, DType
    assert positions.dtype == DType.vector3
    assert positions.ndim == 1, "Only one-dimensional paths can have a length"
    dim = positions.dims[0]
    diff = positions[dim, 1:] - positions[dim, :-1]
    return sum(sqrt(dot(diff, diff)))


def scalar_property(disk: Group, name: str):
    from numpy import std
    assert name in disk, f"The '{name}' property is optional in NeXus but required here"
    value = disk[name][...]
    if value.ndim > 0:
        assert std(value.values) == 0, f"Non-singular '{name}' not supported (std({value}) != 0)"
        return value[0]
    return value


def optional_scalar_property(disk: Group, name: str, value: Variable):
    return scalar_property(disk, name) if name in disk else value


def one_dim_property(disk: Group, name: str):
    from scipp import squeeze
    assert name in disk, f"The '{name}' property is optional in NeXus but required here"
    value = disk[name][...]
    if value.ndim > 1:
        value = squeeze(value)
    if value.ndim != 1:
        raise RuntimeWarning(f"Expected a one-dimensional property but {name} is {value.ndim}-D")
    return value


def pulsed_source(source: Group, frequency: Variable, duration: Variable, delay: Variable, velocity_range: Variable):
    """Construct a pulsed source from NeXus information

    Parameters
    ----------
    source: scippnexus.Group
        The NXsource, NXmoderator, or other group of a HDF5 NeXus file representing the source of neutrons
    frequency: scipp.Variable
        The scalar repetition frequency of the source pulses
    duration: scipp.Variable
        The scalar duration of each source pulse
    delay: scipp.Variable
        The time after each pulse start before neutrons leave the source component, likely wavelength-dependent
    velocity_range: scipp.Variable
        At least the minimal and maximal useful neutron velocities emitted by the source. If delay is wavelength
        dependent, the dimensionality and size of velocity and delay must match.

    Returns
    -------
    A choppera.PulsedSource
    """
    from .primary import PulsedSource
    if 'type' in source:
        assert 'Neutron' in source['type'][...]
    if 'frequency' in source:
        frequency = scalar_property(source, 'frequency')
    elif 'period' in source:
        frequency = 1/scalar_property(source, 'period')
    if 'pulse_width' in source:
        duration = scalar_property(source, 'pulse_width')
    if 'distribution' in source:
        raise RuntimeError("The source distribution could be used to set velocity_range, but is not at this moment")
    return PulsedSource(frequency, duration, delay, velocity_range)


def primary_spectrometer(inst: Group,
                         source_name: str | None = None,
                         sample_name: str | None = None,
                         source_frequency: Variable | None = None,
                         source_duration: Variable | None = None,
                         source_delay: Variable | None = None,
                         velocity_range: Variable | None = None):
    """Construct a primary spectrometer from NeXus information

    Note
    ----
    It is assumed that all guide elements and choppers are present, and that all component names start with
    a unique path positioning index which is monotonic increasing of the form NNN_{component name} where NNN is a zero
    padded integer such that all components have the same-width index prefix.

    Parameters
    ----------
    inst:  scippnexus.Group
        The NXinstrument entry of a NeXus HDF5 file as opened by scippnexus, but without having loaded its data
    source_name: str | None
        The name of NXsource, NXmoderator, or other 'source' group in the NXinstrument
    sample_name: str | None
        The name of the NXsample or other 'sample' group in the NXinstrument
    source_frequency: scipp.Variable | None
        The repetition frequency of the PulsedSource, should be scalar. Will be replaced by 14 Hz if None
    source_duration: scipp.Variable | None
        The length of each pulse from the source, should be scalar. Will be replaced by 3 microseconds if None
    source_delay: scipp.Variable | None
        The time for neutrons to exit the source (or moderator). Either wavelength dependent or scalar.
    velocity_range: scipp.Variable | None
        The useful range of neutron velocities which leave the source/moderator (and are transmitted through the guide)

    Returns
    -------
    A complete choppera.PrimarySpectrometer object
    """
    from scippnexus import NXsource, NXmoderator, NXguide, NXdisk_chopper, NXfermi_chopper, NXsample
    from scippnexus import compute_positions
    from scipp import concat, allclose, sqrt, scalar

    from .chopper import Aperture, DiscChopper
    from .flightpaths import FlightPath
    from .primary import PrimarySpectrometer

    assert 'NX_class' in inst.attrs and inst.attrs['NX_class'] == 'NXinstrument'

    source_name = determine_name(inst, source_name, [NXsource, NXmoderator], 'source')
    sample_name = determine_name(inst, sample_name, [NXsample], 'sample')
    if source_frequency is None:
        source_frequency = scalar(14.0, unit='Hz')
    if source_duration is None:
        source_duration = scalar(3., unit='us').to(unit='s')
    if source_delay is None:
        source_delay = scalar(0., unit='s')
    if velocity_range is None:
        velocity_range = Variable(values=[10, 1e9], unit='m/s', dims=['wavelength'])

    source = pulsed_source(inst[source_name], source_frequency, source_duration, source_delay, velocity_range)

    disk_names = list(inst[NXdisk_chopper])
    fermi_names = list(inst[NXfermi_chopper])
    assert len(fermi_names) == 0, "Fermi choppers are not yet supported by this module"

    def pos(obj):
        return compute_positions(obj[...])['position']

    def length_between(precede: str, follow: str):
        pi, fi = [named_index(x) for x in (precede, follow)]
        b = [v for k, v in inst[NXguide].items() if pi < named_index(k) < fi]
        return b, path_length(concat((pos(inst[precede]), *[pos(x) for x in b], pos(inst[follow])), dim='path'))

    last = source_name
    pairs = []
    for disk in disk_names:
        # find the path length from 'last' to this disk, passing through all guide elements in between
        between, length = length_between(last, disk)

        # pull out relevant chopper details:
        radius = scalar_property(inst[disk], 'radius').to(unit='m')
        phase = scalar_property(inst[disk], 'phase').to(unit='radian')
        frequency = scalar_property(inst[disk], 'rotation_speed').to(unit='Hz')
        edges = one_dim_property(inst[disk], 'slit_edges')
        assert len(edges) % 2 == 0, "There must be an equal number of slit edges"
        windows = edges.fold(dim=edges.dim[0], sizes={'slot': -1, 'window': 2}).to(unit='radian')

        # get the aperture size: guess that is the size of the last guide, or the same as the previous chopper?
        # better might be to check the _next_ guide entrance too
        if len(between):
            _, (width, height) = entry_exit_size(between[-1][...])
        elif len(pairs):
            width, height = pairs[-1][1].aperture.width, pairs[-1][1].aperture.height
        else:
            raise RuntimeError("Can not set chopper aperture without preceding guide or chopper")

        # Calculate the offset that puts the aperture corner at the disk radius:
        limit = sqrt(radius*radius - width * width / 4) - height
        offset = optional_scalar_property(inst[disk], 'offset', limit).to(unit='m')
        if not allclose(offset, limit) and offset > limit:
            print("Disk offset is too large to fully illuminate aperture")

        # if slit height isn't provided we're guessing a lot
        min_sh = radius - offset
        slit_height = optional_scalar_property(inst[disk], 'slit_height', min_sh).to(unit='m')
        if not allclose(min_sh, slit_height) and min_sh > slit_height:
            print('The disk slit_height is too small to fully illuminate the aperture')

        aperture = Aperture(width, height, offset)
        disc = DiscChopper(disk, radius=radius, frequency=frequency, phase=phase, aperture=aperture, windows=windows)
        path = FlightPath(f'{last} to {disk}', velocity_range, length)
        pairs.append((path, disc))

        # update the last reference to be the current disk
        last = disk

    # final flight path from the last chopper to the sample
    between, length = length_between(last, sample_name)
    final_path = FlightPath(f'{last} to {sample_name}', velocity_range, length)

    return PrimarySpectrometer(source, pairs, final_path)


def primary_periods(*args, **kwargs):
    from numpy import hstack
    from scipp import Variable, min, max
    primary = primary_spectrometer(*args, **kwargs)
    t_vs_inverse_v_polys_at_sample, s_layers = primary.project_transmitted_on_sample()
    # extract all times from the polygons
    times = Variable(values=hstack([p.vertices[:, 0] for p in t_vs_inverse_v_polys_at_sample]), dims=['time'], unit='s')
    min_time = min(times)
    max_time = max(times)
    period = (1 / primary.source.frequency).to(unit='s')
    assert max_time - min_time <= period, "The time range is too large for the source frequency"
    n = min_time % period
    delta = min_time - n * period
    return n, delta


def primary_shortest_time(inst: Group,
                          source: str,
                          sample: str,
                          frequency: Variable,
                          duration: Variable,
                          delay: Variable,
                          velocity: Variable):
    from numpy import hstack
    from scipp import min
    primary = primary_spectrometer(inst, source, sample, frequency, duration, delay, velocity)
    t_vs_inverse_v_polys_at_sample, s_layers = primary.project_transmitted_on_sample()
    # extract all times from the polygons
    times = Variable(values=hstack([p.vertices[:, 0] for p in t_vs_inverse_v_polys_at_sample]), dims=['time'], unit='s')
    return min(times)


def unwrap(times: Variable, frequency: Variable, shortest_time: Variable):
    from scipp import floor_divide, min, max
    unit = times.unit
    period = (1 / frequency).to(unit=unit, copy=False)
    shortest_time = shortest_time.to(unit=unit, copy=False)
    reference_time = floor_divide(shortest_time, period) * period
    contiguous = (times + reference_time - shortest_time) % period
    assert min(contiguous).value >= 0, "Negative time in unwrapped times"
    assert max(contiguous) <= period, "Time exceeds period in unwrapped times"
    return contiguous + shortest_time
