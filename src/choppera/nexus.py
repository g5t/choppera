from scippnexus import Group


def entry_exit_size(nxguide: Group):
    # Find the extent of the guide at its entry and exit planes, using the dot product to find the two extremes
    from scipp import dot, vector, min, max
    vertices = nxguide['OFF_GEOMETRY/vertices'][...]
    x = vector(value=[1., 0, 0], unit='1')
    y = vector(value=[0, 1., 0], unit='1')
    z = vector(value=[0, 0, 1.], unit='1')

    v_z = dot(vertices, z)
    z_min, z_max = min(v_z), max(v_z)
    entry = vertices[v_z == z_min]
    exits = vertices[v_z == z_max]
    en_x = dot(entry, x)
    en_y = dot(entry, y)
    ex_x = dot(exits, x)
    ex_y = dot(exits, y)

    en_w = max(en_x) - min(en_x)
    en_h = max(en_y) - min(en_y)
    ex_w = max(ex_x) - min(ex_x)
    ex_h = max(ex_y) - min(ex_y)

    return (en_w, en_h), (ex_w, ex_h)


def guess_group_geometry(group: Group):
    guesses = 'OFF_GEOMETRY', 'OFF', 'GEOMETRY'
    children = 'vertices', 'faces', 'winding_order'
    for x in guesses + tuple(y.lower() for y in guesses):
        if x in group and all(y in group[x] for y in children):
            return x
    raise RuntimeError(f'Specify which of {list(group)} contains OFF geometry fields {children}')


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

