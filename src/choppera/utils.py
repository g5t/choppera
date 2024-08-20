from numpy import ndarray
from polystar import Polygon


def interpolate(positions: ndarray[float], values: ndarray[float], at: float) -> float:
    from numpy import searchsorted
    second = searchsorted(positions, at)
    t = (at - positions[second - 1]) / (positions[second] - positions[second - 1])
    return values[second - 1] + t * (values[second] - values[second - 1])


def skew_smear(poly: Polygon, factor0, factor1) -> Polygon:
    from numpy import vstack, einsum
    from scipp import Variable
    if isinstance(factor0, Variable):
        factor0 = factor0.value
    if isinstance(factor1, Variable):
        factor1 = factor1.value

    # TODO investigate why poly.skew nor poly.transform works in some cases -- object lifetime issues?
    # The problem call path is primary::project_all_on_source -> utils::skew_smear -> poly.skew
    # which uses a list comprehension. Switching the list comprehension for a for loop does not fix the issue.
    t0 = [[1., factor0], [0., 1.]]
    t1 = [[1., factor1], [0., 1.]]
    # v0 = poly.transform(t0).vertices
    # v1 = poly.transform(t1).vertices

    # the polygon may have extraneous vertices, or internal wires that we are ignoring
    # so we need to extract only the border vertices before skewing
    v0 = einsum('ij,kj->ki', t0, poly.vertices[poly.border, :])
    v1 = einsum('ij,kj->ki', t1, poly.vertices[poly.border, :])


    # Concatenate all skewed vertices, then find the convex hull polygon containing them all
    return Polygon(vstack((v0, v1)))
