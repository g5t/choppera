from numpy import ndarray
from polystar import Polygon


def interpolate(positions: ndarray[float], values: ndarray[float], at: float) -> float:
    from numpy import searchsorted
    second = searchsorted(positions, at)
    t = (at - positions[second - 1]) / (positions[second] - positions[second - 1])
    return values[second - 1] + t * (values[second] - values[second - 1])


def skew_smear(poly: Polygon, factor0, factor1) -> Polygon:
    from numpy import vstack
    from scipp import Variable
    if isinstance(factor0, Variable):
        factor0 = factor0.value
    if isinstance(factor1, Variable):
        factor1 = factor1.value
    source = 1
    sink = 0
    p0 = poly.skew(factor0, source, sink)
    p1 = p0.skew(factor1, source, sink)
    # Concatenate all skewed vertices, then find the convex hull polygon containing them all
    return Polygon(vstack((p0.vertices, p1.vertices)))
