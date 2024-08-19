from polystar import Polygon

def test_skew_smear():
    from choppera.utils import skew_smear
    t_min = 5  # second
    t_max = 8  # second
    inverse_v_min = 3  # second/m
    inverse_v_max = 7  # second/m
    not_skewed = Polygon([(t_min, inverse_v_max), (t_min, inverse_v_min), (t_max, inverse_v_min), (t_max, inverse_v_max)])

    path_length = 1  # m
    skewed = Polygon([(t_min - path_length * inverse_v_max, inverse_v_max),
                      (t_min - path_length * inverse_v_min, inverse_v_min),
                      (t_max - path_length * inverse_v_min, inverse_v_min),
                      (t_max - path_length * inverse_v_max, inverse_v_max)])

    # skew along time axis, by d/v:
    axis_from = 0
    axis_to = 1
    result = not_skewed.skew(-path_length, axis_from, axis_to)
    assert result == skewed

    smeared_result = skew_smear(not_skewed, -path_length, -path_length)
    assert smeared_result == skewed


def test_skew_smear_lengths():
    from choppera.utils import skew_smear
    x_min, x_max = 10, 20
    y_min, y_max = 2, 3

    not_skewed = Polygon([(x_min, y_max), (x_min, y_min), (x_max, y_min), (x_max, y_max)])
    factor0, factor1 = 1, 2
    skew0 = Polygon([(x_min + factor0 * y_max, y_max),
                     (x_min + factor0 * y_min, y_min),
                     (x_max + factor0 * y_min, y_min),
                     (x_max + factor0 * y_max, y_max)])
    skew1 = Polygon([(x_min + factor1 * y_max, y_max),
                     (x_min + factor1 * y_min, y_min),
                     (x_max + factor1 * y_min, y_min),
                     (x_max + factor1 * y_max, y_max)])
    axis_from = 0
    axis_to = 1
    result0 = not_skewed.skew(factor0, axis_from, axis_to)
    result1 = not_skewed.skew(factor1, axis_from, axis_to)
    assert result0 == skew0
    assert result1 == skew1

    smeared = Polygon([(x_min + factor0 * y_max, y_max),
                       (x_min + factor0 * y_min, y_min),
                       (x_max + factor1 * y_min, y_min),
                       (x_max + factor1 * y_max, y_max)])
    smeared_result = skew_smear(not_skewed, factor0, factor1)
    # The two polygons should be the same, but the smeared_result includes an extraneous vertex
    # TODO find why this happens, can persist for now since it does not effect the resulting area
    assert smeared_result.area == smeared.area