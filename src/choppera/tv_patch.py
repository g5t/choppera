# SPDX-FileCopyrightText: 2024-present Gregory Tucker <gregory.tucker@ess.eu>
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from numpy import ndarray


@dataclass
class TVPatch:
    coords: ndarray

    def __init__(self, coords: ndarray):
        from numpy import all
        assert coords.shape[0] > 2
        assert coords.shape[1] == 2
        assert all(coords >= 0.)
        self.coords = coords
    #
    # def polygon(self, **kwargs):
    #     from matplotlib.patches import Polygon
    #     return Polygon(self.coords, **kwargs)
