# SPDX-FileCopyrightText: 2024-present Gregory Tucker <gregory.tucker@ess.eu>
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import List, Tuple
from numpy import ndarray, array
from polystar import Polygon

from .flightpaths import FlightPath, AnalyzerArm


@dataclass
class SecondarySpectrometer:
    detectors: List[FlightPath]

    def inv_velocity_limits(self):
        all_limits = [d.inv_velocity_limits() for d in self.detectors]
        return min([min(al) for al in all_limits]), max([max(al) for al in all_limits])

    def sample_scatter(self, regions: List[Polygon]) -> List[Polygon]:
        iv_min, iv_max = self.inv_velocity_limits()
        scattered_regions = []
        for region in regions:
            t_min, t_max = region.min(), region.max()
            b = array([[t_min, iv_max], [t_min, iv_min], [t_max, iv_min], [t_max, iv_max]])
            scattered_regions.append(Polygon(b))
        return scattered_regions

    def project_on_detectors(self, sample_regions: List[Polygon]) -> List[List[Polygon]]:
        scattered = self.sample_scatter(sample_regions)
        projected = []
        for detector in self.detectors:
            short, long = detector.tinv_transforms()
            at_detector = []
            for index, rep in enumerate(scattered):
                skewed = rep.skew_smear(short, long)
                if index + 1 < len(scattered):
                    # we need to cut off this skewed region at the next region start time
                    y_min, y_max = skewed.min(1), skewed.max(1)
                    t = scattered[index + 1].min(0)
                    skewed = skewed.cut([t, y_min], [t, y_max])
                if not skewed.isempty:
                    overlap = detector.tinv_overlap(skewed)
                    at_detector.append(overlap)
            # at_detector = [detector.tinv_overlap(x.skew_smear(short, long)) for x in scattered]
            projected.append(at_detector)
        return projected

    def forward_time_distance_diagram(self, first: List[Polygon], zero=0.):
        from numpy import array

        def td_poly(low, up, a, b):
            verts = array([[low.min(), a], [low.max(), a], [up.max(), b], [up.min(), b]])
            return Polygon(verts)

        parts = []
        second_per_detector = self.project_on_detectors(first)
        for detector, second in zip(self.detectors, second_per_detector):
            d = detector.td_length()
            parts.append([td_poly(l, u, zero, zero + d) for l, u in zip(first, second)])

        return parts
