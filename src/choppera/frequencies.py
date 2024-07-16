# SPDX-FileCopyrightText: 2024-present Gregory Tucker <gregory.tucker@ess.eu>
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import List, Tuple, Dict
# from __future__ import annotations ## uncomment this if we need 3.7 or 3.8 support

@dataclass
class AllowedHarmonics:
    name: str

    @property
    def is_dependent(self):
        return False

    @property
    def depends_on(self):
        return ""

    def is_allowed(self, *args):
        pass

@dataclass
class IndependentHarmonics(AllowedHarmonics):
    allowed: List[int]

    def is_allowed(self, value: int, _: int = 0):
        return value in self.allowed

@dataclass
class DependentHarmonics(AllowedHarmonics):
    independent: IndependentHarmonics
    allowed: Dict[int, List[int]]

    @property
    def is_dependent(self):
        return True

    @property
    def depends_on(self):
        return self.independent.name

    def is_allowed(self, value: int, independent_value: int):
        assert self.independent.is_allowed(independent_value)
        assert independent_value in self.allowed
        return value in self.allowed[independent_value]


@dataclass
class Frequencies:
    base: float
    named: List[AllowedHarmonics]
    harmonics: List[int]

    @property
    def names(self):
        return [x.name for x in self.named]

    def __has_name__(self, item):
        assert item in self.names, f"{item} is not in items {self.names}"

    def __name_at__(self, item):
        return [i for i, x in enumerate(self.named) if item == x.name][0]

    def __getitem__(self, item):
        self.__has_name__(item)
        return self.harmonics[self.__name_at__(item)] * self.base

    def __setitem__(self, item, value):
        self.__has_name__(item)
        idx = self.__name_at__(item)
        allowed = self.named[idx]
        val2 = self.harmonics[self.__name_at__(allowed.depends_on)] if allowed.is_dependent else 0
        assert allowed.is_allowed(value, val2), f"the harmonic {value} is not allowed for {item}"
        self.harmonics[idx] = value

    def absolute(self):
        return {n.name: v * self.base for n, v in zip(self.named, self.harmonics)}
