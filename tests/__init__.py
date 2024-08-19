# SPDX-FileCopyrightText: 2024-present Gregory Tucker <gregory.tucker@ess.eu>
#
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any
import pytest

pytest.register_assert_rewrite('scipp.testing.assertions')


def pytest_assertrepr_compare(op: str, left: Any, right: Any) -> list[str]:
    from scipp import Unit, DType
    if isinstance(left, Unit) and isinstance(right, Unit):
        return [f'Unit({left}) {op} Unit({right})']
    if isinstance(left, DType) or isinstance(right, DType):
        return [f'{left!r} {op} {right!r}']
