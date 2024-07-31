from scipp import Variable, DataArray


class Phase:
    _data: DataArray

    def __init__(self, velocity: Variable, early: Variable, late: Variable):
        assert velocity.sizes == early.sizes == late.sizes
        mask = Variable(dims=velocity.dims, values=[False] * len(velocity))  # only works if velocity is 1-D?
        coords = {'velocity': velocity.to(unit='m/s'), 'early': early.to(unit='s'), 'late': late.to(unit='s')}
        self._data = DataArray(mask, coords=coords)

    @staticmethod
    def from_data(data: DataArray):
        from scipp import any
        assert 'velocity' in data.coords
        assert 'early' in data.coords
        assert 'late' in data.coords
        assert not any(data.data)
        return Phase(data.coords['velocity'], data.coords['early'], data.coords['late'])

    def copy(self):
        # remove masked data returning the resulting Phase object
        data = self._data[~self._data.data]  # this is already a copy ... but we're going to make some more
        return Phase.from_data(data)

    def min(self):
        from scipp import min as sc_min
        return sc_min(self._data.coords['early'])

    def max(self):
        from scipp import max as sc_max
        return sc_max(self._data.coords['late'])

    @property
    def velocity(self):
        return self._data.coords['velocity']

    @property
    def left(self):
        return self._data.coords['early']

    @property
    def right(self):
        return self._data.coords['late']

    def mask(self, mask: Variable):
        assert mask.sizes == self._data.sizes
        self._data.data |= mask

    def shift(self, which: str,  selector: Variable, target: Variable):
        assert which in self._data.coords
        assert selector.sizes == self._data.sizes
        assert selector.dtype == bool
        self._data.coords[which][selector] = target

    def __str__(self):
        return f"Phase[]"

    def __bool__(self):
        from scipp import any
        return not any(self._data.data)
