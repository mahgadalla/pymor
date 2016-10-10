# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from numbers import Number
from weakref import WeakKeyDictionary, WeakSet, ref

import numpy as np
from scipy.sparse import issparse

from pymor.core import NUMPY_INDEX_QUIRK
from pymor.vectorarrays.interfaces import VectorArrayInterface, VectorSpace, _INDEXTYPES


class NumpyVectorArray(VectorArrayInterface):
    """|VectorArray| implementation via |NumPy arrays|.

    This is the default |VectorArray| type used by all |Operators|
    in pyMOR's discretization toolkit. Moreover, all reduced |Operators|
    are based on |NumpyVectorArray|.

    Note that this class is just a thin wrapper around the underlying
    |NumPy array|. Thus, while operations like
    :meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.axpy` or
    :meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.dot`
    will be quite efficient, removing or appending vectors will
    be costly.
    """

    def __init__(self, array, copy=False, *, _copies=None):
        assert not isinstance(array, np.matrixlib.defmatrix.matrix)

        if type(array) is np.ndarray:
            if copy:
                array = array.copy()
        elif issparse(array):
            array = array.toarray()
        elif hasattr(array, 'data'):
            array = array.data
            if copy:
                array = array.copy()
        else:
            array = np.array(array, ndmin=2, copy=copy)

        if array.ndim != 2:
            assert array.ndim == 1
            array = np.reshape(array, (1, -1))

        self._array = array
        self._len = len(array)
        self._copies = _copies or [0, WeakKeyDictionary()]

    @classmethod
    def from_data(cls, data, subtype):
        return cls(data)

    @classmethod
    def from_file(cls, path, key=None, single_vector=False, transpose=False):
        assert not (single_vector and transpose)
        from pymor.tools.io import load_matrix
        array = load_matrix(path, key=key)
        assert isinstance(array, np.ndarray)
        assert array.ndim <= 2
        if array.ndim == 1:
            array = array.reshape((1, -1))
        if single_vector:
            assert array.shape[0] == 1 or array.shape[1] == 1
            array = array.reshape((1, -1))
        if transpose:
            array = array.T
        return cls(array)

    @classmethod
    def make_array(cls, subtype=None, count=0, reserve=0):
        assert isinstance(subtype, _INDEXTYPES)
        assert count >= 0
        assert reserve >= 0
        if reserve > count:
            real_array = np.zeros((reserve, subtype))
            U = cls(real_array[:count])
            U._real_array = real_array
            return U
        else:
            return cls(np.zeros((count, subtype)))

    @property
    def data(self):
        return self._array

    @property
    def real(self):
        return NumpyVectorArray(self._array.real, copy=True)

    @property
    def imag(self):
        return NumpyVectorArray(self._array.imag, copy=True)

    def __len__(self):
        return self._len

    @property
    def subtype(self):
        return self._array.shape[1]

    @property
    def dim(self):
        return self._array.shape[1]

    def _map_ind(self, ind):
        if ind is None:
            return slice(0, self._len)
        elif type(ind) is slice:
            return slice(*ind.indices(self._len))
        elif not hasattr(ind, '__len__'):
            ind = ind if 0 <= ind else self._len+ind
            return slice(ind, ind+1)
        else:
            return ind

    def copy(self, ind=None, deep=False):
        assert self.check_ind(ind)

        ind = self._map_ind(ind)

        if deep or hasattr(ind, '__len__'):
            new_array = self._array[ind]
            if not new_array.flags['OWNDATA']:
                new_array = new_array.copy()
            return NumpyVectorArray(new_array)

        new_array = self._array[ind]
        assert not new_array.flags['OWNDATA']
        C = NumpyVectorArray(new_array, _copies=self._copies)

        my_ind = getattr(self, '_ind', None)
        if my_ind is None:
            if hasattr(self, '_real_array'):
                ind = range(*ind.indices(self._len))
                C._ind = ind
                self._copies[1][C] = ind
            elif C._len == self._len:
                self._copies[0] += 1
            else:
                ind = range(*ind.indices(self._len))
                C._ind = ind
                self._copies[1][C] = ind
        else:
            ind = my_ind[ind]
            C._ind = ind
            self._copies[1][C] = ind

        return C

    __getitem__ = copy

    def append(self, other, remove_from_other=False):
        assert self.dim == other.dim
        assert other is not self or not remove_from_other

        len_other = other._len
        if len_other == 0:
            return

        if hasattr(self, '_ind'):
            del self._copies[1][self]
            self._copies = [0, WeakKeyDictionary()]
            self._array = np.vstack((self._array, other._array))
            self._len += len_other
            del self._ind
        elif hasattr(self, '_real_array'):
            if self._len + len_other <= len(self._real_array):
                if self._real_array.dtype != other._array.dtype:
                    self._transfer_ownership()
                    self._real_array = self._real_array.astype(np.promote_types(self._array.dtype, other._array.dtype))
                self._real_array[self._len:self._len + len_other] = other._array
                self._len += len_other
                self._array = self._real_array[:self._len]
            else:
                self._transfer_ownership()
                self._array = np.vstack((self._array, other._array))
                self._len += len_other
                del self._real_array
        else:
            self._transfer_ownership()
            self._array = np.vstack((self._array, other._array))
            self._len += len_other

        if remove_from_other:
            other.remove()

    def remove(self, ind=None):
        assert self.check_ind(ind)

        if hasattr(self, '_ind'):
            del self._ind
            del self._copies[1][self]
            self._copies = [0, WeakKeyDictionary()]
        else:
            self._transfer_ownership()
            if hasattr(self, '_real_array'):
                del self._real_array

        if ind is None:
            self._array = np.zeros((0, self.dim))
            self._len = 0
            return

        l = len(self)
        ind = (range(*ind.indices(l)) if type(ind) is slice else
               [ind if 0 <= ind else l+ind] if not hasattr(ind, '__len__') else
               [i if 0 <= i else l+i for i in ind])
        if len(ind) == 0:
            return

        remaining = sorted(set(range(l)) - set(ind))
        self._array = self._array[remaining]
        self._len = self._array.shape[0]
        assert self._array.flags['OWNDATA']

    __delitem__ = remove

    def scal(self, alpha, *, ind=None):
        assert self.check_ind_unique(ind)
        assert isinstance(alpha, _INDEXTYPES) \
            or isinstance(alpha, np.ndarray) and alpha.shape == (self.len_ind(ind),)

        ind = self._map_ind(ind)
        self._inplace_copy_if_needed(ind)

        if NUMPY_INDEX_QUIRK and self._len == 0:
            return

        if isinstance(alpha, np.ndarray):
            alpha = alpha[:, np.newaxis]

        alpha_type = type(alpha)
        alpha_dtype = alpha.dtype if alpha_type is np.ndarray else alpha_type
        if self._array.dtype != alpha_dtype:
            self._array = self._array.astype(np.promote_types(self._array.dtype, alpha_dtype))
        self._array[ind] *= alpha

    def axpy(self, alpha, x, *, ind=None):
        assert self.check_ind_unique(ind)
        assert self.dim == x.dim
        assert self.len_ind(ind) == len(x) or len(x) == 1
        assert isinstance(alpha, _INDEXTYPES) \
            or isinstance(alpha, np.ndarray) and alpha.shape == (self.len_ind(ind),)

        if self._len == 0:
            return

        if np.all(alpha == 0):
            return

        ind = self._map_ind(ind)
        self._inplace_copy_if_needed(ind)

        B = x._array

        alpha_type = type(alpha)
        alpha_dtype = alpha.dtype if alpha_type is np.ndarray else alpha_type
        if self._array.dtype != alpha_dtype or self._array.dtype != B.dtype:
            dtype = np.promote_types(self._array.dtype, alpha_dtype)
            dtype = np.promote_types(dtype, B.dtype)
            self._array = self._array.astype(dtype)

        if np.all(alpha == 1):
            self._array[ind] += B
        elif np.all(alpha == -1):
            self._array[ind] -= B
        else:
            if isinstance(alpha, np.ndarray):
                alpha = alpha[:, np.newaxis]
            self._array[ind] += B * alpha

    def dot(self, other):
        assert self.dim == other.dim

        A = self._array
        B = other._array

        if B.dtype in _complex_dtypes:
            return A.dot(B.conj().T)
        else:
            return A.dot(B.T)

    def pairwise_dot(self, other):
        assert self.dim == other.dim
        assert len(self) == len(other)

        A = self._array
        B = other._array

        if B.dtype in _complex_dtypes:
            return np.sum(A * B.conj(), axis=1)
        else:
            return np.sum(A * B, axis=1)

    def lincomb(self, coefficients):
        assert 1 <= coefficients.ndim <= 2
        if coefficients.ndim == 1:
            coefficients = coefficients[np.newaxis, ...]
        assert coefficients.shape[1] == len(self)

        return NumpyVectorArray(coefficients.dot(self._array), copy=False)

    def l1_norm(self):
        return np.linalg.norm(self._array, ord=1, axis=1)

    def l2_norm(self):
        return np.linalg.norm(self._array, axis=1)

    def l2_norm2(self):
        A = self._array
        return np.sum((A * A.conj()).real, axis=1)

    def components(self, component_indices):
        assert isinstance(component_indices, list) and (len(component_indices) == 0 or min(component_indices) >= 0) \
            or (isinstance(component_indices, np.ndarray) and component_indices.ndim == 1
                and (len(component_indices) == 0 or np.min(component_indices) >= 0))
        # NumPy 1.9 is quite permissive when indexing arrays of size 0, so we have to add the
        # following check:
        assert self._len > 0 \
            or (isinstance(component_indices, list)
                and (len(component_indices) == 0 or max(component_indices) < self.dim)) \
            or (isinstance(component_indices, np.ndarray) and component_indices.ndim == 1
                and (len(component_indices) == 0 or np.max(component_indices) < self.dim))

        if NUMPY_INDEX_QUIRK and (self._len == 0 or self.dim == 0):
            assert isinstance(component_indices, list) \
                and (len(component_indices) == 0 or max(component_indices) < self.dim) \
                or isinstance(component_indices, np.ndarray) \
                and component_indices.ndim == 1 \
                and (len(component_indices) == 0 or np.max(component_indices) < self.dim)
            return np.zeros((0, len(component_indices)))

        return self._array[:, component_indices]

    def amax(self):
        assert self.dim > 0

        if self._array.shape[1] == 0:
            l = len(self)
            return np.ones(l) * -1, np.zeros(l)

        A = np.abs(self._array)
        max_ind = np.argmax(A, axis=1)
        max_val = A[np.arange(len(A)), max_ind]
        return max_ind, max_val

    def __str__(self):
        return self._array.__str__()

    def __repr__(self):
        return 'NumpyVectorArray({})'.format(self._array.__str__())

    def __del__(self):
        if not hasattr(self, '_ind'):
            self._transfer_ownership(False)

    def _transfer_ownership(self, clear=True):
        full_copies = self._copies[0]
        if full_copies:
            self._copies[0] = full_copies - 1
        else:
            for U in self._copies[1].keys():
                U._array = U._array.copy()
                U._copies = [0, WeakKeyDictionary()]
                del U._ind
        if clear:
            self._copies = [0, WeakKeyDictionary()]

    def _inplace_copy_if_needed(self, ind):
        if hasattr(self, '_ind'):
            del self._ind
            del self._copies[1][self]
            self._copies = [0, WeakKeyDictionary()]
            self._array = self._array.copy()
            return

        if hasattr(ind, '__len__'):
            if len(ind) == 0:
                return
            start = min(ind)
            stop = max(ind) + 1
        else:
            start, stop, _ = ind.indices(self._len)
        if start == stop:
            return
        if self._copies[0]:
            self._transfer_ownership()
            self._array = self._array.copy()
        else:
            for U, o_ind in list(self._copies[1].items()):
                if start < o_ind.stop and stop > o_ind.start:
                    U._array = U._array.copy()
                    U._copies = [0, WeakKeyDictionary()]
                    del U._ind

    def __reduce__(self):
        return NumpyVectorArray, (self._array,)


def NumpyVectorSpace(dim):
    """Shorthand for |VectorSpace| `(NumpyVectorArray, dim)`."""
    return VectorSpace(NumpyVectorArray, dim)


_complex_dtypes = (np.complex64, np.complex128)
