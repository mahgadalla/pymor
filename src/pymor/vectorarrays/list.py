# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from numbers import Number

import numpy as np

from pymor.core.interfaces import BasicInterface, abstractmethod, abstractclassmethod, abstractproperty
from pymor.vectorarrays.interfaces import VectorArrayInterface, _INDEXTYPES


class VectorInterface(BasicInterface):
    """Interface for vectors.

    This Interface is intended to be used in conjunction with |ListVectorArray|.
    All pyMOR algorithms operate on |VectorArrays| instead of single vectors!
    All methods of the interface have a direct counterpart in the |VectorArray|
    interface.
    """

    @abstractclassmethod
    def make_zeros(cls, subtype=None):
        pass

    @classmethod
    def from_data(cls, data, subtype):
        raise NotImplementedError

    @abstractproperty
    def dim(self):
        pass

    @property
    def subtype(self):
        return None

    @abstractmethod
    def copy(self, deep=False):
        pass

    @abstractmethod
    def scal(self, alpha):
        pass

    @abstractmethod
    def axpy(self, alpha, x):
        pass

    @abstractmethod
    def dot(self, other):
        pass

    @abstractmethod
    def l1_norm(self):
        pass

    @abstractmethod
    def l2_norm(self):
        pass

    @abstractmethod
    def l2_norm2(self):
        pass

    def sup_norm(self):
        if self.dim == 0:
            return 0.
        else:
            _, max_val = self.amax()
            return max_val

    @abstractmethod
    def components(self, component_indices):
        pass

    @abstractmethod
    def amax(self):
        pass

    def __add__(self, other):
        result = self.copy()
        result.axpy(1, other)
        return result

    def __iadd__(self, other):
        self.axpy(1, other)
        return self

    __radd__ = __add__

    def __sub__(self, other):
        result = self.copy()
        result.axpy(-1, other)
        return result

    def __isub__(self, other):
        self.axpy(-1, other)
        return self

    def __mul__(self, other):
        result = self.copy()
        result.scal(other)
        return result

    def __imul__(self, other):
        self.scal(other)
        return self

    def __neg__(self):
        result = self.copy()
        result.scal(-1)
        return result


class CopyOnWriteVector(VectorInterface):

    @abstractclassmethod
    def from_instance(cls, instance):
        pass

    @abstractmethod
    def _copy_data(self):
        pass

    @abstractmethod
    def _scal(self, alpha):
        pass

    @abstractmethod
    def _axpy(self, alpha, x):
        pass

    def copy(self, deep=False):
        c = self.from_instance(self)
        if deep:
            c._copy_data()
        else:
            try:
                self._refcount[0] += 1
            except AttributeError:
                self._refcount = [2]
            c._refcount = self._refcount
        return c

    def scal(self, alpha):
        self._copy_data_if_needed()
        self._scal(alpha)

    def axpy(self, alpha, x):
        self._copy_data_if_needed()
        self._axpy(alpha, x)

    def __del__(self):
        try:
            self._refcount[0] -= 1
        except AttributeError:
            pass

    def _copy_data_if_needed(self):
        try:
            if self._refcount[0] > 1:
                self._refcount[0] -= 1
                self._copy_data()
                self._refcount = [1]
        except AttributeError:
            self._refcount = [1]


class NumpyVector(CopyOnWriteVector):
    """Vector stored in a NumPy 1D-array."""

    def __init__(self, instance, dtype=None, copy=False, order=None, subok=False):
        if isinstance(instance, np.ndarray) and not copy:
            self._array = instance
        else:
            self._array = np.array(instance, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=1)
        assert self._array.ndim == 1

    @classmethod
    def from_instance(cls, instance):
        return cls(instance._array)

    @classmethod
    def make_zeros(cls, subtype=None):
        assert isinstance(subtype, Number)
        return NumpyVector(np.zeros(subtype))

    @classmethod
    def from_data(cls, data, subtype):
        assert isinstance(data, np.ndarray) and data.ndim == 1
        assert len(data) == subtype
        return cls(data)

    @property
    def data(self):
        return self._array

    @property
    def dim(self):
        return len(self._array)

    @property
    def subtype(self):
        return len(self._array)

    def _copy_data(self):
        self._array = self._array.copy()

    def _scal(self, alpha):
        self._array *= alpha

    def _axpy(self, alpha, x):
        assert self.dim == x.dim
        if alpha == 0:
            return
        if alpha == 1:
            self._array += x._array
        elif alpha == -1:
            self._array -= x._array
        else:
            self._array += x._array * alpha

    def dot(self, other):
        assert self.dim == other.dim
        return np.sum(self._array * other._array)

    def l1_norm(self):
        return np.sum(np.abs(self._array))

    def l2_norm(self):
        return np.linalg.norm(self._array)

    def l2_norm2(self):
        return np.sum((self._array * self._array.conj()).real)

    def components(self, component_indices):
        return self._array[component_indices]

    def amax(self):
        A = np.abs(self._array)
        max_ind = np.argmax(A)
        max_val = A[max_ind]
        return max_ind, max_val


class ListVectorArray(VectorArrayInterface):
    """|VectorArray| implementation via a Python list of vectors.

    The :attr:`subtypes <pymor.vectorarrays.interfaces.VectorArrayInterface.subtype>`
    a |ListVectorArray| can have are tuples `(vector_type, vector_subtype)`
    where `vector_type` is a subclass of :class:`VectorInterface` and
    `vector_subtype` is a valid subtype for `vector_type`.

    Parameters
    ----------
    vectors
        List of :class:`vectors <VectorInterface>` contained in
        the array.
    subtype
        If `vectors` is empty, the array's
        :attr:`~pymor.vectorarrays.interfaces.VectorArrayInterface.subtype`
        must be provided, as the subtype cannot be derived from `vectors`
        in this case.
    copy
        If `True`, make copies of the vectors in `vectors`.
    """

    _NONE = ()

    def __init__(self, vectors, subtype=_NONE, copy=True):
        vectors = list(vectors)
        if subtype is self._NONE:
            assert len(vectors) > 0
            subtype = (type(vectors[0]), vectors[0].subtype)
        if not copy:
            self._list = vectors
        else:
            self._list = [v.copy() for v in vectors]
        self.vector_type, self.vector_subtype = vector_type, vector_subtype = subtype
        assert all(v.subtype == vector_subtype for v in self._list)

    @classmethod
    def make_array(cls, subtype=None, count=0, reserve=0):
        assert count >= 0
        assert reserve >= 0
        vector_type, vector_subtype = subtype
        return cls([vector_type.make_zeros(vector_subtype) for _ in range(count)], subtype=subtype, copy=False)

    @classmethod
    def from_data(cls, data, subtype):
        vector_type, vector_subtype = subtype
        return cls([vector_type.from_data(v, vector_subtype) for v in data], subtype=subtype)

    def __len__(self):
        return len(self._list)

    @property
    def data(self):
        if not hasattr(self.space.type, 'data'):
            raise TypeError('{} does not have a data attribute'.format(self.space.type))
        if len(self._list) > 0:
            return np.array([v.data for v in self._list])
        else:
            return np.empty((0, self.dim))

    @property
    def dim(self):
        if len(self._list) > 0:
            return self._list[0].dim
        else:
            return self.vector_type.make_zeros(self.vector_subtype).dim

    @property
    def subtype(self):
        return (self.vector_type, self.vector_subtype)

    def copy(self, ind=None, deep=False):
        assert self.check_ind(ind)

        if ind is None:
            vecs = [v.copy(deep=deep) for v in self._list]
        elif type(ind) is slice:
            vecs = [self._list[i].copy(deep=deep) for i in range(*ind.indices(len(self._list)))]
        elif not hasattr(ind, '__len__'):
            vecs = [self._list[ind].copy(deep=deep)]
        else:
            vecs = [self._list[i].copy(deep=deep) for i in ind]

        return type(self)(vecs, subtype=self.subtype, copy=False)

    __getitem__ = copy

    def append(self, other, remove_from_other=False):
        assert other.space == self.space
        assert other is not self or not remove_from_other

        other_list = other._list
        if not remove_from_other:
            self._list.extend([v.copy() for v in other_list])
        else:
            self._list.extend(other_list)
            other._list = []

    def remove(self, ind=None):
        assert self.check_ind(ind)
        if ind is None:
            self._list = []
        elif type(ind) is slice or not hasattr(ind, '__len__'):
            del self._list[ind]
        else:
            thelist = self._list
            l = len(thelist)
            remaining = sorted(set(range(l)) - set(i if 0 <= i else l+i for i in ind))
            self._list = [thelist[i] for i in remaining]

    __delitem__ = remove

    def scal(self, alpha, *, ind=None):
        assert self.check_ind_unique(ind)
        assert isinstance(alpha, Number) \
            or isinstance(alpha, np.ndarray) and alpha.shape == (self.len_ind(ind),)

        if ind is None:
            if isinstance(alpha, np.ndarray):
                for a, v in zip(alpha, self._list):
                    v.scal(a)
            else:
                for v in self._list:
                    v.scal(alpha)
        elif isinstance(ind, _INDEXTYPES):
            if isinstance(alpha, np.ndarray):
                alpha = alpha[0]
            self._list[ind].scal(alpha)
        else:
            l = self._list
            if type(ind) is slice:
                ind = range(*ind.indices(len(l)))
            if isinstance(alpha, np.ndarray):
                for a, i in zip(alpha, ind):
                    l[i].scal(a)
            else:
                for i in ind:
                    l[i].scal(alpha)

    def axpy(self, alpha, x, ind=None):
        assert self.check_ind_unique(ind)
        assert self.space == x.space
        len_x = len(x)
        assert self.len_ind(ind) == len_x or len_x == 1
        assert isinstance(alpha, _INDEXTYPES) \
            or isinstance(alpha, np.ndarray) and alpha.shape == (self.len_ind(ind),)

        if self is x:
            if ind is None:
                self.scal(1 + alpha)
                return
            else:
                self.axpy(alpha, x.copy(), ind=ind)  # improve this?
                return

        if ind is None:
            Y = iter(self._list)
            len_Y = len(self._list)
        elif isinstance(ind, _INDEXTYPES):
            Y = iter([self._list[ind]])
            len_Y = 1
        else:
            if type(ind) is slice:
                ind = range(*ind.indices(len(self._list)))
            Y = (self._list[i] for i in ind)
            len_Y = len(ind)

        X = iter(x._list)

        if np.all(alpha == 0):
            return
        elif len_x == 1:
            xx = next(X)
            if isinstance(alpha, np.ndarray):
                for a, y in zip(alpha, Y):
                    y.axpy(a, xx)
            else:
                for y in Y:
                    y.axpy(alpha, xx)
        else:
            assert len_x == len_Y
            if isinstance(alpha, np.ndarray):
                for a, xx, y in zip(alpha, X, Y):
                    y.axpy(a, xx)
            else:
                for xx, y in zip(X, Y):
                    y.axpy(alpha, xx)

    def dot(self, other):
        assert self.space == other.space
        R = np.empty((len(self._list), len(other)))
        for i, a in enumerate(self._list):
            for j, b in enumerate(other._list):
                R[i, j] = a.dot(b)
        return R

    def pairwise_dot(self, other):
        assert self.space == other.space
        assert len(self._list) == len(other)
        return np.array([a.dot(b) for a, b in zip(self._list, other._list)])

    def gramian(self):
        l = len(self._list)
        R = np.empty((l, l))
        for i in range(l):
            for j in range(i, l):
                R[i, j] = self._list[i].dot(self._list[j])
                R[j, i] = R[i, j]
        return R

    def lincomb(self, coefficients):
        assert 1 <= coefficients.ndim <= 2
        if coefficients.ndim == 1:
            coefficients = coefficients[np.newaxis, :]
        assert coefficients.shape[1] == len(self._list)

        RL = []
        for coeffs in coefficients:
            R = self.vector_type.make_zeros(self.vector_subtype)
            for v, c in zip(self._list, coeffs):
                R.axpy(c, v)
            RL.append(R)

        return type(self)(RL, subtype=self.subtype, copy=False)

    def l1_norm(self):
        return np.array([v.l1_norm() for v in self._list])

    def l2_norm(self):
        return np.array([v.l2_norm() for v in self._list])

    def l2_norm2(self):
        return np.array([v.l2_norm2() for v in self._list])

    def sup_norm(self):
        return np.array([v.sup_norm() for v in self._list])

    def components(self, component_indices):
        assert isinstance(component_indices, list) and (len(component_indices) == 0 or min(component_indices) >= 0) \
            or (isinstance(component_indices, np.ndarray) and component_indices.ndim == 1
                and (len(component_indices) == 0 or np.min(component_indices) >= 0))

        if len(self._list) == 0:
            assert len(component_indices) == 0 \
                or isinstance(component_indices, list) and max(component_indices) < self.dim \
                or isinstance(component_indices, np.ndarray) and np.max(component_indices) < self.dim
            return np.empty((0, len(component_indices)))

        R = np.empty((len(self._list), len(component_indices)))
        for k, v in enumerate(self._list):
            R[k] = v.components(component_indices)

        return R

    def amax(self):
        assert self.dim > 0

        MI = np.empty(len(self._list), dtype=np.int)
        MV = np.empty(len(self._list))

        for k, v in enumerate(self._list):
            MI[k], MV[k] = v.amax()

        return MI, MV

    def __str__(self):
        return 'ListVectorArray of {} {}s of dimension {}'.format(len(self._list), str(self.vector_type), self.dim)
