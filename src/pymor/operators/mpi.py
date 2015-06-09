# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from pymor.operators.basic import OperatorBase
from pymor.operators.constructions import LincombOperator, VectorArrayOperator
from pymor.tools import mpi
from pymor.vectorarrays.interfaces import VectorSpace
from pymor.vectorarrays.mpi import MPIVectorArray


class MPIOperator(OperatorBase):

    def __init__(self, obj_id, functional=False, vector=False, with_apply2=False, array_type=MPIVectorArray):
        assert not (functional and vector)
        self.obj_id = obj_id
        self.op = op = mpi.get_object(obj_id)
        self.functional = functional
        self.vector = vector
        self.with_apply2 = with_apply2
        self.linear = op.linear
        self.name = op.name
        self.build_parameter_type(inherits=(op,))
        if vector:
            self.source = op.source
        else:
            subtypes = mpi.call(_MPIOperator_get_source_subtypes, obj_id)
            if all(subtype == subtypes[0] for subtype in subtypes):
                subtypes = (subtypes[0],)
            self.source = VectorSpace(array_type, (op.source.type, subtypes))
        if functional:
            self.range = op.range
        else:
            subtypes = mpi.call(_MPIOperator_get_range_subtypes, obj_id)
            if all(subtype == subtypes[0] for subtype in subtypes):
                subtypes = (subtypes[0],)
            self.range = VectorSpace(array_type, (op.range.type, subtypes))

    @property
    def invert_options(self):
        return self.op.invert_options

    def apply(self, U, ind=None, mu=None):
        assert U in self.source
        mu = self.parse_parameter(mu)
        U = U if self.vector else U.obj_id
        if self.functional:
            return mpi.call(mpi.method_call, self.obj_id, 'apply', U, ind=ind, mu=mu)
        else:
            space = self.range
            return space.type(space.subtype[0], space.subtype[1],
                              mpi.call(mpi.method_call_manage, self.obj_id, 'apply', U, ind=ind, mu=mu))

    def as_vector(self, mu=None):
        mu = self.parse_parameter(mu)
        if self.functional:
            space = self.source
            return space.type(space.subtype[0], space.subtype[1],
                              mpi.call(mpi.method_call_manage, self.obj_id, 'as_vector', mu=mu))
        else:
            raise NotImplementedError


    def apply2(self, V, U, U_ind=None, V_ind=None, mu=None, product=None):
        if not self.with_apply2:
            return super(MPIOperator, self).apply2(V, U, U_ind=U_ind, V_ind=V_ind, mu=mu, product=product)
        assert V in self.range
        assert U in self.source
        mu = self.parse_parameter(mu)
        U = U if self.vector else U.obj_id
        V = V if self.functional else V.obj_id
        product = product and product.obj_id
        return mpi.call(mpi.method_call, self.obj_id, 'apply2',
                        V, U, U_ind=U_ind, V_ind=V_ind, mu=mu, product=product)

    def pairwise_apply2(self, V, U, U_ind=None, V_ind=None, mu=None, product=None):
        assert V in self.range
        assert U in self.source
        mu = self.parse_parameter(mu)
        U = U if self.vector else U.obj_id
        V = V if self.functional else V.obj_id
        product = product and product.obj_id
        return mpi.call(mpi.method_call, self.obj_id, 'pairwise_apply2',
                        V, U, U_ind=U_ind, V_ind=V_ind, mu=mu, product=product)

    def apply_adjoint(self, U, ind=None, mu=None, source_product=None, range_product=None):
        assert U in self.range
        mu = self.parse_parameter(mu)
        U = U if self.functional else U.obj_id
        source_product = source_product and source_product.obj_id
        range_product = range_product and range_product.obj_id
        if self.vector:
            return mpi.call(mpi.method_call, self.obj_id, 'apply_adjoint',
                            U, ind=ind, mu=mu, source_product=source_product, range_product=range_product)
        else:
            space = self.source
            return space.type(space.subtype[0], space.subtype[1],
                              mpi.call(mpi.method_call_manage, self.obj_id, 'apply_adjoint',
                                       U, ind=ind, mu=mu, source_product=source_product, range_product=range_product))

    def apply_inverse(self, U, ind=None, mu=None, options=None):
        if self.vector or self.functional:
            raise NotImplementedError
        assert U in self.range
        mu = self.parse_parameter(mu)
        space = self.source
        return space.type(space.subtype[0], space.subtype[1],
                          mpi.call(mpi.method_call_manage, self.obj_id, 'apply_inverse',
                                   U.obj_id, ind=ind, mu=mu, options=options))

    def jacobian(self, U, mu=None):
        assert U in self.source
        mu = self.parse_parameter(mu)
        return type(self)(mpi.call(mpi.method_call_manage, self.obj_id, 'jacobian', U.obj_id, mu=mu),
                          functional=self.functional, vector=self.vector,
                          with_apply2=self.with_apply2, array_type=self.source.type)

    def assemble(self, mu=None):
        mu = self.parse_parameter(mu)
        return type(self)(mpi.call(mpi.method_call_manage, self.obj_id, 'assemble', mu=mu),
                          functional=self.functional, vector=self.vector,
                          with_apply2=self.with_apply2, array_type=self.source.type)

    def assemble_lincomb(self, operators, coefficients, name=None):
        if not all(isinstance(op, MPIOperator) for op in operators):
            return None
        operators = [op.obj_id for op in operators]
        obj_id = mpi.call(mpi.method_call_manage, self.obj_id, 'assemble_lincomb', operators, coefficients, name=name)
        op = mpi.get_object(obj_id)
        if op is None:
            mpi.call(mpi.remove_object, obj_id)
            return None
        else:
            return type(self)(obj_id, functional=self.functional, vector=self.vector,
                              with_apply2=self.with_apply2, array_type=self.source.type)

    def restricted(self, dofs):
        return mpi.call(mpi.method_call, self.obj_id, dofs)

    def __del__(self):
        mpi.call(mpi.remove_object, self.obj_id)


def _MPIOperator_get_source_subtypes(self):
    self = mpi.get_object(self)
    subtypes = mpi.comm.gather(self.source.subtype, root=0)
    if mpi.rank0:
        return tuple(subtypes)


def _MPIOperator_get_range_subtypes(self):
    self = mpi.get_object(self)
    subtypes = mpi.comm.gather(self.range.subtype, root=0)
    if mpi.rank0:
        return tuple(subtypes)


def mpi_wrap_operator(obj_id, functional=False, vector=False, with_apply2=False, array_type=MPIVectorArray):
    op = mpi.get_object(obj_id)
    if isinstance(op, LincombOperator):
        obj_ids = mpi.call(_mpi_wrap_operator_LincombOperator_manage_operators, obj_id)
        return LincombOperator([mpi_wrap_operator(o, functional, vector, with_apply2, array_type) for o in obj_ids],
                               op.coefficients, name=op.name)
    elif isinstance(op, VectorArrayOperator):
        array_obj_id, subtypes = mpi.call(_mpi_wrap_operator_VectorArrayOperator_manage_array, obj_id)
        if all(subtype == subtypes[0] for subtype in subtypes):
            subtypes = (subtypes[0],)
        return VectorArrayOperator(array_type(type(op._array), subtypes, array_obj_id),
                                   transposed=op.transposed, copy=False, name=op.name)
    else:
        return MPIOperator(obj_id, functional, vector, with_apply2, array_type)


def _mpi_wrap_operator_LincombOperator_manage_operators(obj_id):
    op = mpi.get_object(obj_id)
    obj_ids = [mpi.manage_object(o) for o in op.operators]
    mpi.remove_object(obj_id)
    if mpi.rank0:
        return obj_ids


def _mpi_wrap_operator_VectorArrayOperator_manage_array(obj_id):
    op = mpi.get_object(obj_id)
    array_obj_id = mpi.manage_object(op._array)
    subtypes = mpi.comm.gather(op._array.subtype, root=0)
    mpi.remove_object(obj_id)
    return array_obj_id, subtypes