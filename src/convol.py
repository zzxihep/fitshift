# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.1
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _convol
else:
    import _convol

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


class SwigPyIterator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _convol.delete_SwigPyIterator

    def value(self):
        return _convol.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _convol.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _convol.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _convol.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _convol.SwigPyIterator_equal(self, x)

    def copy(self):
        return _convol.SwigPyIterator_copy(self)

    def next(self):
        return _convol.SwigPyIterator_next(self)

    def __next__(self):
        return _convol.SwigPyIterator___next__(self)

    def previous(self):
        return _convol.SwigPyIterator_previous(self)

    def advance(self, n):
        return _convol.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _convol.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _convol.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _convol.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _convol.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _convol.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _convol.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self

# Register SwigPyIterator in _convol:
_convol.SwigPyIterator_swigregister(SwigPyIterator)

class DoubleVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _convol.DoubleVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _convol.DoubleVector___nonzero__(self)

    def __bool__(self):
        return _convol.DoubleVector___bool__(self)

    def __len__(self):
        return _convol.DoubleVector___len__(self)

    def __getslice__(self, i, j):
        return _convol.DoubleVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _convol.DoubleVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _convol.DoubleVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _convol.DoubleVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _convol.DoubleVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _convol.DoubleVector___setitem__(self, *args)

    def pop(self):
        return _convol.DoubleVector_pop(self)

    def append(self, x):
        return _convol.DoubleVector_append(self, x)

    def empty(self):
        return _convol.DoubleVector_empty(self)

    def size(self):
        return _convol.DoubleVector_size(self)

    def swap(self, v):
        return _convol.DoubleVector_swap(self, v)

    def begin(self):
        return _convol.DoubleVector_begin(self)

    def end(self):
        return _convol.DoubleVector_end(self)

    def rbegin(self):
        return _convol.DoubleVector_rbegin(self)

    def rend(self):
        return _convol.DoubleVector_rend(self)

    def clear(self):
        return _convol.DoubleVector_clear(self)

    def get_allocator(self):
        return _convol.DoubleVector_get_allocator(self)

    def pop_back(self):
        return _convol.DoubleVector_pop_back(self)

    def erase(self, *args):
        return _convol.DoubleVector_erase(self, *args)

    def __init__(self, *args):
        _convol.DoubleVector_swiginit(self, _convol.new_DoubleVector(*args))

    def push_back(self, x):
        return _convol.DoubleVector_push_back(self, x)

    def front(self):
        return _convol.DoubleVector_front(self)

    def back(self):
        return _convol.DoubleVector_back(self)

    def assign(self, n, x):
        return _convol.DoubleVector_assign(self, n, x)

    def resize(self, *args):
        return _convol.DoubleVector_resize(self, *args)

    def insert(self, *args):
        return _convol.DoubleVector_insert(self, *args)

    def reserve(self, n):
        return _convol.DoubleVector_reserve(self, n)

    def capacity(self):
        return _convol.DoubleVector_capacity(self)
    __swig_destroy__ = _convol.delete_DoubleVector

# Register DoubleVector in _convol:
_convol.DoubleVector_swigregister(DoubleVector)


def poly(arrx, arrpar):
    return _convol.poly(arrx, arrpar)

def map_wave(wave, map_par):
    return _convol.map_wave(wave, map_par)

def gauss_filter(wave, flux, arrpar):
    return _convol.gauss_filter(wave, flux, arrpar)

def legendre_poly(arrx, arrpar):
    return _convol.legendre_poly(arrx, arrpar)


