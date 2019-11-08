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
    from . import _rebin
else:
    import _rebin

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
    __swig_destroy__ = _rebin.delete_SwigPyIterator

    def value(self):
        return _rebin.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _rebin.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _rebin.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _rebin.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _rebin.SwigPyIterator_equal(self, x)

    def copy(self):
        return _rebin.SwigPyIterator_copy(self)

    def next(self):
        return _rebin.SwigPyIterator_next(self)

    def __next__(self):
        return _rebin.SwigPyIterator___next__(self)

    def previous(self):
        return _rebin.SwigPyIterator_previous(self)

    def advance(self, n):
        return _rebin.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _rebin.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _rebin.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _rebin.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _rebin.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _rebin.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _rebin.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self

# Register SwigPyIterator in _rebin:
_rebin.SwigPyIterator_swigregister(SwigPyIterator)

class DoubleVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _rebin.DoubleVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _rebin.DoubleVector___nonzero__(self)

    def __bool__(self):
        return _rebin.DoubleVector___bool__(self)

    def __len__(self):
        return _rebin.DoubleVector___len__(self)

    def __getslice__(self, i, j):
        return _rebin.DoubleVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _rebin.DoubleVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _rebin.DoubleVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _rebin.DoubleVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _rebin.DoubleVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _rebin.DoubleVector___setitem__(self, *args)

    def pop(self):
        return _rebin.DoubleVector_pop(self)

    def append(self, x):
        return _rebin.DoubleVector_append(self, x)

    def empty(self):
        return _rebin.DoubleVector_empty(self)

    def size(self):
        return _rebin.DoubleVector_size(self)

    def swap(self, v):
        return _rebin.DoubleVector_swap(self, v)

    def begin(self):
        return _rebin.DoubleVector_begin(self)

    def end(self):
        return _rebin.DoubleVector_end(self)

    def rbegin(self):
        return _rebin.DoubleVector_rbegin(self)

    def rend(self):
        return _rebin.DoubleVector_rend(self)

    def clear(self):
        return _rebin.DoubleVector_clear(self)

    def get_allocator(self):
        return _rebin.DoubleVector_get_allocator(self)

    def pop_back(self):
        return _rebin.DoubleVector_pop_back(self)

    def erase(self, *args):
        return _rebin.DoubleVector_erase(self, *args)

    def __init__(self, *args):
        _rebin.DoubleVector_swiginit(self, _rebin.new_DoubleVector(*args))

    def push_back(self, x):
        return _rebin.DoubleVector_push_back(self, x)

    def front(self):
        return _rebin.DoubleVector_front(self)

    def back(self):
        return _rebin.DoubleVector_back(self)

    def assign(self, n, x):
        return _rebin.DoubleVector_assign(self, n, x)

    def resize(self, *args):
        return _rebin.DoubleVector_resize(self, *args)

    def insert(self, *args):
        return _rebin.DoubleVector_insert(self, *args)

    def reserve(self, n):
        return _rebin.DoubleVector_reserve(self, n)

    def capacity(self):
        return _rebin.DoubleVector_capacity(self)
    __swig_destroy__ = _rebin.delete_DoubleVector

# Register DoubleVector in _rebin:
_rebin.DoubleVector_swigregister(DoubleVector)


def rebin(wave, flux, new_wave):
    return _rebin.rebin(wave, flux, new_wave)

def rebin_err(wave, err, new_wave):
    return _rebin.rebin_err(wave, err, new_wave)


