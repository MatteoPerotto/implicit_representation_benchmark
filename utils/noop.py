# import builtins
#
# _real_import = builtins.__import__
#
#
# def my_import(name, globals=None, locals=None, fromlist=(), level=0):
#     if name == 'noop':
#         name = 'utils.noop'
#         fromlist = tuple('NoOp' for _ in fromlist)
#     print(fromlist, level)
#     res = _real_import(name, globals, locals, fromlist, level)
#     return res
#
#
# builtins.__import__ = my_import
#
import abc


def noop(*args, **kwargs):
    return NoOp


class NoOpMeta(abc.ABCMeta):

    def __getattribute__(cls, item):
        try:
            return abc.ABCMeta.__getattribute__(cls, item)
        except AttributeError:
            return NoOp

    def __call__(cls, *args, **kwargs):
        return NoOp

    def __bool__(self):
        return False

    def __str__(self):
        return 'NoOp'


class NoOp(metaclass=NoOpMeta):

    def __getattribute__(self, item):
        return object.__getattribute__(self, item)


