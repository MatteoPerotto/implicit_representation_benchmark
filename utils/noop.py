import abc


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


