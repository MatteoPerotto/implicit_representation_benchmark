import __main__
import json
from pathlib import Path


class BaseConfig:

    @classmethod
    def init(cls):
        with Path('configs/__init__.py').open('w+') as f:
            f.writelines([f'from .{Path(str(__main__)).stem } import Config'])
            f.flush()

    @classmethod
    def to_json(cls):
        return json.dumps(cls.to_dict())

    @classmethod
    def to_dict(cls):
        target = cls

        res = {}
        for k in dir(target):
            if not k.startswith('_') and k not in ['to_dict', 'to_json', 'to_list', 'init']:
                attr = getattr(target, k)
                # If it's a class inside config, get inside it,
                # else just log module and name in the dict as a string
                if type(attr) == type:
                    # if we are executing the config the module is __main__. If we are importing it is config
                    if attr.__module__.split('.')[0] in ['configs', '__main__']:
                        res[k] = attr.to_dict()
                    else:
                        res[k] = f'{attr.__module__}.{attr.__name__}'
                # If it's not a class save it. This is done for basic types.
                # Could cause problems with complex objects
                else:
                    res[k] = attr
        return res

    @classmethod
    def to_list(cls):
        target = cls

        res = []
        for k in dir(target):
            if not k.startswith('_') and k not in ['to_dict', 'to_json', 'to_list', 'init']:
                attr = getattr(target, k)
                # If it's a class inside config, get inside it,
                # else just log module and name in the dict as a string
                if type(attr) == type:
                    if attr.__module__.split('.')[0] in ['configs', '__main__']:
                        res.append(attr.to_list())
                    else:
                        res.append(f'{attr.__module__}.{attr.__name__}')
                # If it's not a class save it. This is done for basic types.
                # Could cause problems with complex objects
                else:
                    res.append(attr)
        return res

    def __getattribute__(self, item):
        return object.__getattribute__(self, item)
