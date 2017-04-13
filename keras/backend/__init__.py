from __future__ import absolute_import
from __future__ import print_function
import os
import json
import sys
from .common import epsilon
from .common import floatx
from .common import set_epsilon
from .common import set_floatx
from .common import cast_to_floatx
from .common import image_data_format
from .common import set_image_data_format
from .common import is_keras_tensor

'''
    通过读取
        - 配置文件 .keras.json
        - 环境变量 KERAS_BACKEND='tensorflow' 或 KERAS_BACKEND='theano'
    来决定使用哪种backend
    最后，使用
    ```python
        if _BACKEND == 'theano':
            sys.stderr.write('Using Theano backend.\n')
            from .theano_backend import *
        elif _BACKEND == 'tensorflow':
            sys.stderr.write('Using TensorFlow backend.\n')
            from .tensorflow_backend import *
        else:
            raise ValueError('Unknown backend: ' + str(_BACKEND))
    ```
    来引用backend中的函数和类。这两个文件分别位于backend包中的tensorflow_backend.py和theano_backend.py中
    
    这两个文件中的函数大致相同。
    
    这里有一个小技巧（知识点）:
        python的__init__.py函数有几个作用
        - 若文件夹中有此文件（内容可以为空），那么这个目录被定义为一个python package
        - 在 `__init__.py`中，可以定义一些package level的变量或者像本package中引入某些变量
        __init__.py被调用的时机：
        当执行代码 
        ```
            import pack
        ``` 或 
        ``` 
            from pack import module 
        ```
        或 
        ```
            from pack.module import object
        ```
        的时候，都会首先调用调用 `pack/__init__.py`，此时，可以向pack中注入变量（就像Keras在这个包中做的一样)，
        通过在`pack/__init__.py`中 
        ```
            from some.module import obj 
        ``` 
        可以把对象注入package层，从而被引用的变量可以通过使用
        ```
            import pack as P
            call(P.obj)
        ```
        来使用该对象，**注意**，该对象已经被引入到package层面了
'''
# Obtain Keras base dir path: either ~/.keras or /tmp.
_keras_base_dir = os.path.expanduser('~')
if not os.access(_keras_base_dir, os.W_OK):
    _keras_base_dir = '/tmp'
_keras_dir = os.path.join(_keras_base_dir, '.keras')

# Default backend: TensorFlow.
_BACKEND = 'tensorflow'

# Attempt to read Keras config file.
# 配置文件读取backend
_config_path = os.path.expanduser(os.path.join(_keras_dir, 'keras.json'))
if os.path.exists(_config_path):
    try:
        _config = json.load(open(_config_path))
    except ValueError:
        _config = {}
    _floatx = _config.get('floatx', floatx())
    assert _floatx in {'float16', 'float32', 'float64'}
    _epsilon = _config.get('epsilon', epsilon())
    assert isinstance(_epsilon, float)
    _backend = _config.get('backend', _BACKEND)
    assert _backend in {'theano', 'tensorflow'}
    _image_data_format = _config.get('image_data_format',
                                     image_data_format())
    assert _image_data_format in {'channels_last', 'channels_first'}

    set_floatx(_floatx)
    set_epsilon(_epsilon)
    set_image_data_format(_image_data_format)
    _BACKEND = _backend

# Save config file, if possible.
if os.access(_keras_base_dir, os.W_OK):
    if not os.path.exists(_keras_dir):
        try:
            os.makedirs(_keras_dir)
        except OSError:
            pass
    if not os.path.exists(_config_path):
        _config = {'floatx': floatx(),
                   'epsilon': epsilon(),
                   'backend': _BACKEND,
                   'image_data_format': image_data_format()}
        with open(_config_path, 'w') as f:
            f.write(json.dumps(_config, indent=4))

# Set backend based on KERAS_BACKEND flag, if applicable.
if 'KERAS_BACKEND' in os.environ:
    _backend = os.environ['KERAS_BACKEND']
    assert _backend in {'theano', 'tensorflow'}
    _BACKEND = _backend

# Import backend functions.
if _BACKEND == 'theano':
    # 这里打印出正在使用的Backend，不知道为什么要用sys.stderr.write，可能是比较明显吧
    sys.stderr.write('Using Theano backend.\n')
    from .theano_backend import *
elif _BACKEND == 'tensorflow':
    sys.stderr.write('Using TensorFlow backend.\n')
    from .tensorflow_backend import *
else:
    raise ValueError('Unknown backend: ' + str(_BACKEND))


def backend():
    """Publicly accessible method
    for determining the current backend.

    # Returns
        String, the name of the backend Keras is currently using.

    # Example
    ```python
        >>> keras.backend.backend()
        'tensorflow'
    ```
    """
    return _BACKEND
