# The copy module in Python provides functions for creating copies of objects. There are two main types of copies you can make:
# Shallow Copy: This creates a new object with the same references to the original object's elements. If the original object contains mutable elements (like lists or dictionaries), changes to the copy will also affect the original.
# Deep Copy: This creates a new object with entirely new copies of the original object's elements. Changes to the copy won't affect the original and vice versa.
# The copy module provides the following functions:

# copy.copy(x): Creates a shallow copy of an object x.
# copy.deepcopy(x): Creates a deep copy of an object x.

import copy
import os

# The inspect module in Python allows you to examine the live objects within your code at runtime.
# This means you can get information about functions, classes, modules, tracebacks, and more. Here are some of the key functionalities provided by inspect:
# Type Checking:
# inspect.isclass(object): Checks if an object is a class.
# inspect.isfunction(object): Checks if an object is a function.
# inspect.ismodule(object): Checks if an object is a module.

# Inspecting Classes and Functions:
# inspect.getargspec(function): Returns a tuple containing argument information for a function (deprecated in Python 3, use inspect.signature instead).
# inspect.signature(function): Provides a more detailed analysis of function arguments in Python 3.
# inspect.getmembers(object): Returns a list of members (e.g., methods, attributes) of an object.
import inspect

# Log an informational message
# logger.info("This is an informational message.")
import logging
import torch
import numpy as np
import torch.nn as nn
import torch.autograd as autograd
import yaml
import random
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict, namedtuple

# NoOp serves as a placeholder for functionality that does nothing.
class NoOp:
    def __getattr__(self, *args):
        def no_op(*args, **kwargs): pass
        return no_op

# the Odict.append method provides a convenient way to merge contents from another OrderedDict while ensuring that the order of elements is preserved within the current Odict.
# It also handles cases where values might not be lists initially and converts them for consistent merging behavior.
class Odict(OrderedDict):
    def append(self, odict):
        dst_keys = self.keys()
        for k, v in odict.items():
            if not is_list(v):
                v = [v]
            if k in dst_keys:
                if is_list(self[k]):
                    self[k] += v
                else:
                    self[k] = [self[k]] + v
            else:
                self[k] = v

# the Ntuple function provides a more concise and user-friendly way to create namedtuples in Python, especially when dealing with single key/value scenarios.
def Ntuple(description, keys, values):
    if not is_list_or_tuple(keys):
        keys = [keys]
        values = [values]
    Tuple = namedtuple(description, keys)
    return Tuple._make(values)

# this function helps ensure that you're using the correct arguments for your functions and classes by filtering out any unexpected configurations.
# It also provides optional logging for unexpected arguments.
def get_valid_args(obj, input_args, free_keys=[]):
    if inspect.isfunction(obj):
        expected_keys = inspect.getfullargspec(obj)[0]
    elif inspect.isclass(obj):
        expected_keys = inspect.getfullargspec(obj.__init__)[0]
    else:
        raise ValueError('Just support function and class object!')
    unexpect_keys = list()
    expected_args = {}
    for k, v in input_args.items():
        if k in expected_keys:
            expected_args[k] = v
        elif k in free_keys:
            pass
        else:
            unexpect_keys.append(k)
    if unexpect_keys != []:
        logging.info("Find Unexpected Args(%s) in the Configuration of - %s -" %
                     (', '.join(unexpect_keys), obj.__name__))
    return expected_args


def get_attr_from(sources, name):
    try:
        return getattr(sources[0], name)
    except:
        return get_attr_from(sources[1:], name) if len(sources) > 1 else getattr(sources[0], name)


def is_list_or_tuple(x):
    return isinstance(x, (list, tuple))


def is_bool(x):
    return isinstance(x, bool)


def is_str(x):
    return isinstance(x, str)


def is_list(x):
    return isinstance(x, list) or isinstance(x, nn.ModuleList)


def is_dict(x):
    return isinstance(x, dict) or isinstance(x, OrderedDict) or isinstance(x, Odict)


def is_tensor(x):
    return isinstance(x, torch.Tensor)


def is_array(x):
    return isinstance(x, np.ndarray)


def ts2np(x):
    return x.cpu().data.numpy()


def ts2var(x, **kwargs):
    return autograd.Variable(x, **kwargs).cuda()


def np2var(x, **kwargs):
    return ts2var(torch.from_numpy(x), **kwargs)


def list2var(x, **kwargs):
    return np2var(np.array(x), **kwargs)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def MergeCfgsDict(src, dst):
    for k, v in src.items():
        if (k not in dst.keys()) or (type(v) != type(dict())):
            dst[k] = v
        else:
            if is_dict(src[k]) and is_dict(dst[k]):
                MergeCfgsDict(src[k], dst[k])
            else:
                dst[k] = v


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def config_loader(path):
    with open(path, 'r') as stream:
        src_cfgs = yaml.safe_load(stream)
    with open("./configs/default.yaml", 'r') as stream:
        dst_cfgs = yaml.safe_load(stream)
    MergeCfgsDict(src_cfgs, dst_cfgs)
    return dst_cfgs


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def handler(signum, frame):
    logging.info('Ctrl+c/z pressed')
    os.system(
        "kill $(ps aux | grep main.py | grep -v grep | awk '{print $2}') ")
    logging.info('process group flush!')


def ddp_all_gather(features, dim=0, requires_grad=True):
    '''
        inputs: [n, ...]
    '''

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    feature_list = [torch.ones_like(features) for _ in range(world_size)]
    torch.distributed.all_gather(feature_list, features.contiguous())

    if requires_grad:
        feature_list[rank] = features
    feature = torch.cat(feature_list, dim=dim)
    return feature


# https://github.com/pytorch/pytorch/issues/16885
class DDPPassthrough(DDP):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def get_ddp_module(module, find_unused_parameters=False, **kwargs):
    if len(list(module.parameters())) == 0:
        # for the case that loss module has not parameters.
        return module
    device = torch.cuda.current_device()
    module = DDPPassthrough(module, device_ids=[device], output_device=device,
                            find_unused_parameters=find_unused_parameters, **kwargs)
    return module


def params_count(net):
    n_parameters = sum(p.numel() for p in net.parameters())
    return 'Parameters Count: {:.5f}M'.format(n_parameters / 1e6)
