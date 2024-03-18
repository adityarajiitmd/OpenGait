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

# The provided function get_attr_from retrieves an attribute from a list of objects recursively. Here's a breakdown of how it works:
# this function attempts to get the specified attribute from the first object in the list. If not found, it keeps trying with the remaining objects one by one until it either finds the attribute or reaches the end of the list.
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

# this function takes a PyTorch tensor x and returns a new NumPy array containing a copy of the tensor's data on the CPU.
# This conversion is often necessary when you want to use NumPy functionality on the tensor data or interact with libraries that expect NumPy arrays.
def ts2np(x):
    return x.cpu().data.numpy()

 # this function takes a NumPy array or a PyTorch tensor and converts it into a PyTorch variable suitable for training neural networks with automatic differentiation.
# The **kwargs arguments allow you to customize the behavior of the variable creation, such as enabling gradients or using volatile variables.
def ts2var(x, **kwargs):
    return autograd.Variable(x, **kwargs).cuda()

# the np2var function acts as a convenience function for converting NumPy arrays to PyTorch variables suitable for training neural networks with automatic differentiation.
def np2var(x, **kwargs):
    return ts2var(torch.from_numpy(x), **kwargs)

# the list2var function provides a way to convert Python lists to PyTorch variables suitable for neural network training.
# It achieves this by converting the list to a NumPy array first and then leveraging the np2var function for the variable creation process.
def list2var(x, **kwargs):
    return np2var(np.array(x), **kwargs)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# this function merges two dictionaries recursively.
# It prioritizes values from the source dictionary (src) and merges sub-dictionaries if both source and destination have them. It avoids conflicts by overwriting destination values with source values when necessary.
def MergeCfgsDict(src, dst):
    for k, v in src.items():
        if (k not in dst.keys()) or (type(v) != type(dict())):
            dst[k] = v
        else:
            if is_dict(src[k]) and is_dict(dst[k]):
                MergeCfgsDict(src[k], dst[k])
            else:
                dst[k] = v

# the clones function offers a convenient way to create multiple independent copies of a PyTorch module for use in various neural network structures.
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# this function provides a way to load configurations from a specified file, merge them with a default configuration file, and return the resulting merged configuration.
# This approach allows you to override default settings with custom configurations from the provided path.
def config_loader(path):
    with open(path, 'r') as stream:
        src_cfgs = yaml.safe_load(stream)
    with open("./configs/default.yaml", 'r') as stream:
        dst_cfgs = yaml.safe_load(stream)
    MergeCfgsDict(src_cfgs, dst_cfgs)
    return dst_cfgs

# The provided function init_seeds is designed to initialize random number generators (RNGs) used by various libraries in PyTorch for deterministic behavior.
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

# this function provides a basic way to handle Ctrl+C/Z interrupts by logging a message and potentially terminating the entire process group.
# However, a more controlled and safer approach is recommended for production environments.
def handler(signum, frame):
    logging.info('Ctrl+c/z pressed')
    os.system(
        "kill $(ps aux | grep main.py | grep -v grep | awk '{print $2}') ")
    logging.info('process group flush!')

# this function provides a mechanism to collect features from all processes involved in DDP training and combine them into a single tensor. 
# This can be useful for tasks like calculating global statistics or creating complete representations from distributed partial computations
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

# the DDPPassthrough class offers a way to access model attributes directly within a DDP context.
# However, it's crucial to use it cautiously and understand the potential implications for distributed training. Consider the trade-offs between convenience and potential disruptions to DDP optimizations.
# https://github.com/pytorch/pytorch/issues/16885
class DDPPassthrough(DDP):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

# the get_ddp_module function provides a flexible way to conditionally wrap PyTorch modules for DDP training.
# It offers features like early return for modules without parameters, customizable unused parameter tracking, and additional control through **kwargs.
def get_ddp_module(module, find_unused_parameters=False, **kwargs):
    if len(list(module.parameters())) == 0:
        # for the case that loss module has not parameters.
        return module
    device = torch.cuda.current_device()
    module = DDPPassthrough(module, device_ids=[device], output_device=device,
                            find_unused_parameters=find_unused_parameters, **kwargs)
    return module

# this function provides a simple way to count the parameters in a PyTorch neural network and present the count in a human-readable format suitable for tracking model complexity.
def params_count(net):
    n_parameters = sum(p.numel() for p in net.parameters())
    return 'Parameters Count: {:.5f}M'.format(n_parameters / 1e6)
