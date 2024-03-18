import time
import torch

import numpy as np
import torchvision.utils as vutils
import os.path as osp
from time import strftime, localtime

from torch.utils.tensorboard import SummaryWriter
from .common import is_list, is_tensor, ts2np, mkdir, Odict, NoOp
import logging

# This class is responsible for handling logging messages, writing summaries to TensorBoard, and managing training information.
class MessageManager:
    # Creates an Odict (ordered dictionary) named info_dict to store training information.
    def __init__(self):
        self.info_dict = Odict()
        self.writer_hparams = ['image', 'scalar']
        self.time = time.time()
# Sets iteration (training step counter) and log_iter (logging frequency).
# Creates a directory for TensorBoard summaries.
# Initializes a SummaryWriter object for writing to TensorBoard.
# Calls init_logger to set up logging.
    def init_manager(self, save_path, log_to_file, log_iter, iteration=0):
        self.iteration = iteration
        self.log_iter = log_iter
        mkdir(osp.join(save_path, "summary/"))
        self.writer = SummaryWriter(
            osp.join(save_path, "summary/"), purge_step=self.iteration)
        self.init_logger(save_path, log_to_file)
# Creates a logger named 'opengait' with INFO level.
# Sets up a formatter for log messages (including timestamps and severity levels).
# Optionally creates a log file and a file handler based on the log_to_file argument.
# Adds a console handler for logging messages to the console (stdout/stderr).
    def init_logger(self, save_path, log_to_file):
        # init logger
        self.logger = logging.getLogger('opengait')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        if log_to_file:
            mkdir(osp.join(save_path, "logs/"))
            vlog = logging.FileHandler(
                osp.join(save_path, "logs/", strftime('%Y-%m-%d-%H-%M-%S', localtime())+'.txt'))
            vlog.setLevel(logging.INFO)
            vlog.setFormatter(formatter)
            self.logger.addHandler(vlog)

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(logging.DEBUG)
        self.logger.addHandler(console)

# Takes a dictionary info containing training information.
# Converts elements in the dictionary to lists if not already a list.
# Converts tensors to NumPy arrays for easier processing.
# Updates the info_dict with the processed information.
    def append(self, info):
        for k, v in info.items():
            v = [v] if not is_list(v) else v
            v = [ts2np(_) if is_tensor(_) else _ for _ in v]
            info[k] = v
        self.info_dict.append(info)

# Clears the info_dict to avoid memory accumulation.
# Flushes data to the underlying TensorBoard writer.
    def flush(self):
        self.info_dict.clear()
        self.writer.flush()

# Iterates through the summary dictionary containing data to be written.
# Extracts the module name from the key (k) in the summary.
# Warns if the data type (module_name) in the summary is not expected (currently supports "image" and "scalar").
# Extracts the board name from the key after removing the module name.
# Retrieves the appropriate writer function (add_image or add_scalar) based on the module name.
# Detaches gradients from tensors (if any) before writing.
# Converts image data to a grid using vutils.make_grid for better visualization in TensorBoard.
# Calculates the mean for scalar values before writing.
# Writes the data to TensorBoard using the appropriate writer function (writer_module) with the board name and current iteration number.
    def write_to_tensorboard(self, summary):

        for k, v in summary.items():
            module_name = k.split('/')[0]
            if module_name not in self.writer_hparams:
                self.log_warning(
                    'Not Expected --Summary-- type [{}] appear!!!{}'.format(k, self.writer_hparams))
                continue
            board_name = k.replace(module_name + "/", '')
            writer_module = getattr(self.writer, 'add_' + module_name)
            v = v.detach() if is_tensor(v) else v
            v = vutils.make_grid(
                v, normalize=True, scale_each=True) if 'image' in module_name else v
            if module_name == 'scalar':
                try:
                    v = v.mean()
                except:
                    v = v
            writer_module(board_name, v, self.iteration)

# Records the current time and calculates the elapsed time since the last log.
# Formats a string with the current iteration number, elapsed time, and additional training information (mean of scalar values) from info_dict.
# Logs the formatted string using log_info.
# Resets the internal timer.
    def log_training_info(self):
        now = time.time()
        string = "Iteration {:0>5}, Cost {:.2f}s".format(
            self.iteration, now-self.time, end="")
        for i, (k, v) in enumerate(self.info_dict.items()):
            if 'scalar' not in k:
                continue
            k = k.replace('scalar/', '').replace('/', '_')
            end = "\n" if i == len(self.info_dict)-1 else ""
            string += ", {0}={1:.4f}".format(k, np.mean(v), end=end)
        self.log_info(string)
        self.reset_time()
# Resets the internal timer to mark the start of a new training step.
    def reset_time(self):
        self.time = time.time()

# Increments the iteration counter.
# Appends the provided training information (info) to the info_dict.
# Checks if the current iteration is a multiple of the log_iter (logging frequency).
# If yes, logs training information using log_training_info.
# Flushes data to TensorBoard and writes summaries using write_to_tensorboard.
    def train_step(self, info, summary):
        self.iteration += 1
        self.append(info)
        if self.iteration % self.log_iter == 0:
            self.log_training_info()
            self.flush()
            self.write_to_tensorboard(summary)

# Provide wrappers for logging messages at different severity levels (debug, info, warning) using the configured logger.
    def log_debug(self, *args, **kwargs):
        self.logger.debug(*args, **kwargs)

    def log_info(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)

    def log_warning(self, *args, **kwargs):
        self.logger.warning(*args, **kwargs)


msg_mgr = MessageManager()
noop = NoOp()

# This approach ensures that logging and message management are primarily handled by the main process in a distributed training setting.
def get_msg_mgr():
    if torch.distributed.get_rank() > 0:
        return noop
    else:
        return msg_mgr
