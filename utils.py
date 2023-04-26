
import math
import time
import datetime
import os
import subprocess
import functools
from collections import defaultdict, deque

import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn

from torchvision import models
from torch.utils.data import DataLoader, Subset

from torchvision.datasets.folder import is_image_file, default_loader

from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]

def build_backbone(path, name):
    """ Build a pretrained torchvision backbone from its name.

    Args:
        path: path to the checkpoint, can be an URL
        name: name of the architecture from torchvision (see https://pytorch.org/vision/stable/models.html) 
        or timm (see https://rwightman.github.io/pytorch-image-models/models/). 
        We highly recommand to use Resnet50 architecture as available in torchvision. 
        Using other architectures (such as non-convolutional ones) might need changes in the implementation.
    """
    if name == 'custom':
        # /checkpoint/matthijs/image_similarity_challenge/from_prod/f300531313.torchscript
        model = torch.jit.load(path)
        return model.to(device, non_blocking=True)
    if hasattr(models, name):
        model = getattr(models, name)(pretrained=True)
    else:
        import timm
        if name in timm.list_models():
            model = timm.models.create_model(name, num_classes=0)
        else:
            raise NotImplementedError('Model %s does not exist in torchvision'%name)
    model.head = nn.Identity()
    model.fc = nn.Identity()
    if path is not None:
        if path.startswith("http"):
            checkpoint = torch.hub.load_state_dict_from_url(path, progress=False)
        else:
            checkpoint = torch.load(path)
        state_dict = checkpoint
        for ckpt_key in ['state_dict', 'model_state_dict', 'teacher']:
            if ckpt_key in checkpoint:
                state_dict = checkpoint[ckpt_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
    return model.to(device, non_blocking=True)

def get_linear_layer(weight, bias):
    """ Create a linear layer from weight and bias matrices """
    dim_out, dim_in = weight.shape
    layer = nn.Linear(dim_in, dim_out)
    layer.weight = nn.Parameter(weight)
    layer.bias = nn.Parameter(bias)
    return layer

def load_normalization_layer(path, mode='whitening'):
    """ Loads the normalization layer from a checkpoint and returns the layer. """
    checkpoint = torch.load(path)
    if mode=='whitening':
        # if PCA whitening is used scale the feature by the dimension of the latent space
        D = checkpoint['weight'].shape[1] 
        weight = torch.nn.Parameter(D*checkpoint['weight'])
        bias = torch.nn.Parameter(D*checkpoint['bias'])
    else:
        weight = checkpoint['weight']
        bias = checkpoint['bias']
    return get_linear_layer(weight, bias).to(device, non_blocking=True)

class NormLayerWrapper(nn.Module):
    """
    Wraps backbone model and normalization layer
    """
    def __init__(self, backbone, head):
        super(NormLayerWrapper, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        output = self.backbone(x)
        return self.head(output)

def generate_messages(n, k):
    """
    Generate random original messages.
    Args:
        n: Number of messages to generate
        k: length of the message
    Returns:
        msgs: boolean tensor of size nxk
    """
    return torch.rand((n,k))>0.5

def string_to_binary(st):
    """ String to binary """
    return ''.join(format(ord(i), '08b') for i in st)

def binary_to_string(bi):
    """ Binary to string """
    return ''.join(chr(int(byte,2)) for byte in [bi[ii:ii+8] for ii in range(0,len(bi),8)] )

def get_num_bits(path, msg_type):
    """ Get the number of bits of the watermark from the text file """ 
    with open(path, 'r') as f:
        lines = [line.strip() for line in f]
    if msg_type == 'bit':
        return max([len(line) for line in lines])
    else:
        return 8*max([len(line) for line in lines])

def load_messages(path, msg_type, N):
    """ Load messages from a file """
    with open(path, 'r') as f:
        lines = [line.strip() for line in f]
    if msg_type == 'bit':
        num_bit = max([len(line) for line in lines])
        lines = [line + '0'*(num_bit-len(line)) for line in lines]
        msgs = [[int(i)==1 for i in line] for line in lines]
    else:
        num_byte = max([len(line) for line in lines])
        lines = [line + ' '*(num_byte-len(line)) for line in lines]
        msgs = [[int(i)==1 for i in string_to_binary(line)] for line in lines]
    msgs = msgs * (N//len(msgs)+1)
    return torch.tensor(msgs[:N])

def save_messages(msgs, path):
    """ Save messages to file """
    txt_msgs = [''.join(map(str, x.type(torch.int).tolist())) for x in msgs]
    txt_msgs = '\n'.join(txt_msgs)
    with open(os.path.join(path), 'w') as f:
        f.write(txt_msgs)

def parse_params(s):
    """
    Parse parameters into a dictionary, used for optimizer and scheduler parsing.
    Example: 
        "SGD,lr=0.01" -> {"name": "SGD", "lr": 0.01}
    """
    s = s.replace(' ', '').split(',')
    params = {}
    params['name'] = s[0]
    for x in s[1:]:
        x = x.split('=')
        params[x[0]]=float(x[1])
    return params

def build_optimizer(name, model_params, **optim_params):
    """ Build optimizer from a dictionary of parameters """
    torch_optimizers = sorted(name for name in torch.optim.__dict__
        if name[0].isupper() and not name.startswith("__")
        and callable(torch.optim.__dict__[name]))
    if hasattr(torch.optim, name):
        return getattr(torch.optim, name)(model_params, **optim_params)
    raise ValueError(f'Unknown optimizer "{name}", choose among {str(torch_optimizers)}')

def adjust_learning_rate(optimizer, step, steps, warmup_steps, blr, min_lr=1e-6):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if step < warmup_steps:
        lr = blr * step / warmup_steps 
    else:
        lr = min_lr + (blr - min_lr) * 0.5 * (1. + math.cos(math.pi * (step - warmup_steps) / (steps - warmup_steps)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr



def bool_inst(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected in args')

def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


### Data loading

@functools.lru_cache()
def get_image_paths(path):
    paths = []
    for path, _, files in os.walk(path):
        for filename in files:
            paths.append(os.path.join(path, filename))
    return sorted([fn for fn in paths if is_image_file(fn)])

class ImageFolder:
    """An image folder dataset intended for self-supervised learning."""

    def __init__(self, path, transform=None, loader=default_loader):
        self.samples = get_image_paths(path)
        self.loader = loader
        self.transform = transform

    def __getitem__(self, idx: int):
        assert 0 <= idx < len(self)
        img = self.loader(self.samples[idx])
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)

def collate_fn(batch):
    """ Collate function for data loader. Allows to have img of different size"""
    return batch

def get_dataloader(data_dir, transform, batch_size=128, num_imgs=None, shuffle=False, num_workers=4, collate_fn=collate_fn):
    """ Get dataloader for the images in the data_dir. The data_dir must be of the form: input/0/... """
    dataset = ImageFolder(data_dir, transform=transform)
    if num_imgs is not None:
        dataset = Subset(dataset, np.random.choice(len(dataset), num_imgs, replace=False))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=False, collate_fn=collate_fn)

def pil_imgs_from_folder(folder):
    """ Get all images in the folder as PIL images """
    images = []
    filenames = []
    for filename in os.listdir(folder):
        try:
            img = Image.open(os.path.join(folder,filename))
            if img is not None:
                filenames.append(filename)
                images.append(img)
        except:
            print("Error opening image: ", filename)
    return images, filenames


### Metric logging

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(header, total_time_str, total_time / (len(iterable)+1)))
