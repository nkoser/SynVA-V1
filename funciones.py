import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
from collections import OrderedDict


import os
import numpy as np
import Stage1.modelsMultitalk.stage1_vocaset as models
from Stage1.modelsMultitalk.stage1_vocaset import VQAutoEncoder
from Stage1.metrics.loss import calc_vq_loss
from Stage1.base.utilities import AverageMeter

class Args:
    def __init__(self):
        # LOSS settings
        self.quant_loss_weight = 1.

        # NETWORK settings 
        #self.arch = 'stage1_vocaset'
        self.in_dim = 4
        self.hidden_size = 1024
        self.num_hidden_layers = 6
        self.num_attention_heads = 8
        self.intermediate_size = 1536
        self.window_size = 1
        self.quant_factor = 0
        self.face_quan_num = 16
        self.neg = 0.2
        self.INaffine = False

        # Quantization mode
        # legacy: single codebook, split into face_quan_num chunks
        # factorized: multiple codebooks (factor_count) with factor_dim each
        self.quantization_mode = "legacy"
        self.factor_count = 4
        self.factor_dim = 128
        # factor projection: split | linear_shared | linear_per_factor
        self.factor_proj = "split"
        # Optional K-classification head
        self.use_k_head = False
        self.k_loss_weight = 1.0
        self.k_classes = 4

        # VQuantizer settings
        self.n_embed = 256
        self.zquant_dim = 64#64
        self.vq_beta = 0.25

        # TRAIN settings
        self.batch_size = 1  # batch size for training
        self.batch_size_val = 1  # batch size for validation during training
        self.base_lr = 0.0001
        self.StepLR = True
        self.poly_lr = False
        self.epochs = 50000
        self.step_size = 200
        self.gamma = 0.9


        ##stage 2
        self.device = 'cuda'  # or 'cpu'
        self.feature_dim = 128  # dimension for the feature after audio encoding
        self.vertice_dim = 31  # number of vertices * 3 (e.g., V * 3 for 3D coordinates)
        self.n_head = 8  # number of attention heads in the transformer decoder
        self.num_layers = 6  # number of layers in the transformer decoder
        self.period = 2#100  # period for positional encoding
        self.vqvae_pretrained_path = 'modelos-entrenados/major-oath.pth'  # path to pretrained VQ-VAE


K_MODE_SET = {"pre_order_kcount", "pre_order_k", "pre_order_kdir", "pre_order_k_lr"}


def _get_spline_slices(feature_dim, mode):
    offset = 1 if mode in K_MODE_SET else 0
    needed = offset + 3 + 36
    if feature_dim < needed:
        return None
    ctrl_start = offset + 3
    ctrl_end = ctrl_start + 24
    knot_end = ctrl_end + 12
    return offset, ctrl_start, ctrl_end, knot_end


def _get_xyz_slices(feature_dim, mode):
    offset = 1 if mode in K_MODE_SET else 0
    xyz_end = offset + 3
    if feature_dim < xyz_end:
        return None
    return offset, xyz_end


def _decode_child_count_from_k(k_value, mode):
    k_int = int(np.rint(float(k_value)))
    if mode in {"pre_order_kdir", "pre_order_k_lr"}:
        if k_int <= 0:
            return 0
        if k_int == 3:
            return 2
        return 1
    return int(np.clip(k_int, 0, 2))


def _parent_indices_from_k_values(k_values, mode):
    n = int(len(k_values))
    parents = np.full(n, -1, dtype=np.int64)
    stack = []

    for idx in range(n):
        while stack and stack[-1][1] <= 0:
            stack.pop()
        if idx > 0:
            if stack:
                parents[idx] = int(stack[-1][0])
                stack[-1][1] -= 1
            else:
                # Fallback for malformed sequences: keep chain valid.
                parents[idx] = idx - 1
        child_count = _decode_child_count_from_k(k_values[idx], mode)
        if child_count > 0:
            stack.append([idx, child_count])
    return parents


def _apply_spline_delta_torch(tensor, mode, inverse=False):
    if tensor.dim() != 2:
        return tensor
    slices = _get_spline_slices(tensor.shape[1], mode)
    if slices is None:
        return tensor
    _, ctrl_start, ctrl_end, _ = slices
    out = tensor.clone()
    coeffs = out[:, ctrl_start:ctrl_end].reshape(-1, 3, 8)
    if inverse:
        coeffs = torch.cumsum(coeffs, dim=-1)
    else:
        deltas = coeffs.clone()
        deltas[:, :, 1:] = coeffs[:, :, 1:] - coeffs[:, :, :-1]
        coeffs = deltas
    out[:, ctrl_start:ctrl_end] = coeffs.reshape(-1, 24)
    return out


def _apply_xyz_delta_torch(tensor, mode, inverse=False, root_zero=True):
    if tensor.dim() != 2 or mode not in K_MODE_SET:
        return tensor
    xyz_slices = _get_xyz_slices(tensor.shape[1], mode)
    if xyz_slices is None:
        return tensor

    out = tensor.clone()
    xyz_start, xyz_end = xyz_slices
    k_values = out[:, 0].detach().cpu().numpy().tolist()
    parents = _parent_indices_from_k_values(k_values, mode=mode)
    xyz = out[:, xyz_start:xyz_end]

    if inverse:
        abs_xyz = xyz.clone()
        for idx in range(abs_xyz.shape[0]):
            parent_idx = int(parents[idx])
            if parent_idx >= 0:
                abs_xyz[idx] = abs_xyz[parent_idx] + xyz[idx]
        out[:, xyz_start:xyz_end] = abs_xyz
    else:
        delta_xyz = xyz.clone()
        for idx in range(delta_xyz.shape[0]):
            parent_idx = int(parents[idx])
            if parent_idx >= 0:
                delta_xyz[idx] = xyz[idx] - xyz[parent_idx]
            elif root_zero:
                delta_xyz[idx] = torch.zeros_like(delta_xyz[idx])
        out[:, xyz_start:xyz_end] = delta_xyz

    return out


def apply_spline_delta(tree_tensor, mode):
    if torch.is_tensor(tree_tensor):
        return _apply_spline_delta_torch(tree_tensor, mode, inverse=False)
    tensor = torch.tensor(tree_tensor, dtype=torch.float32)
    return _apply_spline_delta_torch(tensor, mode, inverse=False).numpy()


def invert_spline_delta(tree_tensor, mode):
    if torch.is_tensor(tree_tensor):
        return _apply_spline_delta_torch(tree_tensor, mode, inverse=True)
    tensor = torch.tensor(tree_tensor, dtype=torch.float32)
    return _apply_spline_delta_torch(tensor, mode, inverse=True).numpy()


def apply_xyz_delta(tree_tensor, mode, root_zero=True):
    if torch.is_tensor(tree_tensor):
        return _apply_xyz_delta_torch(tree_tensor, mode, inverse=False, root_zero=root_zero)
    tensor = torch.tensor(tree_tensor, dtype=torch.float32)
    return _apply_xyz_delta_torch(tensor, mode, inverse=False, root_zero=root_zero).numpy()


def invert_xyz_delta(tree_tensor, mode):
    if torch.is_tensor(tree_tensor):
        return _apply_xyz_delta_torch(tree_tensor, mode, inverse=True)
    tensor = torch.tensor(tree_tensor, dtype=torch.float32)
    return _apply_xyz_delta_torch(tensor, mode, inverse=True).numpy()

class Tree:

    def __init__(self, data, right = None, left = None):

        self.id = id(self)
        self.data = data

        self.right = right
        self.left = left

def deserialize_post_order(serial):

    serial = serial.copy()

    def post_order(serial):

        if serial[-4:] == [0.0] * 4:
            for i in range(4): serial.pop()
            return None
        
        data = {}

        data["r"] = serial.pop()
        data["z"] = serial.pop()
        data["y"] = serial.pop()
        data["x"] = serial.pop()

        tree = Tree(data)

        tree.right = post_order(serial)
        tree.left = post_order(serial)
        
        return tree    
    
    return post_order(serial)

def deserialize_pre_order(serial):
    
    serial = serial.copy()

    if len(serial) > 0:

        if serial[:4] != [0.0] * 4:
            
            data = {}

            data["x"] = serial.pop(0)
            data["y"] = serial.pop(0)
            data["z"] = serial.pop(0)
            data["r"] = serial.pop(0)

            tree = Tree(data)

            left, ret = deserialize_pre_order(serial)
            right, ret = deserialize_pre_order(ret)

            tree.left = left
            tree.right = right

            return tree, ret

        else:
            return None, serial[4:]
        
    else:
        return None, []

def serialize_pre_order(tree, k):

    if tree == None: return [0.0] * k
    return list(tree.data.values())[::-1] + serialize_pre_order(tree.left) + serialize_pre_order(tree.right)

def serialize_pre_order_kcount(tree, k=4):

    if tree is None:
        return []

    if tree.left is not None and tree.right is not None:
        children = [tree.left, tree.right]
        k_children = 2
    elif tree.left is not None:
        children = [tree.left]
        k_children = 1
    elif tree.right is not None:
        children = [tree.right]
        k_children = 1
    else:
        children = []
        k_children = 0

    if k == 4:
        attrs = [tree.data["x"], tree.data["y"], tree.data["z"], tree.data["r"]]
    else:
        attrs = [tree.data["x"], tree.data["y"], tree.data["z"]] + list(tree.data["r"])

    serial = [float(k_children)] + attrs
    for child in children:
        serial.extend(serialize_pre_order_kcount(child, k))
    return serial

def deserialize(serial, mode = "pre_order", k=4):

    if mode == "pre_order": return deserialize_pre_order(serial)[0]
    if mode == "post_order": return deserialize_post_order(serial)
    if mode in {"pre_order_kcount", "pre_order_k"}:
        return deserialize_pre_order_kcount(serial, k=k)[0]
    if mode in {"pre_order_kdir", "pre_order_k_lr"}:
        return deserialize_pre_order_kdir(serial, k=k)[0]

    print("UNSUPPORTED DESERIALIZATION MODE")

def deserialize_pre_order_kcount(serial, k=4):

    serial = serial.copy()

    def parse_k_children(value):
        k_children = int(round(float(value)))
        return max(0, min(2, k_children))

    def parse(seq):
        if len(seq) < 1 + k:
            return None, seq

        k_children = parse_k_children(seq.pop(0))

        data = {
            "x": seq.pop(0),
            "y": seq.pop(0),
            "z": seq.pop(0),
        }
        if k == 4:
            data["r"] = seq.pop(0)
        else:
            data["r"] = [seq.pop(0) for _ in range(k - 3)]

        tree = Tree(data)

        if k_children == 0:
            return tree, seq
        if k_children >= 1:
            left, seq = parse(seq)
            tree.left = left
        if k_children >= 2:
            right, seq = parse(seq)
            tree.right = right

        return tree, seq

    return parse(serial)

def serialize_pre_order_kdir(tree, k=4):

    if tree is None:
        return []

    if tree.left is not None and tree.right is not None:
        children = [tree.left, tree.right]
        k_children = 3
    elif tree.left is not None:
        children = [tree.left]
        k_children = 1
    elif tree.right is not None:
        children = [tree.right]
        k_children = 2
    else:
        children = []
        k_children = 0

    if k == 4:
        attrs = [tree.data["x"], tree.data["y"], tree.data["z"], tree.data["r"]]
    else:
        attrs = [tree.data["x"], tree.data["y"], tree.data["z"]] + list(tree.data["r"])

    serial = [float(k_children)] + attrs
    for child in children:
        serial.extend(serialize_pre_order_kdir(child, k))
    return serial

def deserialize_pre_order_kdir(serial, k=4):

    serial = serial.copy()

    def parse_k_children(value):
        k_children = int(round(float(value)))
        return max(0, min(3, k_children))

    def parse(seq):
        if len(seq) < 1 + k:
            return None, seq

        k_children = parse_k_children(seq.pop(0))

        data = {
            "x": seq.pop(0),
            "y": seq.pop(0),
            "z": seq.pop(0),
        }
        if k == 4:
            data["r"] = seq.pop(0)
        else:
            data["r"] = [seq.pop(0) for _ in range(k - 3)]

        tree = Tree(data)

        if k_children == 0:
            return tree, seq
        if k_children in (1, 3):
            left, seq = parse(seq)
            tree.left = left
        if k_children in (2, 3):
            right, seq = parse(seq)
            tree.right = right

        return tree, seq

    return parse(serial)
    
class IntraDataset(Dataset):

    def __init__(
        self,
        file_list,
        root_dir,
        mode = "pre_order",
        p = None,
        val = False,
        delta_spline=False,
        delta_xyz=False,
        noise_std=0.0,
        canonical_kcount_from_dir=False,
        cache_tensors=False,
        cache_max_items=0,
    ):

        
        #self.folder_path = folder_path
        #self.file_list = os.listdir(folder_path)  # Call os.listdir only once
        self.mode = mode
        self.root_dir = Path(root_dir)
        self.file_list = []
        self.val = val
        self.delta_spline = delta_spline
        self.delta_xyz = bool(delta_xyz)
        self.noise_std = float(noise_std)
        self.canonical_kcount_from_dir = bool(canonical_kcount_from_dir)
        self.cache_tensors = bool(cache_tensors)
        self.cache_max_items = int(cache_max_items) if cache_max_items is not None else 0
        if self.cache_max_items < 0:
            self.cache_max_items = 0
        self._tensor_cache = OrderedDict() if self.cache_tensors else None
        for rel_path in file_list:
            if p is not None:
                full_path = self.root_dir / f"p{p}" / rel_path
            else:
                full_path = self.root_dir / rel_path
            self.file_list.append(str(full_path))
        
        # Split dataset for train and validation
        total_files = len(self.file_list)
       
        print("Dataset size:", len(self))
        if self.cache_tensors:
            max_items_msg = "all" if self.cache_max_items <= 0 else str(self.cache_max_items)
            print(f"Tensor cache enabled: max_items={max_items_msg}")

    def __len__(self):
        return len(self.file_list)

    def _cache_get(self, idx):
        if self._tensor_cache is None:
            return None
        key = int(idx)
        item = self._tensor_cache.get(key)
        if item is not None:
            self._tensor_cache.move_to_end(key)
        return item

    def _cache_put(self, idx, value):
        if self._tensor_cache is None:
            return
        key = int(idx)
        self._tensor_cache[key] = value
        self._tensor_cache.move_to_end(key)
        if self.cache_max_items > 0:
            while len(self._tensor_cache) > self.cache_max_items:
                self._tensor_cache.popitem(last=False)

    def _load_tree_tensor(self, file_path):
        # Use memory mapping to avoid loading full file into memory
        tree_data_np = np.load(file_path, mmap_mode='r')
        
        # Convert to tensor only when accessed
        tree_tensor = torch.tensor(tree_data_np, dtype=torch.float32)
        if tree_tensor.dim() == 2:
            file_dim = tree_tensor.shape[1]
        else:
            file_dim = None
        if file_dim is None:
            if self.mode in {"pre_order_kcount", "pre_order_k", "pre_order_kdir", "pre_order_k_lr"}:
                file_dim = 40 if (tree_tensor.numel() % 40 == 0) else 39
            else:
                file_dim = 39
        tree_tensor = tree_tensor.reshape((-1, file_dim))
        if torch.isnan(tree_tensor).any() or torch.isinf(tree_tensor).any():
            tree_tensor = torch.nan_to_num(tree_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        if self.mode == "pre_order":

            serial_tree = list(tree_tensor.flatten().numpy())

            print(len(serial_tree))

            tree = deserialize(serial_tree, mode = "post_order")
            serial_tree = serialize_pre_order(tree, k=39)
            np_tree = np.array(serial_tree).reshape((-1,39))
            tree_tensor = torch.tensor(np_tree, dtype = torch.float32)

        if self.mode in {"pre_order_kcount", "pre_order_k"}:

            if file_dim >= 40:
                if self.canonical_kcount_from_dir:
                    # Existing datasets in pre_order_kdir may use {0,2,3}.
                    # Convert to child-count semantics {0,1,2} when token 3 is present.
                    k_vals = torch.round(tree_tensor[:, 0])
                    if torch.any(k_vals == 3):
                        k_new = torch.where(
                            k_vals <= 0,
                            torch.zeros_like(k_vals),
                            torch.where(k_vals == 3, torch.full_like(k_vals, 2.0), torch.ones_like(k_vals)),
                        )
                        tree_tensor[:, 0] = k_new
                return tree_tensor, False
            serial_tree = list(tree_tensor.flatten().numpy())
            tree = deserialize(serial_tree, mode="post_order")
            serial_tree = serialize_pre_order_kcount(tree, k=39)
            np_tree = np.array(serial_tree).reshape((-1, 40))
            tree_tensor = torch.tensor(np_tree, dtype=torch.float32)

        if self.mode in {"pre_order_kdir", "pre_order_k_lr"}:

            if file_dim >= 40:
                return tree_tensor, False
            serial_tree = list(tree_tensor.flatten().numpy())
            tree = deserialize(serial_tree, mode="post_order")
            serial_tree = serialize_pre_order_kdir(tree, k=39)
            np_tree = np.array(serial_tree).reshape((-1, 40))
            tree_tensor = torch.tensor(np_tree, dtype=torch.float32)

            
        if self.delta_spline:
            tree_tensor = apply_spline_delta(tree_tensor, self.mode)
        if self.delta_xyz:
            tree_tensor = apply_xyz_delta(tree_tensor, self.mode)

        return tree_tensor, True

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        cached = self._cache_get(idx)
        if cached is None:
            tree_tensor, allow_noise = self._load_tree_tensor(file_path)
            self._cache_put(idx, (tree_tensor, allow_noise))
        else:
            tree_tensor, allow_noise = cached

        if allow_noise and self.noise_std > 0 and not self.val:
            tree_tensor = tree_tensor + torch.randn_like(tree_tensor) * self.noise_std

        if not self.val:
            return tree_tensor
        else:
            return tree_tensor, file_path


def save_best_model(model, optimizer, epoch, loss, best_loss, model_save_path="best_model.pth"):
    """
    Save the model if the current loss is better than the best recorded loss.
    
    Args:
        model (torch.nn.Module): The model being trained.
        optimizer (torch.optim.Optimizer): The optimizer used in training.
        epoch (int): The current epoch number.
        loss (float): The current epoch's loss.
        best_loss (float): The best loss recorded so far.
        model_save_path (str): Path to save the best model.
    
    Returns:
        float: The updated best loss (could be the same or updated if the model improved).
    """
    if loss < best_loss:
        #print(f"Epoch [{epoch+1}], New best model found! Loss: {loss:.4f}")
        best_loss = loss
        
        # Save the model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, model_save_path)

    return best_loss

def save_best_model_gpt2(model, optimizer, epoch, loss, best_loss, model_save_path):
    """
    Save the model if the current loss is better than the best recorded loss.
    
    Args:
        model (torch.nn.Module): The model being trained.
        optimizer (torch.optim.Optimizer): The optimizer used in training.
        epoch (int): The current epoch number.
        loss (float): The current epoch's loss.
        best_loss (float): The best loss recorded so far.
        model_save_path (str): Path to save the best model.
    
    Returns:
        float: The updated best loss (could be the same or updated if the model improved).
    """
    if loss < best_loss:
        #print(f"Epoch [{epoch+1}], New best model found! Loss: {loss:.4f}")
        best_loss = loss
        
        # Save the model
        model.save_pretrained(model_save_path)
        
    return best_loss

import os

def erase_all_files(folder_path):

    # Iterate through all items in the folder

    for filename in os.listdir(folder_path):

        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path): os.remove(file_path)
