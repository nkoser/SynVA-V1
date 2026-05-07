import torch
import numpy as np

class Tree:

    def __init__(self, data, right = None, left = None):

        self.id = id(self)
        self.data = data

        self.right = right
        self.left = left

def deserialize_post_order_k(serial, k = 4):

	serial = serial.copy()

	def post_order(serial, k):

		if serial[-k:] == [0.0] * k:
			for i in range(k): serial.pop()
			return None
		
		data = {}

		if k == 4:
			data["r"] = serial.pop()
		else:
			data["r"] = []
			for i in range(k - 3):  data["r"].insert(0, serial.pop())

		data["z"] = serial.pop()
		data["y"] = serial.pop()
		data["x"] = serial.pop()

		tree = Tree(data)

		tree.right = post_order(serial, k)
		tree.left = post_order(serial, k)
		
		return tree    
	
	return post_order(serial, k)

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
    
def deserialize_pre_order_k(serial, k = 4):
    
    serial = serial.copy()

    if len(serial) > 0:

        if serial[:k] != [0.0] * k:
            
            data = {}

            data["x"] = serial.pop(0)
            data["y"] = serial.pop(0)
            data["z"] = serial.pop(0)

            if k == 4:
                data["r"] = serial.pop(0)
            else:
                data["r"] = []
                for i in range(k - 3):  data["r"].append(serial.pop(0))

            tree = Tree(data)

            left, ret = deserialize_pre_order_k(serial, k)
            right, ret = deserialize_pre_order_k(ret, k)

            tree.left = left
            tree.right = right

            return tree, ret

        else:
            return None, serial[k:]
        
    else:
        return None, []

def serialize_pre_order(tree):

    if tree == None: return [0.0] * 4
    return list(tree.data.values())[::-1] + serialize_pre_order(tree.left) + serialize_pre_order(tree.right)

def serialize_pre_order_k(tree, k = 4):

    if tree == None: return [0.0] * k
    return [tree.data["x"], tree.data["y"], tree.data["z"]] + list(tree.data["r"]) + serialize_pre_order_k(tree.left, k) + serialize_pre_order_k(tree.right, k)

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

def serialize_post_order_k(tree, k = 4):

    features = []

    def post_order(node):
        if node:

            post_order(node.left)
            post_order(node.right)
            
            features.append(list(map(float, list(node.data.values())[0])))  # Convert to float list and append
        else:
            features.append([0.0] * k)

    post_order(tree)

    return np.array(features, dtype=np.float32)  # Convert to NumPy array

def deserialize(serial, mode = "pre_order", k = 4):

    if mode == "pre_order": return deserialize_pre_order_k(serial, k)[0]
    if mode == "post_order": return deserialize_post_order_k(serial, k)
    if mode in {"pre_order_kcount", "pre_order_k"}:
        return deserialize_pre_order_kcount(serial, k)[0]
    if mode in {"pre_order_kdir", "pre_order_k_lr"}:
        return deserialize_pre_order_kdir(serial, k)[0]

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


def preorder_kcount_parent_indices(k_counts):

    values = np.asarray(k_counts, dtype=np.float32).reshape(-1)
    parents = np.full((values.shape[0],), -1, dtype=np.int64)
    stack = []

    for idx, value in enumerate(values.tolist()):
        while stack and stack[-1][1] <= 0:
            stack.pop()

        if stack:
            parents[idx] = int(stack[-1][0])
            stack[-1][1] -= 1

        k_children = int(round(float(value)))
        k_children = max(0, min(2, k_children))
        if k_children > 0:
            stack.append([idx, k_children])

    return parents


def absolute_positions_to_parent_relative(data, position_slice=(1, 4), copy=True):

    arr = np.array(data, dtype=np.float32, copy=copy)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D tree array, got shape {arr.shape}.")

    start, end = int(position_slice[0]), int(position_slice[1])
    if start < 0 or end > arr.shape[1] or start >= end:
        raise ValueError(f"Invalid position_slice={position_slice} for shape {arr.shape}.")

    parents = preorder_kcount_parent_indices(arr[:, 0])
    mask = parents >= 0
    if mask.any():
        base = np.asarray(data, dtype=np.float32)
        arr[mask, start:end] = base[mask, start:end] - base[parents[mask], start:end]
    return arr


def parent_relative_positions_to_absolute(data, position_slice=(1, 4), copy=True):

    arr = np.array(data, dtype=np.float32, copy=copy)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D tree array, got shape {arr.shape}.")

    start, end = int(position_slice[0]), int(position_slice[1])
    if start < 0 or end > arr.shape[1] or start >= end:
        raise ValueError(f"Invalid position_slice={position_slice} for shape {arr.shape}.")

    parents = preorder_kcount_parent_indices(arr[:, 0])
    for idx, parent_idx in enumerate(parents.tolist()):
        if parent_idx >= 0:
            arr[idx, start:end] = arr[parent_idx, start:end] + arr[idx, start:end]
    return arr


def absolute_control_points_to_node_local(
    data,
    position_slice=(1, 4),
    control_point_slices=((4, 12), (12, 20), (20, 28)),
    copy=True,
):

    arr = np.array(data, dtype=np.float32, copy=copy)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D tree array, got shape {arr.shape}.")

    pos_start, pos_end = int(position_slice[0]), int(position_slice[1])
    if (pos_end - pos_start) != len(control_point_slices):
        raise ValueError(
            "position_slice dimensionality must match the number of control_point_slices."
        )

    base = np.asarray(data, dtype=np.float32)
    for axis_idx, cp_slice in enumerate(control_point_slices):
        cp_start, cp_end = int(cp_slice[0]), int(cp_slice[1])
        if cp_start < 0 or cp_end > arr.shape[1] or cp_start >= cp_end:
            raise ValueError(f"Invalid control point slice {cp_slice} for shape {arr.shape}.")
        arr[:, cp_start:cp_end] = base[:, cp_start:cp_end] - base[:, pos_start + axis_idx : pos_start + axis_idx + 1]
    return arr


def node_local_control_points_to_absolute(
    data,
    position_slice=(1, 4),
    control_point_slices=((4, 12), (12, 20), (20, 28)),
    copy=True,
):

    arr = np.array(data, dtype=np.float32, copy=copy)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D tree array, got shape {arr.shape}.")

    pos_start, pos_end = int(position_slice[0]), int(position_slice[1])
    if (pos_end - pos_start) != len(control_point_slices):
        raise ValueError(
            "position_slice dimensionality must match the number of control_point_slices."
        )

    for axis_idx, cp_slice in enumerate(control_point_slices):
        cp_start, cp_end = int(cp_slice[0]), int(cp_slice[1])
        if cp_start < 0 or cp_end > arr.shape[1] or cp_start >= cp_end:
            raise ValueError(f"Invalid control point slice {cp_slice} for shape {arr.shape}.")
        arr[:, cp_start:cp_end] = arr[:, cp_start:cp_end] + arr[:, pos_start + axis_idx : pos_start + axis_idx + 1]
    return arr


def absolute_tree_to_local_geometry(
    data,
    position_slice=(1, 4),
    control_point_slices=((4, 12), (12, 20), (20, 28)),
    relative_positions=False,
    node_local_control_points=False,
    copy=True,
):

    base = np.array(data, dtype=np.float32, copy=copy)
    out = np.array(base, dtype=np.float32, copy=True)
    if node_local_control_points:
        out = absolute_control_points_to_node_local(
            base,
            position_slice=position_slice,
            control_point_slices=control_point_slices,
            copy=True,
        )
    if relative_positions:
        out = absolute_positions_to_parent_relative(out, position_slice=position_slice, copy=False)
    return out


def recenter_node_local_cps(
    data,
    control_point_slices=((4, 12), (12, 20), (20, 28)),
    copy=False,
):
    """Re-center the 8 cross-section CPs of every node at the origin in
    node-local frame. Required for generated trees because the model
    regresses each CP component independently — small per-component
    errors accumulate into a systematic centroid offset (CP centroid
    drifts ~7σ away from origin), which makes the generated cross-section
    sit beside the centerline instead of perpendicular and centered on it.
    GT cross-sections always have centroid ≈ 0 in node-local frame.
    """
    out = np.array(data, dtype=np.float32, copy=copy)
    (xa, xb), (ya, yb), (za, zb) = control_point_slices
    cp_x = out[:, xa:xb]
    cp_y = out[:, ya:yb]
    cp_z = out[:, za:zb]
    out[:, xa:xb] = cp_x - cp_x.mean(axis=1, keepdims=True)
    out[:, ya:yb] = cp_y - cp_y.mean(axis=1, keepdims=True)
    out[:, za:zb] = cp_z - cp_z.mean(axis=1, keepdims=True)
    return out


def local_geometry_tree_to_absolute(
    data,
    position_slice=(1, 4),
    control_point_slices=((4, 12), (12, 20), (20, 28)),
    relative_positions=False,
    node_local_control_points=False,
    copy=True,
):

    out = np.array(data, dtype=np.float32, copy=copy)
    if relative_positions:
        out = parent_relative_positions_to_absolute(out, position_slice=position_slice, copy=False)
    if node_local_control_points:
        out = node_local_control_points_to_absolute(
            out,
            position_slice=position_slice,
            control_point_slices=control_point_slices,
            copy=False,
        )
    return out

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

def tokens_to_data(tokens, device, decoder, null_id=None):

    if len(tokens.shape) != 1 : raise Exception("'tokens' shape must be 1")

    if tokens[0] == 256:
        tokens = tokens[1:]
    if tokens[-1] == 256:
        tokens = tokens[:-1]

    tokens = tokens.to(device)
    mask = None
    if null_id is not None:
        mask = tokens == null_id
        if mask.any():
            tokens = tokens.clone()
            tokens[mask] = 0

    quant_mode = str(getattr(getattr(decoder, "args", None), "quantization_mode", "legacy"))
    if quant_mode in {"stream", "stream_v2", "fsq"} and hasattr(decoder, "streams"):
        if hasattr(decoder, "get_stream_token_keys"):
            token_keys = list(decoder.get_stream_token_keys(include_k_count=True))
        else:
            stream_names = list(getattr(decoder, "streams", {}).keys())
            if not stream_names:
                raise RuntimeError("Stream decoder has no streams defined.")
            token_keys = ["k_count"] + stream_names
        tokens_per_row = len(token_keys)
        n_full = (tokens.numel() // tokens_per_row) * tokens_per_row
        if n_full <= 0:
            raise RuntimeError("Not enough tokens to decode any stream rows.")
        if n_full != tokens.numel():
            tokens = tokens[:n_full]
            if mask is not None:
                mask = mask[:n_full]

        rows = tokens.reshape(-1, tokens_per_row)
        indices = {}
        for i, key in enumerate(token_keys):
            indices[key] = rows[:, i].reshape(1, -1).long()
        if "k_count" not in indices:
            raise RuntimeError("Stream token layout must contain 'k_count'.")

        quant = decoder.entry_to_feature(indices, (1, rows.shape[0], 1))
        dec = decoder.decode(quant).detach()  # [1, L, 39]
        k_count = indices["k_count"].unsqueeze(-1).float()  # [1, L, 1]
        data = torch.cat([k_count, dec], dim=-1).cpu()  # [1, L, 40]

        if mask is not None and mask.any():
            row_mask = mask.reshape(rows.shape[0], tokens_per_row).all(dim=1).cpu()
            data[:, row_mask, :] = 0
    else:
        feat = decoder.entry_to_feature(tokens, (-1, 64))
        feat = feat.T.unsqueeze(0)

        data = decoder.decode(feat).detach().cpu()
        if mask is not None and mask.any():
            tokens_len = mask.numel()
            seq_len = data.shape[1]
            if seq_len > 0 and tokens_len % seq_len == 0:
                tokens_per_row = tokens_len // seq_len
                row_mask = mask.reshape(seq_len, tokens_per_row).all(dim=1).cpu()
                data[:, row_mask, :] = 0
            else:
                row_mask = mask[:data.shape[1]].cpu()
                data[:, row_mask, :] = 0

    return data

def tokens_to_tree(tokens, threshold = 1e-2, mode = "pre_order", device = None, decoder = None, null_id=None):

    tree = Tree({"x":0, "y":0, "z":0, "r":0})

    try:
        
        data = tokens_to_data(tokens, device, decoder, null_id=null_id)

        data[torch.abs(data) < threshold] = 0

        serial = list(data.flatten())
        if mode in {"pre_order_kcount", "pre_order_k", "pre_order_kdir", "pre_order_k_lr"}:
            k = int(data.shape[-1] - 1)
            tree = deserialize(serial, mode, k=k)
        else:
            tree = deserialize(serial, mode)

    except: print("< tokens_to_tree error >")

    return tree        

def is_valid_tree(tokens, threshold = 1e-2, mode = "pre_order", device = None, decoder = None, null_id=None):

    try:
        
        data = tokens_to_data(tokens, device, decoder, null_id=null_id)

        data[data < threshold] = 0

        serial = list(data.flatten())
        if mode in {"pre_order_kcount", "pre_order_k", "pre_order_kdir", "pre_order_k_lr"}:
            k = int(data.shape[-1] - 1)
            deserialize(serial, mode, k=k)
        else:
            deserialize(serial, mode)

        return True

    except: return False

def serialize_post_order_str(tree):

    def post_order(tree):

        if tree:

            post_order(tree.left)
            post_order(tree.right)

            ret[0] += '1_'+ str([np.round(float(v), 4) for v in list(tree.data.values())]) +';'

        else:
            ret[0] += '#;'

    ret = ['']
    post_order(tree)
    return ret[0][:-1]
