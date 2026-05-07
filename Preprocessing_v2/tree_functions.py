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

    if tokens[0] == 256: tokens = tokens[1:]
    if tokens[-1] == 256: tokens = tokens[:-1]

    tokens = tokens.to(device)
    mask = None
    if null_id is not None:
        mask = tokens == null_id
        if mask.any():
            tokens = tokens.clone()
            tokens[mask] = 0

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
