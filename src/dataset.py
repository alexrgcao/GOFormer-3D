# dataset.py
import os
import json
import pickle
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import numpy as np
import glob

class ProteinDataset(Dataset):
    """
    Lazily loads protein records from many .pkl files stored in subfolders under a root directory.
    Additionally, it loads a global GO vocabulary and a global atom order from JSON files.
    
    Each pickle file is expected to be a dictionary representing a single protein record with keys such as:
      - "protein": a protein identifier
      - "true_go": a list of GO term strings (e.g., ['GO:0009057', ...])
      - "coords": coordinate tensor (either [L, num_atoms, 3] or [L, 3])
      - "esm": ESM-2 embeddings (a NumPy array)
      - "plddt": pLDDT scores (a tensor or list)  [per-residue]
      - "pae": PAE matrix (a tensor or list)
      - "length": sequence length (optional)
      - Optionally, "atom_names": a list of atom names (if you need to reorder using global_atom_order)
      - Optionally, precomputed features (e.g. "centroid", "euclidean_distances", etc.)
      
    The global GO vocabulary JSON should map GO term strings to integer indices.
    The global atom order JSON should be a list of atom names in the desired order.
    """
    def __init__(self, root_dir, go_vocab_file, atom_order_file, seqs, eval_mode=False):
        self.eval_mode = eval_mode
        self.file_paths = []
        if not eval_mode:
            for dirpath, _, filenames in os.walk(root_dir):
                for filename in filenames:
                    if filename.endswith(".pkl"):
                        self.file_paths.append(os.path.join(dirpath, filename))
        else:
            for pid in seqs:
                pattern = os.path.join(root_dir, pid, "*.pkl")
                matches = glob.glob(pattern)
                if not matches:
                    print(f"[warn] no .pkl files found for {pid} at {pattern}")
                self.file_paths.extend(matches)

        self.file_paths.sort()
        with open(go_vocab_file, "r") as f:
            # Expecting a dictionary like {"GO:0009057": 0, "GO:0016998": 1, ...}
            self.go_vocab = json.load(f)

        with open(atom_order_file, "r") as f:
            # Expecting a list like ["CA", "N", "C", "O", ...]
            self.atom_order = json.load(f)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with open(file_path, "rb") as f:
            record = pickle.load(f)

        seq_embed = record['esm']  # [L, d_emb]
        if not isinstance(seq_embed, torch.Tensor):
            seq_embed = torch.tensor(seq_embed, dtype=torch.float32)

        coords = record["coords"]
        if not isinstance(coords, torch.Tensor):
            coords = torch.tensor(coords, dtype=torch.float32)
            if coords.ndim == 3 and "atom_names" in record:
                atom_names = record["atom_names"]  # e.g., ["N", "CA", "C", "O", ...]
                order_indices = [atom_names.index(atom) for atom in self.atom_order if atom in atom_names]
                coords = coords[:, order_indices, :]

        if coords.ndim == 3 and "atom_names" in record:
            atom_names = record["atom_names"]
            if isinstance(atom_names, str):
                atom_names = [atom_names]
            
            L = coords.shape[0]
            num_global_atoms = len(self.atom_order)

            new_coords = torch.zeros((L, num_global_atoms, 3), dtype=coords.dtype)
            
            for j, global_atom in enumerate(self.atom_order):
                if global_atom in atom_names:
                    local_index = atom_names.index(global_atom)
                    new_coords[:, j, :] = coords[:, local_index, :]

            coords = new_coords

        if "centroid" in record:
            centroid = record["centroid"]
            if not isinstance(centroid, torch.Tensor):
                centroid = torch.tensor(centroid, dtype=torch.float32)
        else:
            centroid = coords.mean(dim=1)  # [L, 3]

        if "euclidean_distances" in record:
            euclidean_distances = record["euclidean_distances"]
            if not isinstance(euclidean_distances, torch.Tensor):
                euclidean_distances = torch.tensor(euclidean_distances, dtype=torch.float32)
        else:
            diff = centroid.unsqueeze(1) - centroid.unsqueeze(0)  # [L, L, 3]
            euclidean_distances = torch.norm(diff, dim=-1)  # [L, L]

        if "edge_vectors" in record:
            edge_vectors = record["edge_vectors"]
            if not isinstance(edge_vectors, torch.Tensor):
                edge_vectors = torch.tensor(edge_vectors, dtype=torch.float32)
        else:
            edge_vectors = diff  # [L, L, 3]

        if "orientation_vectors" in record:
            orientation_vectors = record["orientation_vectors"]
            if not isinstance(orientation_vectors, torch.Tensor):
                orientation_vectors = torch.tensor(orientation_vectors, dtype=torch.float32)
        else:
            if "atom_names" in record:
                atom_names = record["atom_names"]
                try:
                    ca_index = atom_names.index("CA")
                    ca_coords = coords[:, ca_index, :]  # [L, 3]
                except ValueError:
                    ca_coords = centroid
            else:
                ca_coords = centroid
            orientation_vectors = ca_coords[1:] - ca_coords[:-1]  # [L-1, 3]
            norm = torch.norm(orientation_vectors, dim=-1, keepdim=True) + 1e-8
            orientation_vectors = orientation_vectors / norm
            orientation_vectors = torch.cat([orientation_vectors, orientation_vectors[-1:].clone()], dim=0)

        if "side_chain_vectors" in record:
            side_chain_vectors = record["side_chain_vectors"]
            if not isinstance(side_chain_vectors, torch.Tensor):
                side_chain_vectors = torch.tensor(side_chain_vectors, dtype=torch.float32)
        else:
            backbone_set = {"N", "CA", "C", "O"}
            if "atom_names" in record:
                atom_names = record["atom_names"]
                try:
                    ca_index = atom_names.index("CA")
                    ca_coords = coords[:, ca_index, :]  # [L, 3]
                except ValueError:
                    ca_coords = centroid
                side_chain_centroids = []
                for i in range(coords.shape[0]):
                    side_chain_atoms = []
                    for j, atom in enumerate(atom_names):
                        if atom not in backbone_set:
                            side_chain_atoms.append(coords[i, j, :])
                    if side_chain_atoms:
                        side_chain_atoms = torch.stack(side_chain_atoms, dim=0)  # [num_side_atoms, 3]
                        sc_centroid = side_chain_atoms.mean(dim=0)
                    else:
                        sc_centroid = ca_coords[i]
                    side_chain_centroids.append(sc_centroid)
                side_chain_centroids = torch.stack(side_chain_centroids, dim=0)  # [L, 3]
                side_chain_vectors = side_chain_centroids - ca_coords  # [L, 3]
            else:
                side_chain_vectors = torch.zeros_like(centroid)

        pae = record.get("pae")
        if pae is not None and not isinstance(pae, torch.Tensor):
            pae = torch.tensor(pae, dtype=torch.float32)
        plddt = record.get("plddt")
        if plddt is not None and not isinstance(plddt, torch.Tensor):
            plddt = torch.tensor(plddt, dtype=torch.float32)

        protein_id = record.get("protein", "unknown")
        seq_length = record.get("length", seq_embed.shape[0])

        out = {
            "id": protein_id,
            "seq_embed": seq_embed,          # [L, d_emb]
            "coords": coords,                # [L, num_atoms, 3] or [L, 3]
            "pae": pae,                      # [L, L] or None
            "plddt": plddt,                  # [L] or None
            "length": seq_length,
            "centroid": centroid,            # [L, 3]
            "euclidean_distances": euclidean_distances,  # [L, L]
            "edge_vectors": edge_vectors,    # [L, L, 3]
            "orientation_vectors": orientation_vectors,  # [L, 3]
            "side_chain_vectors": side_chain_vectors       # [L, 3]
        }

        if not self.eval_mode:
            label = record["label"]
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label, dtype=torch.float32)
            out["label"] = label

        return out

def protein_collate_fn(batch):
    """
    Optimized collate function for variable-length protein records.
    Pads *all* per-residue features to the same max_len.
    """
    batch.sort(key=lambda x: x["length"], reverse=True)
    ids     = [item["id"]     for item in batch]
    lengths = [item["length"] for item in batch]
    max_len = max(lengths)
    B       = len(batch)

    seqs = [item["seq_embed"] for item in batch]
    padded_seq = pad_sequence(seqs, batch_first=True, padding_value=0.)  # [B, max_len, d_emb]

    coords_list = [item["coords"] for item in batch]
    if coords_list[0].dim() == 3:
        # (L, A, 3) → pad L→max_len
        padded_coords = torch.stack([
            F.pad(c, (0,0, 0,0, 0, max_len - c.size(0)))
            for c in coords_list
        ], dim=0)
    else:
        # (L, 3) → pad L→max_len
        padded_coords = torch.stack([
            F.pad(c, (0, max_len - c.size(0)))
            for c in coords_list
        ], dim=0)

    # 4) pad PAE to [B, max_len, max_len]
    if batch[0].get("pae") is not None:
        pae_list = [item["pae"] for item in batch]
        padded_pae = torch.stack([
            F.pad(p, (0, max_len - p.size(1), 0, max_len - p.size(0)))
            for p in pae_list
        ], dim=0)
    else:
        padded_pae = None

    # 5) pad pLDDT to [B, max_len]
    if batch[0].get("plddt") is not None:
        plddt_list = [item["plddt"] for item in batch]
        padded_plddt = torch.stack([
            F.pad(pl, (0, max_len - pl.size(0)))
            for pl in plddt_list
        ], dim=0)
    else:
        padded_plddt = None

    labels = torch.stack([item["label"].squeeze(0) for item in batch], dim=0) \
             if "label" in batch[0] else None
    
    attn_mask = torch.zeros((B, max_len), dtype=torch.bool)
    for i, L in enumerate(lengths):
        attn_mask[i, L:] = True

    def pad_vec_list(key, feat_dim):
        vecs = [item[key] for item in batch]
        return torch.stack([
            F.pad(v, (0,0, 0, max_len - v.size(0)))
            for v in vecs
        ], dim=0)

    padded_centroid    = pad_vec_list("centroid",    3)
    padded_orientation = pad_vec_list("orientation_vectors", 3)
    padded_sidechain   = pad_vec_list("side_chain_vectors",   3)

    euclid_list = [item["euclidean_distances"] for item in batch]
    padded_euclid = torch.stack([
        F.pad(e, (0, max_len - e.size(1), 0, max_len - e.size(0)))
        for e in euclid_list
    ], dim=0)

    edge_list = [item["edge_vectors"] for item in batch]
    padded_edge = torch.stack([
        F.pad(e, (0,0, 0, max_len - e.size(1), 0, max_len - e.size(0)))
        for e in edge_list
    ], dim=0)

    return {
        "ids":                  ids,
        "seq_embed":           padded_seq,
        "coords":              padded_coords,
        "pae":                  padded_pae,
        "plddt":               padded_plddt,
        "label":               labels,
        "attn_mask":           attn_mask,
        "centroid":            padded_centroid,
        "orientation_vectors": padded_orientation,
        "side_chain_vectors":  padded_sidechain,
        "euclidean_distances": padded_euclid,
        "edge_vectors":        padded_edge,
    }

