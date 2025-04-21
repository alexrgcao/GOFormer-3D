#!/usr/bin/env python
import os
import pickle
import argparse
import time

import torch
import requests
from transformers import AutoTokenizer, AutoModel
from io import StringIO
from Bio.PDB import PDBParser
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from datetime import datetime

model_lock = Lock()

def load_esm2_model():
    """Load the ESM-2 model and tokenizer from Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
    model = AutoModel.from_pretrained("facebook/esm2_t36_3B_UR50D")
    model.eval()
    return tokenizer, model

def esm2_embed_sequence(seq, tokenizer, model, device="cpu"):
    inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with model_lock:
        with torch.no_grad():
            outputs = model(**inputs)
    embedding = outputs.last_hidden_state.squeeze(0).cpu().numpy()
    return embedding

def fetch_alphafold_features(uniprot_id, global_atom_order_path, local_pdb_dir, timeout=10):
    """
    Fetch structure features from the AlphaFold API for the given UniProt ID.
    Instead of downloading the PDB content, it looks up the local PDB file.
    Parses the PDB and extracts:
       - Full-atom coordinates for each residue (as a list of dictionaries mapping atom names to [x, y, z])
       - pLDDT scores (from the B-factor field of the CA atoms)
       - PAE matrix (downloaded from the 'paeDocUrl' if available)
    Then converts the full-atom coordinates into a dense tensor of shape (L, M, 3),
    where L is the number of residues and M is determined by the global atom order.
    pLDDT scores and the PAE matrix are also converted to torch tensors.
    
    """
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if not data:
            print(f"No structure available for {uniprot_id}.")
            return None
        entry = data[0]

        pdb_path = os.path.join(local_pdb_dir, f"{uniprot_id}.pdb")
        if not os.path.exists(pdb_path):
            print(f"PDB file not found locally for {uniprot_id} at {pdb_path}.")
            return None

        with open(pdb_path, "r") as f:
            pdb_content = f.read()

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(uniprot_id, StringIO(pdb_content))

        raw_coords = []      # List of dictionaries (one per residue): {atom_name: coordinate (np.array of shape (3,))}
        plddt_scores = []    # pLDDT scores from CA atoms.
        model_struct = next(structure.get_models())
        for chain in model_struct:
            for residue in chain:
                residue_coords = {}
                for atom in residue:
                    residue_coords[atom.get_name()] = atom.get_coord()
                if residue_coords:
                    raw_coords.append(residue_coords)
                if residue.has_id("CA"):
                    ca = residue["CA"]
                    plddt_scores.append(ca.get_bfactor())
                else:
                    plddt_scores.append(0.0)

        if global_atom_order_path and os.path.exists(global_atom_order_path):
            with open(global_atom_order_path, "r") as f:
                all_atom_names = json.load(f)
        else:
            all_atom_names = sorted(list({atom_name for residue in raw_coords for atom_name in residue.keys()}))
        M = len(all_atom_names)

        per_id_atom_names = sorted({
            atom_name
            for residue in raw_coords
            for atom_name in residue.keys()
        })

        residue_coord_list = []
        for residue in raw_coords:
            coords_for_residue = []
            
            for atom_name in all_atom_names:
                if atom_name not in all_atom_names:
                    print(f"Warning: Atom name {atom_name} not in global atom order for {uniprot_id}.")
                if atom_name in residue:
                    coords_for_residue.append(residue[atom_name])
                else:
                    coords_for_residue.append(np.zeros(3, dtype=np.float32))

            residue_coord_list.append(coords_for_residue)
        coords_array = np.stack(residue_coord_list)
        coords_tensor = torch.tensor(coords_array, dtype=torch.float32)

        plddt_tensor = torch.tensor(plddt_scores, dtype=torch.float32)

        pae = None
        pae_url = entry.get("paeDocUrl")
        if pae_url:
            pae_response = requests.get(pae_url, timeout=timeout)
            pae_response.raise_for_status()
            pae_json = pae_response.json()
            if isinstance(pae_json, list) and pae_json and isinstance(pae_json[0], dict) and "predicted_aligned_error" in pae_json[0]:
                pae_array = np.array(pae_json[0]["predicted_aligned_error"], dtype=np.float32)
                pae = torch.tensor(pae_array, dtype=torch.float32)
            elif "predicted_aligned_error" in pae_json:
                pae_array = np.array(pae_json["predicted_aligned_error"], dtype=np.float32)
                pae = torch.tensor(pae_array, dtype=torch.float32)
            else:
                pae = None
                print(f"PAE matrix NOT found or incorrect format for {uniprot_id}.")

        return {
            "coords": coords_tensor,         # Tensor of shape (L, M, 3)
            "plddt_scores": plddt_tensor,      # Tensor of shape (L,)
            "pae": pae,                         # Tensor of shape (L, L)
            "atom_names": per_id_atom_names,       # List of atom names in the global order
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching AF2 structure for {uniprot_id}: {e}")
        return None
    except Exception as e:
        print(f"Error processing PDB for {uniprot_id}: {e}")
        return None

def to_label(gos, tok_to_idx, vocab_size):
    """
    Convert a list of GO term strings into a multi-hot label tensor.
    
    Args:
        gos (List[str]): List of GO term IDs for a protein.
        tok_to_idx (dict): Mapping from token to index.
        vocab_size (int): Total vocabulary size.
    
    Returns:
        torch.Tensor: Tensor of shape (1, vocab_size) with 1s at positions corresponding to present GO terms.
    """
    label = torch.zeros((1, vocab_size), dtype=torch.float32)
    for go in gos:
        if go in tok_to_idx:
            label[0, tok_to_idx[go]] = 1.0
    return label

def update_protein_entry(entry, esm_tokenizer, esm_model, device, tok_to_idx, global_atom_order_path, local_pdb_dir):
    """
    Update a single protein entry:
      - Recompute the ESM-2 embedding from "seq".
      - Fetch AF2 structure features (Cα coords and pLDDT) using "protein".
      - Recompute the label vector from "true_go" using global_go_terms.
    """
    seq = entry.get("seq")
    if seq:
        max_retries = 30
        new_esm = None
        for attempt in range(max_retries):
            try:
                new_esm = esm2_embed_sequence(seq, esm_tokenizer, esm_model, device=device)
                break
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"CUDA OOM encountered for {entry.get('protein')}, attempt {attempt+1}/{max_retries}. Clearing cache and retrying...")
                    torch.cuda.empty_cache()
                    time.sleep(5)
                else:
                    print(f"Error computing ESM2 embedding for {entry.get('protein')}: {e}")
                    break
        if new_esm is not None:
            entry["esm"] = new_esm
        else:
            entry["esm"] = []

    else:
        entry["esm"] = []

    prot_id = entry.get("protein")
    if prot_id:
        af_features = fetch_alphafold_features(prot_id, global_atom_order_path, local_pdb_dir)
        if af_features:
            entry["coords"] = af_features.get("coords", torch.Tensor())
            entry["plddt"] = af_features.get("plddt_scores", torch.Tensor())
            entry["pae"] = af_features.get("pae", None)
        else:
            entry["coords"] = torch.Tensor()
            entry["plddt"] = torch.Tensor()
            entry["pae"] = None
    else:
        entry["coords"] = []
        entry["plddt"] = []

    true_go = entry.get("true_go", [])
    if tok_to_idx and true_go:
        entry["label"] = to_label(true_go, tok_to_idx, len(tok_to_idx))
    else:
        entry["label"] = []

    return entry

def process_pickle_file(input_path, output_path, esm_tokenizer, esm_model, device, tok_to_idx, global_atom_order_path, local_pdb_dir):
    """
    Loads a pickle file from input_path, which is a single protein entry (a dict with keys:
    "protein", "true_go", "coords", "label", "plddt", "seq", "esm", "length").
    
    It updates the entry using update_protein_entry (to recompute ESM2 embeddings, 
    fetch AF2 structure features, and update the label vector), and then saves 
    the updated entry to output_path.
    """
    import os, pickle

    try:
        with open(input_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        return False

    if isinstance(data, dict) and "protein" in data:
        updated_entry = update_protein_entry(data, esm_tokenizer, esm_model, device, tok_to_idx, global_atom_order_path, local_pdb_dir)
    else:
        print(f"Data in {input_path} is not a valid protein entry.")
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(output_path, "wb") as f:
            pickle.dump(updated_entry, f)
        return True
    except Exception as e:
        print(f"Error saving {output_path}: {e}")
        return False


def collect_global_go_terms(root):
    """
    Recursively collects all GO terms from the 'true_go' key of pickle files in the given root.
    Returns a sorted list of unique GO terms.
    """
    all_terms = set()
    for current_folder, _, files in os.walk(root):
        for file in files:
            if file.endswith(".pkl"):
                pkl_path = os.path.join(current_folder, file)
                try:
                    with open(pkl_path, "rb") as f:
                        data = pickle.load(f)
                except Exception:
                    continue
                if isinstance(data, dict):
                    entry_list = list(data.values())
                elif isinstance(data, list):
                    entry_list = data
                else:
                    entry_list = [data]
                for entry in entry_list:
                    if isinstance(entry, dict) and "true_go" in entry:
                        all_terms.update(entry["true_go"])
    return sorted(list(all_terms))

def process_root(input_root, output_root, esm_tokenizer, esm_model, device, tok_to_idx, pdb_dir, global_atom_order_path, num_threads):
    """
    Processes all pickle files under input_root and saves the updated files in output_root,
    preserving subfolder structure. Uses multi-threading to speed up processing.
    """
    futures = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for current_folder, _, files in os.walk(input_root):
            for file in files:
                if file.endswith(".pkl"):
                    input_path = os.path.join(current_folder, file)
                    rel_path = os.path.relpath(current_folder, input_root)
                    output_path = os.path.join(output_root, rel_path, file)
                    futures.append(executor.submit(process_pickle_file, input_path, output_path, esm_tokenizer, esm_model, device, tok_to_idx, global_atom_order_path, pdb_dir))
        for future in as_completed(futures):
            try:
                _ = future.result()
            except Exception as exc:
                print(f"Error in processing thread: {exc}")

def main():
    parser = argparse.ArgumentParser(
        description="Update PANDA-3D pickle files with new ESM2 embeddings, AlphaFold2 structure features, and label vectors."
    )
    parser.add_argument("--input_root", required=True,
                        help="Root directory containing cleaned PANDA-3D pickle files (in subfolders).")
    parser.add_argument("--output_root", required=True,
                        help="New root directory where updated pickle files will be saved (in the same subfolder structure).")
    parser.add_argument("--go_vocab_path", required=True,
                        help="Path to a JSON file containing the global GO term vocabulary.")
    parser.add_argument("--num_threads", type=int, default=8,
                        help="Number of threads to use for processing.")
    parser.add_argument("--pdb_dir", required=True,
                        help="Directory containing the local PDB files.")
    parser.add_argument("--global_atom_order_path", required=True,
                        help="Path to the global atom order JSON file.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    esm_tokenizer, esm_model = load_esm2_model()
    esm_model.to(device)

    with open(args.go_vocab_path, "r") as f:
        tok_to_idx = json.load(f)

    process_root(args.input_root, args.output_root, esm_tokenizer, esm_model, device, tok_to_idx, args.pdb_dir, args.global_atom_order_path, args.num_threads)
    print("Processing complete.")

if __name__ == "__main__":
    main()
