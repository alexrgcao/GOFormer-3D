#!/usr/bin/env python3
import os
import time
import glob
import json
import pickle
import argparse
import requests
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from requests.adapters import HTTPAdapter, Retry
from requests.exceptions import HTTPError
from urllib.parse import urljoin

from download_all_pdb import download_pdbs_multithread
from preprocess_panda3d import load_esm2_model, esm2_embed_sequence, fetch_alphafold_features
from dataset import ProteinDataset, protein_collate_fn
from model import ProteinFunctionModel

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import random

random.seed(42)

# 1) decide device once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) load & move the model once
tokenizer, esm_model = load_esm2_model()
esm_model.to(device)
esm_model.eval()
# ── UniProt ID‐mapping setup ───────────────────────────────────────────────────

API_BASE = "https://rest.uniprot.org/"
session = requests.Session()
session.mount("https://", HTTPAdapter(
    max_retries=Retry(total=5, backoff_factor=0.25, status_forcelist=[500,502,503,504])
))

import time
import requests
from requests.adapters import HTTPAdapter, Retry
from requests.exceptions import HTTPError
from urllib.parse import urljoin

API_BASE = "https://rest.uniprot.org/"
session = requests.Session()
session.mount(
    "https://",
    HTTPAdapter(max_retries=Retry(total=5, backoff_factor=0.25,
                                  status_forcelist=[500,502,503,504]))
)

def batch_lookup_accession_by_entry(entry_names, batch_size=500):
    """
    Given a list of UniProt entry‑names, return a dict mapping
    entry_name -> primaryAccession by querying in batches via
    a single OR‑query per batch.
    """
    to_acc = {}
    for i in range(0, len(entry_names), batch_size):
        chunk = entry_names[i : i + batch_size]
        quoted = [f'"{name}"' for name in chunk]
        lucene = "id:(" + " OR ".join(quoted) + ")"
        resp = session.get(
            urljoin(API_BASE, "uniprotkb/search"),
            params={
                "query":  lucene,
                "fields": "accession,id",
                "format": "json",
                "size":   len(chunk),
            },
        )
        resp.raise_for_status()
        for rec in resp.json().get("results", []):
            # 'primaryAccession' is the UniProt accession
            acc   = rec["primaryAccession"]
            # 'uniProtkbId' is the entry name (same as our desc)
            entry = rec.get("uniProtkbId") or rec.get("id")
            to_acc[entry] = acc
    return to_acc


def map_uniprot_ids(
    ids,
    from_db="UniProtKB_AC-ID",
    to_db="UniProtKB",
    fmt="tsv"
):
    """
    Map a list of UniProt entry names (or ACs) to UniProt accessions.
    Returns: List[{"From":<entry_name>,"To":<accession>}, ...]
    """
    url = urljoin(API_BASE, "idmapping/run")
    # IDs may be separated by spaces or commas; we'll use spaces here
    payload = {
        "from": from_db,
        "to":   to_db,
        "ids":  " ".join(ids)
    }

    try:
        resp = session.post(url, data=payload)
        resp.raise_for_status()
    except HTTPError as e:
        # this will now only fire if your parameters are invalid
        print(f"[UniProt mapping] HTTP {e.response.status_code} for IDs {ids[:5]}…: {e}")
        return []

    job_id = resp.json()["jobId"]

    # poll until finished
    status_url = urljoin(API_BASE, f"idmapping/status/{job_id}")
    while True:
        s = session.get(status_url); s.raise_for_status()
        if s.json().get("jobStatus") == "RUNNING":
            time.sleep(3)
            continue
        break

    # get the redirect URL for results
    details = session.get(urljoin(API_BASE, f"idmapping/details/{job_id}"))
    details.raise_for_status()
    results_url = details.json()["redirectURL"]
    # force TSV format
    sep = "&" if "?" in results_url else "?"
    results_url = results_url + f"{sep}format=tsv"

    out = session.get(results_url); out.raise_for_status()
    lines = [l for l in out.text.splitlines() if l]
    header = lines[0].split("\t")
    return [dict(zip(header, ln.split("\t"))) for ln in lines[1:]]


def batch_map_uniprot_ids(ids, max_batch_size=100000):
    """Split ids into chunks ≤100k, map each, and concat results."""
    all_maps = []
    for i in range(0, len(ids), max_batch_size):
        chunk = ids[i:i+max_batch_size]
        print(f"Mapping IDs {i+1}–{i+len(chunk)} of {len(ids)}…")
        all_maps.extend(map_uniprot_ids(chunk))

    print(f"{all_maps[:2]}")
    return all_maps

def build_uniprot_map(mappings):
    """
    Given mappings = [ {col1:val1, col2:val2, …}, … ],
    pick the first column as 'from' and the second as 'to'.
    """
    if not mappings:
        return {}

    cols = list(mappings[0].keys())
    # first column is input, second is output
    from_col = cols[0]
    to_col   = cols[1] if len(cols) > 1 else None
    if to_col is None:
        raise ValueError(f"No target column in mapping: {cols}")

    return { m[from_col]: m[to_col] for m in mappings }


# ── CAFA parsing & enrichment ──────────────────────────────────────────────────

def parse_cafa(fasta_path):
    """
    Read a single FASTA file at fasta_path.
    Returns a dict mapping each record’s ID to {"desc": description, "seq": sequence}.
    """
    seqs = {}
    with open(fasta_path, 'r') as f:
        # Read entire file and split on '>' (skip any leading blank)
        text = f.read()
    for entry in text.split('>')[1:]:
        lines = entry.strip().splitlines()
        if not lines:
            continue
        hdr = lines[0]
        seq = ''.join(lines[1:]).replace(' ', '')
        parts = hdr.split(maxsplit=1)
        pid = parts[0]
        desc = parts[1] if len(parts) > 1 else ''
        seqs[pid] = {'desc': desc, 'seq': seq}
    return seqs


def enrich_with_uniprot(seqs):
    entry_names = [r["desc"] for r in seqs.values() if r["desc"]]
    # first try the ID‐mapping API
    mappings = batch_map_uniprot_ids(entry_names)
    to_acc   = build_uniprot_map(mappings)

    # now batch‑lookup any that are still missing
    missing = [n for n in entry_names if n not in to_acc]
    print(f"{missing[:2]}")
    if missing:
        print(f"Batch‑searching {len(missing)} unmapped names…")
        fb = batch_lookup_accession_by_entry(missing, batch_size=500)
        to_acc.update(fb)

    # inject into seqs
    for pid, rec in seqs.items():
        rec["uniprot_id"] = to_acc.get(rec["desc"])
    return seqs



# ── PDB download & PKL saving ─────────────────────────────────────────────────

def download_pdbs(seqs, output_dir):
    """Download PDB files for each record using its UniProt accession."""
    # list of UniProt IDs 
    uniprot_ids = list(seqs.keys())
    download_pdbs_multithread(uniprot_ids, output_dir)

def get_esm2_embedding(seq):
    """Return ESM-2 embedding tensor for a sequence."""
    with torch.no_grad():
        return esm2_embed_sequence(seq, tokenizer, esm_model, device)

def _process_one(pid, rec, output_dir, local_pdb_dir):
    """
    Worker function: returns (pid, True, None) on success,
    or (pid, False, error_message) on skip/failure.
    """
    out_fp = os.path.join(output_dir, f"{pid}.pkl")
    if os.path.exists(out_fp):
        return pid, False, "already exists"

    pdb_fp = os.path.join(local_pdb_dir, f"{pid}.pdb")
    if not os.path.exists(pdb_fp):
        return pid, False, "no local PDB"

    # 1) fetch AF2 features
    pdb_info = fetch_alphafold_features(pid, local_pdb_dir=local_pdb_dir)
    if pdb_info is None:
        return pid, False, "no AF2 features"

    # 2) compute ESM2 embedding
    try:
        esm_emb = get_esm2_embedding(rec["seq"])
    except Exception as e:
        return pid, False, f"ESM2 error: {e}"

    # 3) assemble data dict
    data = {
        "esm":        esm_emb,
        "length":     len(rec["seq"]),
        "protein":    pid,
        "coords":     pdb_info["coords"],
        "pae":        pdb_info.get("pae"),
        "plddt":      pdb_info.get("plddt_scores"),
        "atom_names": pdb_info.get("atom_names"),
    }

    # 4) write to .pkl
    with open(out_fp, "wb") as f:
        pickle.dump(data, f)

    return pid, True, None

def save_to_pkl(seqs, output_dir, local_pdb_dir, max_workers=32):
    """
    Parallel PKL saver using threads. Shares one GPU model across threads.
    Handles Ctrl+C gracefully by cancelling pending tasks.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Filter to only those with a local PDB file
    valid = {
        pid: rec
        for pid, rec in seqs.items()
        if os.path.exists(os.path.join(local_pdb_dir, f"{pid}.pdb"))
    }
    skipped = set(seqs) - set(valid)
    if skipped:
        print(f"[PKL] Skipping {len(skipped)} entries without PDB")

    if not valid:
        print("[PKL] No sequences with PDBs; nothing to do.")
        return

    # 1) Submit all jobs
    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = {
        executor.submit(_process_one, pid, rec, output_dir, local_pdb_dir): pid
        for pid, rec in valid.items()
    }

    # 2) Collect results, with Ctrl+C handling
    try:
        for fut in tqdm(as_completed(futures),
                        total=len(futures),
                        desc="Saving PKLs"):
            pid = futures[fut]
            try:
                _, success, err = fut.result()
                if not success and err != "already exists":
                    print(f"[PKL] skipped {pid}: {err}")
            except Exception as e:
                print(f"[PKL] unhandled error for {pid}: {e}")
    except KeyboardInterrupt:
        # User hit Ctrl+C: cancel pending and exit
        print("\n[PKL] Interrupted by user, cancelling pending tasks...")
        executor.shutdown(wait=False, cancel_futures=True)
        return
    else:
        # Normal shutdown: wait for running tasks to finish
        executor.shutdown(wait=True)

def run_inference(model_ckpt, cafa_pkl_dir, go_vocab, atom_order, output_tsv, seqs, batch_size):
    go2idx = json.load(open(go_vocab))
    idx2go = [None] * len(go2idx)
    for go, i in go2idx.items():
        idx2go[i] = go

    state = torch.load(model_ckpt, map_location="cpu")
    model = ProteinFunctionModel(
        d_model=512, n_heads=8, dim_ff=1024,
        num_layers=6, num_go_terms=len(go2idx),
        use_struct_bias=True
    )
    sd = state.get("model_state", state)
    model.load_state_dict(sd)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    ds = ProteinDataset(cafa_pkl_dir, go_vocab, atom_order, seqs, eval_mode=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True,
                        collate_fn=protein_collate_fn)

    out_dir = os.path.dirname(output_tsv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_tsv, "w") as out:
        for batch in tqdm(loader, desc="CAFA inference"):
            # move tensors to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            with torch.no_grad():
                logits = model(
                    batch["seq_embed"], batch["attn_mask"],
                    batch.get("pae"), batch.get("plddt"),
                    batch["centroid"], batch["orientation_vectors"],
                    batch["side_chain_vectors"],
                    batch["euclidean_distances"],
                    batch["edge_vectors"],
                )
            probs = torch.sigmoid(logits).cpu().numpy()

            for i, pid in enumerate(batch["id"]):
                for j, score in enumerate(probs[i]):
                    out.write(f"{pid}\t{idx2go[j]}\t{score:.6f}\n")

    print("Inference done →", output_tsv)

def load_ground_truth_ids(gt_tsv_path):
    """
    Read ground_truth.tsv (protein_id <tab> GO_term) and
    return a set of unique protein_ids.
    """
    ids = set()
    with open(gt_tsv_path) as f:
        for line in f:
            if not line.strip(): 
                continue
            pid, _ = line.split("\t", 1)
            ids.add(pid)
    return ids

# ── Main CLI ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cafa_input",  required=True,
                   help="root dir of CAFA .tfa files")
    p.add_argument("--pdb_output",  required=True,
                   help="where to download PDBs")
    p.add_argument("--cafa_output", required=True,
                   help="where to save generated .pkl files")
    p.add_argument("--go_vocab",    required=True,
                   help="path to go_vocab_filtered.json")
    p.add_argument("--atom_order",  required=True,
                   help="path to global_atom_order.json")
    p.add_argument("--model_ckpt",  required=True,
                   help="your trained .pt model checkpoint")
    p.add_argument("--output_tsv",  required=True,
                   help="where to write CAFA prediction TSV")
    p.add_argument("--ground_truth", required=True,
                   help="ground_truth.tsv with protein_id <tab> GO_term")
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--subset_pid", type=str, default=None,
                   help="path to a file with PIDs to sample from CAFA input")
    
    args = p.parse_args()

    os.makedirs(args.pdb_output,  exist_ok=True)
    os.makedirs(args.cafa_output, exist_ok=True)

    print("Parsing CAFA targets…")
    seqs = parse_cafa(args.cafa_input)
    print(f"example: {list(seqs.items())[0]}")
    print(f"   → Found {len(seqs)} sequences")
    
    if args.subset_pid:
        pid_list_file = "/path/to/your/pid_list.txt"
        with open(pid_list_file, "r") as f:
            allowed_pids = { line.strip() for line in f if line.strip() }

        # ——— filter seqs to only those in allowed_pids ———
        original_count = len(seqs)
        seqs = { pid: seq for pid, seq in seqs.items() if pid in allowed_pids }
        print(f"Filtered out {original_count - len(seqs)} sequences; {len(seqs)} remain (only those in {pid_list_file!r})")

    gt_ids = load_ground_truth_ids(args.ground_truth)
    seqs = { pid: rec for pid, rec in seqs.items() if pid in gt_ids }
    print(f"Filtered to {len(seqs)} sequences in ground_truth.tsv")

    print("Mapping to UniProt accessions…")
    seqs = enrich_with_uniprot(seqs)
    print(f"example: {list(seqs.items())[0]}")

    sorted_ids = sorted(seqs.keys())
    
    if not os.listdir(args.pdb_output):
        print("Downloading PDB files…")
        download_pdbs(seqs, args.pdb_output)
    
    pdb_files = {
        os.path.splitext(fn)[0]
        for fn in os.listdir(args.pdb_output)
        if fn.lower().endswith(".pdb")
    }
    print(f"Found {len(pdb_files)} PDB files in {args.pdb_output}")

    seqs = {pid: rec for pid, rec in seqs.items() if pid in pdb_files}
    print(f"{len(seqs)} sequences have PDBs, proceeding to PKL generation")
    
    save_to_pkl(seqs, args.cafa_output, args.pdb_output)

    print("5) Running model inference…")
    run_inference(
        args.model_ckpt,
        args.cafa_output,
        args.go_vocab,
        args.atom_order,
        args.output_tsv,
        seqs,
        args.batch_size
    )

if __name__ == "__main__":
    main()