#!/usr/bin/env python3
import argparse
import random
import requests
import gzip
from io import BufferedReader
import os

def parse_cafa(fasta_path):
    """
    Read a single FASTA file at fasta_path.
    Returns a dict mapping each record’s ID to {"desc": description, "seq": sequence}.
    """
    seqs = {}
    with open(fasta_path, 'r') as f:
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

def filter_gaf(gaf_url, test_ids, output_tsv):
    """
    Stream the gzipped GAF, filter to only test_ids, and write out
    protein_id <tab> GO_term per line. Stops early once all test_ids have been found.
    Returns the set of protein_ids that were actually found.
    """
    test_set = set(test_ids)
    found    = set()

    resp = requests.get(gaf_url, stream=True)
    resp.raise_for_status()
    gz     = gzip.GzipFile(fileobj=resp.raw)
    reader = BufferedReader(gz)

    with open(output_tsv, "w") as out:
        for raw in reader:
            if raw.startswith(b"!"):
                continue
            cols = raw.decode("utf-8").rstrip("\n").split("\t")

            if cols[0] != "UniProtKB" or "NOT" in cols[3]:
                continue
            prot_id = cols[1]
            go_id   = cols[4]
            if prot_id in test_set:
                out.write(f"{prot_id}\t{go_id}\n")
                found.add(prot_id)
                if found >= test_set:
                    break

    return found

def main():
    p = argparse.ArgumentParser(
        description="Build CAFA ground-truth from UniProt-GOA GAF, but only on a random 10% subset."
    )
    p.add_argument(
        "--test_ids", required=True,
        help="Path to your full FASTA of targets (one record per protein)"
    )
    p.add_argument(
        "--gaf_url",
        default="https://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/goa_uniprot_all.gaf.gz",
        help="URL of the UniProt-GOA GAF file (gzipped)"
    )
    p.add_argument(
        "--output", required=True,
        help="Path to write filtered ground_truth.tsv"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    p.add_argument(
        "--subset", type=float, default=0.1,
        help="Fraction to sample (default: 0.1 = 10%%)"
    )
    args = p.parse_args()

    print("Loading all IDs from FASTA…")
    seqs = parse_cafa(args.test_ids)
    all_ids = sorted(seqs.keys())
    total = len(all_ids)
    print(f"  → {total} proteins loaded")

    random.seed(args.seed)

    k = max(1, int(total * args.subset))
    sampled_ids = sorted(random.sample(all_ids, k))
    seqs = {pid: seqs[pid] for pid in sampled_ids}
    output_path = os.path.join(args.output, "sampled_ids.txt")
    with open(output_path, "w") as out:
        for pid in sampled_ids:
            out.write(pid + "\n")

    print(f"Wrote {len(sampled_ids)} PIDs to {output_path}")
    print(f"Sampling {k} proteins ({args.subset*100:.0f}% of {total}) using seed={args.seed}")

    print("Streaming and filtering GAF for sampled IDs…")
    gt_output_path = os.path.join(args.output, "ground_truth.tsv")
    found = filter_gaf(args.gaf_url, sampled_ids, gt_output_path)
    print(f"Ground truth written to {gt_output_path}")
    print(f"\nFound {len(found)}/{k} unique UniProt IDs in GAF")

if __name__ == "__main__":
    main()
