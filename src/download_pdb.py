#!/usr/bin/env python3
import os
import pickle
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
import argparse

def extract_uniprot_ids(input_root):
    """Walk through all .pkl files and collect unique UniProt IDs."""
    uniprot_ids = set()
    for root_dir, _, files in os.walk(input_root):
        for file in files:
            if file.endswith(".pkl"):
                pkl_path = os.path.join(root_dir, file)
                try:
                    with open(pkl_path, "rb") as f:
                        data = pickle.load(f)
                    uniprot_id = data.get("protein")
                    if not uniprot_id:
                        uniprot_id = os.path.splitext(file)[0]
                    uniprot_ids.add(uniprot_id)
                except Exception as e:
                    print(f"Error loading {pkl_path}: {e}")
    return list(uniprot_ids)

def save_uniprot_ids(uniprot_ids, filename):
    with open(filename, "w") as f:
        for uid in uniprot_ids:
            f.write(uid + "\n")

def fetch_pdb_content(uniprot_id, timeout=30, retries=30, backoff=2):
    """
    Fetch PDB with retry + exponential backoff.
    """
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(api_url, timeout=timeout)
            response.raise_for_status()
            data = response.json()

            if not data:
                print(f"No structure available for {uniprot_id}.")
                return None

            pdb_url = data[0].get("pdbUrl")
            if not pdb_url:
                print(f"No PDB URL found for {uniprot_id}.")
                return None

            pdb_response = requests.get(pdb_url, timeout=timeout)
            pdb_response.raise_for_status()
            return pdb_response.text

        except requests.exceptions.RequestException as e:
            print(f"[{uniprot_id}] Attempt {attempt} failed: {e}")
            if attempt < retries:
                sleep_time = backoff ** attempt + random.uniform(0, 1)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"Failed to fetch PDB for {uniprot_id} after {retries} retries.")
                return None

def download_and_save_pdb(uniprot_id, output_dir):
    """Fetch and save PDB file for one UniProt ID, skipping if already exists."""
    output_path = os.path.join(output_dir, f"{uniprot_id}.pdb")
    if os.path.exists(output_path):
        #print(f"Skipping {uniprot_id}, PDB already exists.")
        return

    pdb_content = fetch_pdb_content(uniprot_id)
    if pdb_content:
        try:
            with open(output_path, "w") as f:
                f.write(pdb_content)
            # print(f"Saved {uniprot_id}")
        except Exception as e:
            print(f"Error saving {uniprot_id}: {e}")
    else:
        print(f"Failed to fetch PDB for {uniprot_id}.")


def download_pdbs_multithread(uniprot_ids, output_dir, max_workers=8):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_and_save_pdb, uid, output_dir): uid for uid in uniprot_ids}
        for future in as_completed(futures):
            uid = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Unhandled error for {uid}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download PDB files for UniProt IDs.")
    parser.add_argument("--input_root", required=True, help="Root directory containing input pickle files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save downloaded PDB files.")
    parser.add_argument("--uniprot_list_file", required=True, help="File to save UniProt IDs.")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of threads for downloading.")
    
    args = parser.parse_args()

    ids = extract_uniprot_ids(args.input_root)
    print(f"Found {len(ids)} unique UniProt IDs.")
    
    save_uniprot_ids(ids, args.uniprot_list_file)
    print(f"Saved UniProt IDs to {args.uniprot_list_file}")

    download_pdbs_multithread(ids, args.output_dir, args.max_workers)

if __name__ == "__main__":
    main()

    
