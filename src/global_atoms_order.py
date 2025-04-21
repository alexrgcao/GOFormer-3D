#!/usr/bin/env python
import os
import json
from Bio.PDB import PDBParser
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

def extract_atoms_from_pdb(pdb_path):
    """
    Parse a single PDB file and return a set of atom names found.
    """
    atom_names = set()
    parser = PDBParser(QUIET=True)
    try:
        with open(pdb_path, "r") as f:
            pdb_content = f.read()
        structure = parser.get_structure(os.path.basename(pdb_path), StringIO(pdb_content))
        model = next(structure.get_models())
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom_names.add(atom.get_name())
    except Exception as e:
        print(f"Error processing {pdb_path}: {e}")
    return atom_names

def collect_global_atom_order(pdb_root, output_json, num_threads=8):
    """
    Walks through all PDB files under pdb_root, parses each PDB in parallel to extract atom names,
    unions them, sorts the list, and saves it as a JSON file.
    """
    global_atoms = set()
    pdb_files = []
    for root_dir, _, files in os.walk(pdb_root):
        for file in files:
            if file.lower().endswith(".pdb"):
                pdb_files.append(os.path.join(root_dir, file))
    print(f"Found {len(pdb_files)} PDB files to process.")

    # Use multi-threading to process files concurrently.
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(extract_atoms_from_pdb, pdb) for pdb in pdb_files]
        for future in as_completed(futures):
            atoms = future.result()
            global_atoms.update(atoms)
            
    global_atom_order = sorted(list(global_atoms))
    with open(output_json, "w") as out_f:
        json.dump(global_atom_order, out_f, indent=2)
    print(f"Global atom order saved to {output_json}")

def main():
    parser = argparse.ArgumentParser(description="Collect global atom order from PDB files.")
    parser.add_argument("--pdb_root", default=PDB_ROOT, help="Root directory containing PDB files.")
    parser.add_argument("--output_json", default=OUTPUT_JSON, help="Output JSON file for global atom order.")
    parser.add_argument("--num_threads", type=int, default=8, help="Number of threads for processing.")
    args = parser.parse_args()

    collect_global_atom_order(args.pdb_root, args.output_json, args.num_threads)

if __name__ == "__main__":
    main()
