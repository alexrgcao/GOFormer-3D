import obonet
import requests
import json
from collections import Counter
import torch
from transformers import BertTokenizerFast
from pathlib import Path
import os
import pickle
import argparse

GO_OBO_URL = "https://current.geneontology.org/ontology/go-basic.obo"
MIN_FREQUENCY = 50  # Only keep GO terms appearing in >=50 proteins

def download_and_parse_go_obo(url=GO_OBO_URL):
    response = requests.get(url)
    obo_content = response.text

    with open("go-basic.obo", "w") as f:
        f.write(obo_content)
    graph = obonet.read_obo("go-basic.obo")
    return graph

def load_protein_annotations(root):
    protein_annotations = []
    subdirs = [os.path.join(root, d) for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))]

    for folder in subdirs:
        folder_name = os.path.basename(folder)
        for dirpath, _, files in os.walk(folder):
            
            for file in files:
                if file.endswith('.pkl'):
                    input_file_path = os.path.join(dirpath, file)
                    with open(input_file_path, "rb") as f:
                        data = pickle.load(f)
                    go_terms = data['true_go']
                    protein_annotations.append(go_terms)
                    
    print(f"Total proteins with annotations: {len(protein_annotations)}")
    return protein_annotations

def filter_go(protein_annotations, min_frequency=50):
    go_counter = Counter(go for annot in protein_annotations for go in annot)

    filtered_go_terms = [go for go, count in go_counter.items() if count >= min_frequency]

    if not filtered_go_terms:
        filtered_go_terms = list(go_counter.keys())
    print("Number of GO terms after filtering:", len(filtered_go_terms))
    return filtered_go_terms

def save_vocab(vocab, output_json_path, output_txt_path):
    
    tok_to_idx = {tok: i for i, tok in enumerate(vocab)}
    vocab_size = len(vocab)
    print("Final vocabulary size:", vocab_size)
    print("First 10 vocabulary entries:", list(tok_to_idx.items())[:10])

    with open(VOCAB_OUTPUT_FILE_JSON, "w") as f:
        json.dump(tok_to_idx, f, indent=2)
    print(f"Vocabulary (JSON) saved to {VOCAB_OUTPUT_FILE_JSON}")


def main():
    parser = argparse.ArgumentParser(description="Build vocabulary from GO terms.")
    parser.add_argument("--input_root", required=True, help="Root directory containing input protein annotations.")
    parser.add_argument("--obo_url", default=GO_OBO_URL, help="URL of the GO OBO file")
    parser.add_argument("--min_frequency", type=int, default=MIN_FREQUENCY, help="Minimum frequency for GO terms")
    parser.add_argument("--vocab_output_json_path", help="Output file for vocabulary (JSON)")
    parser.add_argument("--tokenizer_save_dir", help="Directory to save the Hugging Face tokenizer")
    args = parser.parse_args()

    graph = download_and_parse_go_obo()
    all_go_terms = list(graph.nodes())
    print(f"Total GO terms in ontology: {len(all_go_terms)}")

    print("Loading protein annotations...")
    protein_annotations = load_protein_annotations(args.input_root)

    print("Filtering GO terms based on frequency...")
    filtered_go_terms = filter_go(protein_annotations, args.min_frequency)

    print("Building vocabulary...")
    save_vocab(filtered_go_terms, args.vocab_output_json_path)

if __name__ == "__main__":
    main()


