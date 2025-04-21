import os
import pickle
import argparse

def update_pickle_file(pkl_path, keys_to_clear):
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading {pkl_path}: {e}")
        return None

    if isinstance(data, dict):
        for key in keys_to_clear:
            if key in data:
                data[key] = None
            else:
                print(f"Key '{key}' not found in data.")
    elif isinstance(data, list):
        for idx, entry in enumerate(data):
            if isinstance(entry, dict):
                for key in keys_to_clear:
                    if key in entry:
                        entry[key] = None
                    else:
                        print(f"Entry {idx}: key '{key}' not found.")
            else:
                print(f"Entry {idx} is not a dictionary (type {type(entry).__name__}). Skipping update for keys.")
    else:
        print("Data is neither a dictionary nor a list. No updates performed.")
    
    return data

def process_pickles(input_root, output_root, keys_to_clear):
    for dirpath, _, files in os.walk(input_root):
        relative_path = os.path.relpath(dirpath, input_root)
        dest_dir = os.path.join(output_root, relative_path)
        os.makedirs(dest_dir, exist_ok=True)

        for file in files:
            if file.endswith('.pkl'):
                input_file_path = os.path.join(dirpath, file)
                updated_data = update_pickle_file(input_file_path, keys_to_clear)
                if updated_data is not None:
                    output_file_path = os.path.join(dest_dir, file)
                    with open(output_file_path, "wb") as f:
                        pickle.dump(updated_data, f)

def main():
    parser = argparse.ArgumentParser(description="Process and update pickle files.")
    parser.add_argument("--input_root", required=True, help="Root directory containing input pickle files.")
    parser.add_argument("--output_root", required=True, help="Root directory for output pickle files.")
    
    args = parser.parse_args()
    keys_to_clear = ["coords", "esm", "plddt", "label"]

    subdirs = [os.path.join(args.input_root, d) for d in os.listdir(args.input_root)
               if os.path.isdir(os.path.join(args.input_root, d))]
    
    for folder in subdirs:
        folder_name = os.path.basename(folder)
        new_output_root = os.path.join(args.output_root, folder_name)
        process_pickles(folder, new_output_root, keys_to_clear)

if __name__ == "__main__":
    main()
    
