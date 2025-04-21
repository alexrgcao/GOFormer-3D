# train.py (modified for GO annotation with resume functionality)

import os
import random
import json
import numpy as np
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from accelerate import Accelerator
from dataset import ProteinDataset, protein_collate_fn, custom_collate_fn
from model import ProteinFunctionModel

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef, hamming_loss
import torch._dynamo as dynamo


def train_model(
    data_dir,        # Directory containing .pkl files
    output_dir,
    epochs=20,
    batch_size=8,
    lr=1e-4,
    use_struct_bias=True,
    resume_checkpoint=None,  # Path to checkpoint file to resume from
    quick_test=False,
    max_workers=8,
    accumulation_steps=1,
    go_vocab_file=None,
    atom_order_file=None,
    seed=42,
):
    """
    Trains a protein function prediction model using data from pkl files under data_dir,
    with optional checkpoint resumption.
    """
    accelerator = Accelerator()

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    device = accelerator.device
    
    go_vocab_file = "/scratch/rgc4/go_vocab_filtered.json"
    atom_order_file = "/scratch/rgc4/global_atom_order.json"
    dataset_full = ProteinDataset(data_dir, go_vocab_file, atom_order_file)
    num_go_terms = len(dataset_full.go_vocab)
    total_size = len(dataset_full)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset_full, [train_size, val_size, test_size])
    print(f"Dataset sizes: Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=max_workers,
        prefetch_factor=3,
        pin_memory=True,
        collate_fn=protein_collate_fn,
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=max_workers,
        prefetch_factor=3,
        pin_memory=True,
        collate_fn=protein_collate_fn,
    )

    # Initialize model
    model = ProteinFunctionModel(
        d_model=512,
        n_heads=8,
        dim_ff=1024,
        num_layers=6,
        num_go_terms=num_go_terms,
        use_struct_bias=use_struct_bias
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print("Input token embedding dimension (d_emb): 2560")
    print(f"Projected model hidden dimension (d_model): {model.d_model}")
    print(f"Output dimension (number of GO terms): {num_go_terms}")

    
    loss_go = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    accumulation_steps = 36

    model, optimizer, val_loader, test_loader = accelerator.prepare(
        model, optimizer, val_loader, test_loader
    )

    if hasattr(accelerator.state, "deepspeed_plugin"):
        scaler = None

    start_epoch = 1
    if resume_checkpoint is not None and os.path.isfile(resume_checkpoint):
        checkpoint = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        optimizer_state = checkpoint['optimizer_state']

        if "param_groups" not in optimizer_state:
            current_opt_state = optimizer.state_dict()
            optimizer_state["param_groups"] = current_opt_state["param_groups"]
            print("Inserted missing 'param_groups' into optimizer state.")
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded epoch {start_epoch}")
        
        print(f"Resumed model from checkpoint: {resume_checkpoint}, starting at epoch {start_epoch}")

    

    epoch_metrics = []
    metrics_file = os.path.join(output_dir, "evaluation_metrics.csv")
    header_written = os.path.exists(metrics_file)
    best_val_loss = float("inf")
    patience = 3
    counter = 0
    threshold = 0.5

    for epoch in range(start_epoch, epochs + 1):

        generator = torch.Generator()
        generator.manual_seed(42 + epoch)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=max_workers,
            prefetch_factor=9,
            collate_fn=protein_collate_fn,
            pin_memory=True,
            generator=generator
        )
        train_loader = accelerator.prepare(train_loader)

        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} - Training", unit="batch")
        for i, batch in enumerate(train_bar):
            seq_embed         = batch["seq_embed"]
            label             = batch["label"]
            attn_mask         = batch["attn_mask"]
            centroid          = batch["centroid"]
            euclidean_distances = batch["euclidean_distances"]
            edge_vectors      = batch["edge_vectors"]
            orientation_vectors = batch["orientation_vectors"]
            side_chain_vectors   = batch["side_chain_vectors"]
            pae   = batch["pae"] if batch["pae"] is not None else None
            plddt = batch["plddt"] if batch["plddt"] is not None else None

            if scaler is not None:
                with torch.amp.autocast():
                    go_logits = model(
                        seq_embed,
                        attn_mask,
                        pae,
                        plddt,
                        centroid,
                        orientation_vectors,
                        side_chain_vectors,
                        euclidean_distances,
                        edge_vectors,
                    )
            else:
                model_dtype = next(model.parameters()).dtype
                seq_embed = seq_embed.to(model_dtype)
                if pae is not None:
                    pae = pae.to(model_dtype)
                if plddt is not None:
                    plddt = plddt.to(model_dtype)
                centroid = centroid.to(model_dtype)
                orientation_vectors = orientation_vectors.to(model_dtype)
                side_chain_vectors = side_chain_vectors.to(model_dtype)
                euclidean_distances = euclidean_distances.to(model_dtype)
                edge_vectors = edge_vectors.to(model_dtype)
                go_logits = model(
                    seq_embed,
                    attn_mask,
                    pae,
                    plddt,
                    centroid,
                    orientation_vectors,
                    side_chain_vectors,
                    euclidean_distances,
                    edge_vectors,
                )

            loss = loss_go(go_logits, label)
            loss = loss / accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                accelerator.backward(loss)
            total_loss += loss.item() * accumulation_steps * seq_embed.size(0)

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            train_bar.set_postfix(loss=f"{loss.item() * accumulation_steps:.4f}")
            if quick_test and i >= 5:
                print("Quick test mode: breaking training loop after 5 batches.")
                break

        scheduler.step()
        avg_train_loss = total_loss / len(train_dataset)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}/{epochs} - Training Loss: {avg_train_loss:.4f}, LR: {current_lr:.2e}")

        # Validation Loop
        model.eval()
        val_loss = 0.0
        preds_list = []
        probs_list = []
        labels_list = []
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} - Validation", unit="batch")
        with torch.no_grad():
            for i, batch in enumerate(val_bar):
                seq_embed = batch["seq_embed"]
                label     = batch["label"]
                attn_mask = batch["attn_mask"]
                centroid  = batch["centroid"]
                euclidean_distances = batch["euclidean_distances"]
                edge_vectors = batch["edge_vectors"]
                orientation_vectors = batch["orientation_vectors"]
                side_chain_vectors  = batch["side_chain_vectors"]
                pae   = batch["pae"] if batch["pae"] is not None else None
                plddt = batch["plddt"] if batch["plddt"] is not None else None

                if scaler is not None:
                    with torch.amp.autocast():
                        go_logits = model(
                            seq_embed,
                            attn_mask,
                            pae,
                            plddt,
                            centroid,
                            orientation_vectors,
                            side_chain_vectors,
                            euclidean_distances,
                            edge_vectors,
                        )
                else:
                    model_dtype = next(model.parameters()).dtype
                    seq_embed = seq_embed.to(model_dtype)
                    if pae is not None:
                        pae = pae.to(model_dtype)
                    if plddt is not None:
                        plddt = plddt.to(model_dtype)
                    centroid = centroid.to(model_dtype)
                    orientation_vectors = orientation_vectors.to(model_dtype)
                    side_chain_vectors = side_chain_vectors.to(model_dtype)
                    euclidean_distances = euclidean_distances.to(model_dtype)
                    edge_vectors = edge_vectors.to(model_dtype)
                    go_logits = model(
                        seq_embed,
                        attn_mask,
                        pae,
                        plddt,
                        centroid,
                        orientation_vectors,
                        side_chain_vectors,
                        euclidean_distances,
                        edge_vectors,
                    )

                loss = loss_go(go_logits, label)
                val_loss += loss.item() * seq_embed.size(0)

                probs = torch.sigmoid(go_logits)
                preds = (probs >= threshold).int()
                preds_list.append(preds.cpu().numpy())
                probs_list.append(probs.cpu().numpy())
                labels_list.append(label.cpu().numpy())

                if quick_test and i >= 5:
                    print("Quick test mode: breaking validation loop after 5 batches.")
                    break

        avg_val_loss = val_loss / len(val_dataset)
        all_probs = np.concatenate(probs_list, axis=0)
        all_preds = np.concatenate(preds_list, axis=0)
        all_labels = np.concatenate(labels_list, axis=0)

        acc_score = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='micro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='micro', zero_division=0)
        f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        try:
            roc_auc = roc_auc_score(all_labels, all_probs, average='micro')
        except ValueError:
            roc_auc = float('nan')
        auprc = average_precision_score(all_labels, all_preds, average='micro')
        mcc = matthews_corrcoef(all_labels.flatten(), all_preds.flatten())
        ham_loss = hamming_loss(all_labels, all_preds)

        print(f"Epoch {epoch}/{epochs} - Validation Loss: {avg_val_loss:.4f}")
        print(f"Epoch {epoch} - Accuracy: {acc_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(f"Epoch {epoch} - Micro-F1: {f1_micro:.4f}, Macro-F1: {f1_macro:.4f}")
        print(f"Epoch {epoch} - ROC-AUC: {roc_auc:.4f}, AUPRC: {auprc:.4f}")
        print(f"Epoch {epoch} - MCC: {mcc:.4f}, Hamming Loss: {ham_loss:.4f}")

        current_epoch_metrics = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'accuracy': acc_score,
            'precision': precision,
            'recall': recall,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'roc_auc': roc_auc,
            'auprc': auprc,
            'mcc': mcc,
            'hamming_loss': ham_loss
        }
        epoch_metrics.append(current_epoch_metrics)

        with open(metrics_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=current_epoch_metrics.keys())
            if not header_written:
                writer.writeheader()
                header_written = True
            writer.writerow(current_epoch_metrics)
        print(f"Epoch {epoch}: Evaluation metrics saved to {metrics_file}")

        ckpt = {
            'epoch': epoch,
            'model_state': accelerator.unwrap_model(model).state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict()
        }

        if quick_test:
            ckpt_dir = os.path.join(output_dir, f"quick_test_model.pt")
            torch.save(ckpt, ckpt_dir)
            
            print("Quick test mode: breaking after one epoch.")
            break
        else:
            ckpt_dir = os.path.join(output_dir, f"model_epoch{epoch}.pt")
            torch.save(ckpt, ckpt_dir)

    final_path = os.path.join(output_dir, "model_final.pt")
    torch.save(accelerator.unwrap_model(model).state_dict(), final_path)
    print("Training complete. Model saved to", final_path)

    threshold = 0.5

    # Testing loop
    model.eval()
    test_loss = 0.0
    preds_list = []
    probs_list = []
    labels_list = []
    test_bar = tqdm(test_loader, desc="Testing", unit="batch")
    with torch.no_grad():
        for i, batch in enumerate(test_bar):
            seq_embed = batch["seq_embed"].to(accelerator.device)
            label     = batch["label"].to(accelerator.device)
            attn_mask = batch["attn_mask"].to(accelerator.device)
            centroid  = batch["centroid"].to(accelerator.device)
            euclidean_distances = batch["euclidean_distances"].to(accelerator.device)
            edge_vectors = batch["edge_vectors"].to(accelerator.device)
            orientation_vectors = batch["orientation_vectors"].to(accelerator.device)
            side_chain_vectors  = batch["side_chain_vectors"].to(accelerator.device)
            pae   = batch["pae"].to(accelerator.device) if batch["pae"] is not None else None
            plddt = batch["plddt"].to(accelerator.device) if batch["plddt"] is not None else None

            if scaler is not None:
                with torch.amp.autocast():
                    go_logits = model(
                        seq_embed,
                        attn_mask,
                        pae,
                        plddt,
                        centroid,
                        orientation_vectors,
                        side_chain_vectors,
                        euclidean_distances,
                        edge_vectors,
                    )
            else:
                model_dtype = next(model.parameters()).dtype
                seq_embed = seq_embed.to(model_dtype)
                if pae is not None:
                    pae = pae.to(model_dtype)
                if plddt is not None:
                    plddt = plddt.to(model_dtype)
                centroid = centroid.to(model_dtype)
                orientation_vectors = orientation_vectors.to(model_dtype)
                side_chain_vectors = side_chain_vectors.to(model_dtype)
                euclidean_distances = euclidean_distances.to(model_dtype)
                edge_vectors = edge_vectors.to(model_dtype)
                go_logits = model(
                    seq_embed,
                    attn_mask,
                    pae,
                    plddt,
                    centroid,
                    orientation_vectors,
                    side_chain_vectors,
                    euclidean_distances,
                    edge_vectors,
                )

            loss = loss_go(go_logits, label)
            test_loss += loss.item() * seq_embed.size(0)

            probs = torch.sigmoid(go_logits)
            preds = (probs >= threshold).int()
            preds_list.append(preds.cpu().numpy())
            probs_list.append(probs.cpu().numpy())
            labels_list.append(label.cpu().numpy())

            if quick_test and i >= 5:
                print("Quick test mode: breaking training loop after 5 batches.")
                break

    avg_test_loss = test_loss / len(test_dataset)
    print(f"Test Loss: {avg_test_loss:.4f}")

    all_preds = np.concatenate(preds_list, axis=0)
    all_probs = np.concatenate(probs_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='micro', zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    try:
        roc_auc = roc_auc_score(all_labels, all_probs, average='micro')
    except ValueError:
        roc_auc = float('nan')
    auprc = average_precision_score(all_labels, all_preds, average='micro')
    mcc = matthews_corrcoef(all_labels.flatten(), all_preds.flatten())
    ham_loss = hamming_loss(all_labels, all_preds)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"Test Micro-F1: {f1_micro:.4f}, Macro-F1: {f1_macro:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}, AUPRC: {auprc:.4f}")
    print(f"Test MCC: {mcc:.4f}, Hamming Loss: {ham_loss:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train protein function prediction model")
    parser.add_argument("--data_dir", required=True,
                        help="Root directory containing subfolders with .pkl files (one per protein)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--max_workers", type=int, default=8,
                        help="Number of workers for DataLoader")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--accumulation_steps", type=int, default=36,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--no_struct_bias", action="store_true",
                        help="Disable structural bias in attention")
    parser.add_argument("--go_vocab_file", required=True, type=str, default=None,
                        help="Path to the GO vocabulary file")
    parser.add_argument("--atom_order_file", required=True, type=str, default=None,
                        help="Path to the atom order file")
    parser.add_argument("--output_dir", default="outputs",
                        help="Directory to save model checkpoints and metrics")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to a checkpoint to resume training from")
    parser.add_argument("--quick_test", action="store_true",
                        help="Run a quick test on a small subset (e.g. process only 5 batches per loop)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_struct_bias=(not args.no_struct_bias),
        resume_checkpoint=args.resume_checkpoint,
        quick_test=args.quick_test,
        max_workers=args.max_workers,
        accumulation_steps=args.accumulation_steps,
        go_vocab_file=args.go_vocab_file,
        atom_order_file=args.atom_order_file,
        seed=args.seed,
    )