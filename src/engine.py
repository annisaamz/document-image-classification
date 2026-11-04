"""Loop training, validasi, checkpointing, dan prediksi batch untuk model klasifikasi."""

import os

import torch
import torch.nn as nn
from tqdm.auto import tqdm


def run_epoch(model, loader, criterion, device, use_amp, scaler, optimizer=None, grad_clip=1.0):
    """Menjalankan satu epoch training (jika optimizer diberikan) atau validasi.

    Args:
        model (nn.Module): Model yang dijalankan.
        loader (DataLoader): Sumber batch berisi pasangan (x, y).
        criterion (nn.Module): Fungsi loss.
        device (torch.device): Device komputasi.
        use_amp (bool): Mengaktifkan mixed precision (FP16) jika True.
        scaler (torch.amp.GradScaler): Scaler untuk mixed precision.
        optimizer (torch.optim.Optimizer | None): Jika None, berjalan dalam mode evaluasi.
        grad_clip (float): Batas maksimum norma gradien.

    Returns:
        tuple[float, float]: Rata-rata loss dan akurasi pada epoch tersebut.
    """
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for x_b, y_b in tqdm(loader, leave=False):
            x_b = x_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)  # reset gradien sebelum backward

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x_b)
                loss = criterion(logits, y_b)

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # unscale agar clipping bekerja pada gradien asli
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * x_b.size(0)
            correct += (logits.argmax(1) == y_b).sum().item()
            total += x_b.size(0)

    return total_loss / total, correct / total


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler,
                device, config, ckpt_dir, resume=False):
    """Melatih model dengan checkpoint terbaik berdasarkan validation loss dan early stopping.

    Pada setiap epoch disimpan last checkpoint (untuk melanjutkan training). Checkpoint
    terbaik disimpan hanya ketika validation loss membaik. Training berhenti lebih awal
    jika validation loss tidak membaik selama config["patience"] epoch berturut-turut.

    Args:
        model (nn.Module): Model yang dilatih.
        train_loader (DataLoader): Loader data training.
        val_loader (DataLoader): Loader data validasi.
        criterion (nn.Module): Fungsi loss.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler: Learning rate scheduler yang di-step per epoch.
        scaler (torch.amp.GradScaler): Scaler untuk mixed precision.
        device (torch.device): Device komputasi.
        config (dict): Konfigurasi dari src.config.CONFIG.
        ckpt_dir (str): Folder penyimpanan checkpoint.
        resume (bool): Jika True dan last checkpoint tersedia, lanjutkan training dari sana.

    Returns:
        dict: Berisi history (train_loss, val_loss, train_acc, val_acc), best_epoch,
            best_val_loss, dan best_path.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    name = config["model_name"]
    best_path = os.path.join(ckpt_dir, f"best_{name}.pt")
    last_path = os.path.join(ckpt_dir, f"last_{name}.pt")
    use_amp = device.type == "cuda" and config["amp"]

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss, best_epoch, wait, start_epoch = float("inf"), None, 0, 1

    if resume and os.path.exists(last_path):
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        history = state["history"]
        best_val_loss, best_epoch, wait = state["best_val_loss"], state["best_epoch"], state["wait"]
        start_epoch = state["epoch"] + 1
        print(f"Melanjutkan dari epoch {start_epoch}")
    else:
        print("Mulai training dari epoch 1")

    for epoch in range(start_epoch, config["epochs"] + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, use_amp, scaler,
                                    optimizer, config["grad_clip"])
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device, use_amp, scaler)
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()  # cosine annealing dijalankan per epoch

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        print(f"Epoch {epoch:02d} | lr {lr_now:.2e} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_loss < best_val_loss:
            best_val_loss, best_epoch, wait = va_loss, epoch, 0
            torch.save({"model_name": name, "model_state": model.state_dict(),
                        "epoch": epoch, "val_loss": va_loss, "config": config}, best_path)
        else:
            wait += 1

        torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
                    "history": history, "best_val_loss": best_val_loss, "best_epoch": best_epoch,
                    "wait": wait, "epoch": epoch}, last_path)

        if wait >= config["patience"]:
            print(f"Early stopping pada epoch {epoch}. Epoch terbaik: {best_epoch}")
            break

    print("Checkpoint terbaik:", best_path, "| epoch", best_epoch)
    return {"history": history, "best_epoch": best_epoch,
            "best_val_loss": best_val_loss, "best_path": best_path}


def load_checkpoint(model, path, device):
    """Memuat bobot dari checkpoint ke model dan mengaturnya ke mode evaluasi.

    Args:
        model (nn.Module): Model dengan arsitektur yang sama dengan checkpoint.
        path (str): Lokasi berkas checkpoint (.pt).
        device (torch.device): Device tempat bobot dimuat.

    Returns:
        dict: Isi checkpoint lengkap, misalnya epoch dan val_loss.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return ckpt


def predict_batches(model, loader, device):
    """Menghasilkan prediksi kelas untuk seluruh sampel pada sebuah loader.

    Loader boleh berisi label (untuk validasi) atau id (untuk test). Keduanya dikembalikan
    apa adanya agar urutannya bisa dicocokkan.

    Args:
        model (nn.Module): Model terlatih.
        loader (DataLoader): Loader berisi pasangan (x, target).
        device (torch.device): Device komputasi.

    Returns:
        tuple[list, list]: Daftar target (label atau id) dan daftar kelas hasil prediksi.
    """
    model.eval()
    targets, preds = [], []
    with torch.no_grad():
        for x_b, t_b in loader:
            logits = model(x_b.to(device))
            preds += logits.argmax(1).cpu().tolist()
            targets += t_b.tolist() if torch.is_tensor(t_b) else list(t_b)
    return targets, preds
