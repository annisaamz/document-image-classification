"""Fungsi pendukung umum: seed acak, pemilihan device, hashing berkas, dan cek ruang disk."""

import hashlib
import os
import random
import shutil

import numpy as np
import torch


def setup_environment(seed=42):
    """Mengatur seed acak dan opsi cuDNN agar eksperimen dapat diulang.

    Args:
        seed (int): Nilai seed untuk modul random, NumPy, dan PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def get_device():
    """Mengembalikan device CUDA jika tersedia, jika tidak maka CPU.

    Returns:
        torch.device: Device yang dipakai untuk training dan inferensi.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def md5_of(path, chunk_size=1 << 20):
    """Menghitung MD5 dari seluruh byte sebuah berkas.

    Args:
        path (str): Lokasi berkas.
        chunk_size (int): Ukuran potongan baca dalam byte.

    Returns:
        str: Digest MD5 dalam format heksadesimal.
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def has_enough_space(folder, need_gb):
    """Memeriksa apakah ruang bebas di folder cukup untuk menyimpan cache.

    Ruang bebas harus minimal 1,2 kali perkiraan kebutuhan sebagai margin.

    Args:
        folder (str): Folder tujuan penyimpanan cache.
        need_gb (float): Perkiraan kebutuhan ruang dalam GB.

    Returns:
        bool: True jika ruang bebas mencukupi.
    """
    os.makedirs(folder, exist_ok=True)
    free_gb = shutil.disk_usage(folder).free / 1e9
    print(f"Ruang bebas di {folder}: {free_gb:.1f} GB | perkiraan kebutuhan: {need_gb:.1f} GB")
    return free_gb > need_gb * 1.2
