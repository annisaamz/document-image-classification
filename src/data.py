"""Pemuatan data, pembersihan, pembagian train-validasi, augmentasi, cache, dan Dataset PyTorch."""

import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from src.utils import has_enough_space, md5_of


def load_competition_tables(data_dir):
    """Membaca tiga tabel kompetisi dan menambahkan kolom path lengkap ke setiap citra.

    Args:
        data_dir (str): Folder hasil ekstraksi arsip kompetisi.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train_df, test_df, dan sample_sub.
    """
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

    # Struktur arsip Kaggle: data/train/train/<kelas>/*.tif dan data/test/test/*.tif
    train_df["path"] = [os.path.join(data_dir, "train", f) for f in train_df["filename"]]
    test_df["path"] = [os.path.join(data_dir, "test", f) for f in test_df["filename"]]
    return train_df, test_df, sample_sub


def read_image_metadata(paths):
    """Membaca metadata setiap citra tanpa memuat seluruh piksel ke memori.

    Args:
        paths (list[str]): Daftar lokasi citra.

    Returns:
        pd.DataFrame: Kolom width, height, mode, format, n_frames, dan size_kb.
            Citra yang gagal dibuka diberi format 'ERROR'.
    """
    rows = []
    for p in tqdm(paths):
        try:
            with Image.open(p) as img:
                rows.append([img.width, img.height, img.mode, img.format,
                             getattr(img, "n_frames", 1), os.path.getsize(p) / 1024])
        except Exception:
            rows.append([None, None, None, "ERROR", None, None])
    return pd.DataFrame(rows, columns=["width", "height", "mode", "format", "n_frames", "size_kb"])


def remove_unreadable(df):
    """Membuang baris yang citranya gagal dibaca (format bernilai 'ERROR').

    Args:
        df (pd.DataFrame): DataFrame yang memuat kolom format.

    Returns:
        pd.DataFrame: DataFrame tanpa baris rusak, dengan index yang di-reset.
    """
    return df[df["format"] != "ERROR"].reset_index(drop=True)


def add_md5(df):
    """Menambahkan kolom md5 berisi hash isi setiap berkas citra.

    Args:
        df (pd.DataFrame): DataFrame yang memuat kolom path.

    Returns:
        pd.DataFrame: Salinan DataFrame dengan kolom md5 tambahan.
    """
    df = df.copy()
    df["md5"] = [md5_of(p) for p in tqdm(df["path"])]
    return df


def clean_duplicates(train_df):
    """Membersihkan citra duplikat pada data training.

    Grup citra identik dengan label berbeda dibuang seluruhnya, karena label yang benar
    tidak dapat ditentukan. Grup citra identik dengan label sama disisakan satu salinan.

    Args:
        train_df (pd.DataFrame): DataFrame training yang memuat kolom md5 dan label.

    Returns:
        pd.DataFrame: DataFrame training bersih dengan index yang di-reset.
    """
    conflict_md5 = train_df.groupby("md5")["label"].nunique()
    conflict_md5 = conflict_md5[conflict_md5 > 1].index
    train_df = train_df[~train_df["md5"].isin(conflict_md5)]
    train_df = train_df.drop_duplicates(subset="md5", keep="first")
    return train_df.reset_index(drop=True)


def get_class_names(train_df):
    """Mengambil daftar nama kelas yang diurutkan berdasarkan label.

    Args:
        train_df (pd.DataFrame): DataFrame training yang memuat kolom label dan class_name.

    Returns:
        list[str]: Nama kelas dengan posisi indeks sama dengan nilai label.
    """
    pairs = train_df[["label", "class_name"]].drop_duplicates().sort_values("label")
    return pairs["class_name"].tolist()


def split_train_val(train_df, n_splits=5, random_state=42):
    """Membagi data training menjadi subset train dan validasi secara stratified dan group-aware.

    Citra dengan md5 yang sama dikelompokkan sehingga selalu berada pada fold yang sama,
    sehingga tidak ada kebocoran antara train dan validasi. Dengan n_splits=5, satu fold
    dipakai sebagai validasi (sekitar 20% data).

    Args:
        train_df (pd.DataFrame): DataFrame training bersih.
        n_splits (int): Jumlah fold.
        random_state (int): Seed untuk pengacakan fold.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: train_part dan val_part.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, val_idx = next(sgkf.split(train_df, train_df["label"], groups=train_df["md5"]))
    return train_df.iloc[train_idx], train_df.iloc[val_idx]


def build_transforms(img_h, img_w, mean, std):
    """Membuat transformasi untuk training (dengan augmentasi ringan) dan evaluasi (deterministik).

    Augmentasi tidak memakai flip maupun random crop agresif, karena orientasi dan tata
    letak dokumen merupakan ciri pembeda antar kelas.

    Args:
        img_h (int): Tinggi citra setelah resize.
        img_w (int): Lebar citra setelah resize.
        mean (list[float]): Mean normalisasi per kanal.
        std (list[float]): Standar deviasi normalisasi per kanal.

    Returns:
        tuple[transforms.Compose, transforms.Compose]: train_tf dan eval_tf.
    """
    size = (img_h, img_w)
    train_tf = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomAffine(degrees=3, translate=(0.03, 0.03), scale=(0.95, 1.05), fill=255),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


def build_image_cache(paths, name, cache_dir, img_h, img_w):
    """Menyimpan seluruh citra yang sudah di-resize ke berkas numpy memmap bertipe uint8.

    Dengan cache, setiap TIFF hanya didekode sekali sehingga pembacaan tidak menjadi
    bottleneck pada setiap epoch.

    Args:
        paths (list[str]): Daftar lokasi citra asli.
        name (str): Nama berkas cache tanpa ekstensi.
        cache_dir (str): Folder penyimpanan cache.
        img_h (int): Tinggi citra hasil cache.
        img_w (int): Lebar citra hasil cache.

    Returns:
        np.ndarray: Memmap berbentuk (N, img_h, img_w, 3) bertipe uint8.
    """
    os.makedirs(cache_dir, exist_ok=True)
    arr = np.lib.format.open_memmap(os.path.join(cache_dir, f"{name}.npy"), mode="w+",
                                    dtype=np.uint8, shape=(len(paths), img_h, img_w, 3))
    for i, p in enumerate(tqdm(paths, desc=name)):
        with Image.open(p) as img:
            # PIL memakai urutan (lebar, tinggi) saat resize
            arr[i] = np.asarray(img.convert("RGB").resize((img_w, img_h), Image.Resampling.BILINEAR))
    arr.flush()
    return arr


def build_all_caches(train_df, test_df, img_h, img_w, cache_dir):
    """Membangun cache untuk data training dan test jika ruang disk mencukupi.

    Args:
        train_df (pd.DataFrame): DataFrame training bersih.
        test_df (pd.DataFrame): DataFrame test.
        img_h (int): Tinggi citra hasil cache.
        img_w (int): Lebar citra hasil cache.
        cache_dir (str): Folder penyimpanan cache.

    Returns:
        tuple: (train_cache, test_cache). Keduanya None jika ruang disk tidak cukup.
    """
    need_gb = (len(train_df) + len(test_df)) * img_h * img_w * 3 / 1e9
    if not has_enough_space(cache_dir, need_gb):
        print("Ruang disk tidak cukup. Cache dimatikan, gambar dibaca langsung dari TIFF.")
        return None, None
    train_cache = build_image_cache(train_df["path"].tolist(), "train_cache", cache_dir, img_h, img_w)
    test_cache = build_image_cache(test_df["path"].tolist(), "test_cache", cache_dir, img_h, img_w)
    return train_cache, test_cache


def _load_rgb(cache, row, path):
    """Memuat citra RGB dari cache jika tersedia, jika tidak dari berkas TIFF.

    Args:
        cache (np.ndarray | None): Array cache, atau None.
        row (int): Posisi baris pada cache.
        path (str): Lokasi berkas TIFF asli.

    Returns:
        PIL.Image.Image: Citra dalam mode RGB.
    """
    if cache is not None:
        return Image.fromarray(np.array(cache[row]))
    with Image.open(path) as im:
        return im.convert("RGB")


class DocumentTrainDataset(Dataset):
    """Dataset untuk citra berlabel, dipakai pada data training dan validasi.

    Setiap item berupa pasangan (tensor citra, label).
    """

    def __init__(self, df, transform, cache=None):
        """Menyiapkan dataset dari DataFrame.

        Args:
            df (pd.DataFrame): Baris data yang memuat kolom path dan label.
            transform (callable): Transformasi yang diterapkan pada citra PIL.
            cache (np.ndarray | None): Array cache dari build_image_cache, atau None.
        """
        self.idx = df.index.tolist()
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform
        self.cache = cache

    def __len__(self):
        """Mengembalikan jumlah sampel dalam dataset.

        Returns:
            int: Jumlah sampel.
        """
        return len(self.labels)

    def __getitem__(self, i):
        """Mengambil satu sampel beserta labelnya.

        Args:
            i (int): Posisi sampel.

        Returns:
            tuple[torch.Tensor, int]: Citra yang sudah ditransformasi dan label.
        """
        img = _load_rgb(self.cache, self.idx[i], self.paths[i])
        return self.transform(img), self.labels[i]


class DocumentTestDataset(Dataset):
    """Dataset untuk citra uji yang tidak memiliki label.

    Setiap item berupa pasangan (tensor citra, id). Id dipakai untuk mencocokkan
    prediksi dengan sample_submission.csv.
    """

    def __init__(self, df, transform, cache=None):
        """Menyiapkan dataset uji dari DataFrame.

        Args:
            df (pd.DataFrame): Baris data yang memuat kolom path dan id.
            transform (callable): Transformasi yang diterapkan pada citra PIL.
            cache (np.ndarray | None): Array cache dari build_image_cache, atau None.
        """
        self.idx = df.index.tolist()
        self.paths = df["path"].tolist()
        self.ids = df["id"].tolist()
        self.transform = transform
        self.cache = cache

    def __len__(self):
        """Mengembalikan jumlah sampel dalam dataset.

        Returns:
            int: Jumlah sampel.
        """
        return len(self.ids)

    def __getitem__(self, i):
        """Mengambil satu sampel beserta id-nya.

        Args:
            i (int): Posisi sampel.

        Returns:
            tuple[torch.Tensor, str]: Citra yang sudah ditransformasi dan id.
        """
        img = _load_rgb(self.cache, self.idx[i], self.paths[i])
        return self.transform(img), self.ids[i]


def build_dataloaders(train_part, val_part, test_df, train_tf, eval_tf,
                      train_cache=None, test_cache=None, batch_size=32, num_workers=4):
    """Membuat DataLoader untuk training, validasi, dan test.

    Data training diacak pada setiap epoch. Data validasi dan test tidak diacak agar
    urutan keluaran sesuai dengan urutan berkas submission.

    Args:
        train_part (pd.DataFrame): Subset training.
        val_part (pd.DataFrame): Subset validasi.
        test_df (pd.DataFrame): DataFrame test.
        train_tf (callable): Transformasi training.
        eval_tf (callable): Transformasi validasi dan test.
        train_cache (np.ndarray | None): Cache citra training.
        test_cache (np.ndarray | None): Cache citra test.
        batch_size (int): Ukuran batch.
        num_workers (int): Jumlah proses pemuatan data.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: train_loader, val_loader, test_loader.
    """
    train_ds = DocumentTrainDataset(train_part, train_tf, train_cache)
    val_ds = DocumentTrainDataset(val_part, eval_tf, train_cache)
    test_ds = DocumentTestDataset(test_df, eval_tf, test_cache)

    persistent = num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, persistent_workers=persistent)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, persistent_workers=persistent)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
