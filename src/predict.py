"""Pembuatan file submission dari model terbaik pada data uji."""

import os

import pandas as pd

from src.engine import predict_batches


def write_submission(model, test_loader, sample_sub, num_classes, device, output_path):
    """Memprediksi data uji, memverifikasi format, lalu menyimpan submission.csv.

    Verifikasi yang dilakukan: urutan id harus sama dengan sample_submission, jumlah baris
    harus sama, label harus berada pada rentang 0 hingga num_classes - 1, dan kolom harus
    berurutan id lalu label.

    Args:
        model (nn.Module): Model terbaik yang sudah dimuat bobotnya.
        test_loader (DataLoader): Loader data test yang mengembalikan (x, id).
        sample_sub (pd.DataFrame): Acuan format submission dari sample_submission.csv.
        num_classes (int): Jumlah kelas.
        device (torch.device): Device komputasi.
        output_path (str): Lokasi penyimpanan submission.csv.

    Returns:
        pd.DataFrame: DataFrame submission yang sudah diverifikasi.
    """
    test_ids, test_preds = predict_batches(model, test_loader, device)
    submission = pd.DataFrame({"id": test_ids, "label": test_preds})

    assert submission["id"].tolist() == sample_sub["id"].tolist(), "Urutan id tidak sesuai"
    assert len(submission) == len(sample_sub), "Jumlah baris tidak sesuai"
    assert submission["label"].between(0, num_classes - 1).all(), "Label di luar rentang"
    assert list(submission.columns) == ["id", "label"], "Nama kolom salah"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission
