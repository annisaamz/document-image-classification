"""Definisi arsitektur model: CNN buatan sendiri (docnet) dan ResNet18 pretrained (resnet18)."""

import torch.nn as nn
import torchvision.models as models


class DocumentCNN(nn.Module):
    """CNN sederhana yang dilatih dari nol untuk klasifikasi citra dokumen.

    Terdiri atas blok Conv-BatchNorm-ReLU-MaxPool, AdaptiveAvgPool2d, lalu MLP klasifikasi.
    """

    def __init__(self, num_classes=16, dropout=0.3, channels=(32, 64, 128, 256), hidden=256, pool_size=4):
        """Membangun arsitektur CNN.

        Args:
            num_classes (int): Jumlah kelas output.
            dropout (float): Probabilitas dropout pada lapisan klasifikasi.
            channels (tuple[int]): Jumlah filter pada setiap blok konvolusi.
            hidden (int): Jumlah neuron pada lapisan tersembunyi MLP.
            pool_size (int): Ukuran output AdaptiveAvgPool2d (pool_size x pool_size).
        """
        super().__init__()
        blocks = []
        in_ch = 3  # RGB
        for out_ch in channels:
            blocks += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),  # ekstraksi fitur lokal
                nn.BatchNorm2d(out_ch),                              # menstabilkan aktivasi
                nn.ReLU(inplace=True),                               # non-linearitas
                nn.MaxPool2d(2),                                     # memperkecil ukuran spasial
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * pool_size * pool_size, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),  # logit per kelas, tanpa softmax
        )

    def forward(self, x):
        """Meneruskan batch citra melalui jaringan.

        Args:
            x (torch.Tensor): Batch citra berbentuk (N, 3, H, W).

        Returns:
            torch.Tensor: Logit berbentuk (N, num_classes).
        """
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class DocumentResNet(nn.Module):
    """ResNet18 dengan bobot pretrained ImageNet dan lapisan fc diganti untuk klasifikasi dokumen.

    Seluruh layer di-fine-tune. Lapisan fc asli diganti menjadi Dropout lalu Linear
    dengan num_classes output.
    """

    def __init__(self, num_classes=16, dropout=0.5, pretrained=True):
        """Membangun ResNet18 untuk klasifikasi dokumen.

        Args:
            num_classes (int): Jumlah kelas output.
            dropout (float): Probabilitas dropout sebelum lapisan linear.
            pretrained (bool): Jika True, memuat bobot ImageNet IMAGENET1K_V1.
        """
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = models.resnet18(weights=weights)
        in_feats = self.net.fc.in_features
        self.net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feats, num_classes))

    def forward(self, x):
        """Meneruskan batch citra melalui ResNet18.

        Args:
            x (torch.Tensor): Batch citra berbentuk (N, 3, H, W).

        Returns:
            torch.Tensor: Logit berbentuk (N, num_classes).
        """
        return self.net(x)


def build_model(config, num_classes, device):
    """Membuat model sesuai config["model_name"] lalu memindahkannya ke device.

    Jika bobot pretrained gagal dimuat (misalnya tidak ada koneksi internet), model
    memakai inisialisasi acak dan peringatan ditampilkan.

    Args:
        config (dict): Konfigurasi dari src.config.CONFIG.
        num_classes (int): Jumlah kelas output.
        device (torch.device): Device tujuan.

    Returns:
        nn.Module: Model yang sudah berada di device.
    """
    if config["model_name"] == "resnet18":
        try:
            model = DocumentResNet(num_classes, config["dropout"], config["pretrained"])
        except Exception as e:
            print("PERINGATAN: bobot pretrained gagal dimuat:", e)
            print("Model memakai bobot acak. Periksa koneksi internet lalu jalankan ulang.")
            model = DocumentResNet(num_classes, config["dropout"], pretrained=False)
        return model.to(device)

    model = DocumentCNN(num_classes, config["dropout"], tuple(config["channels"]), config["hidden"])
    return model.to(device)
