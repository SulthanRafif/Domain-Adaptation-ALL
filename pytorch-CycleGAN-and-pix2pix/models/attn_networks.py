"""
Komponen jaringan untuk Attention-Guided CycleGAN (replikasi Baydilli, 2025).

Diletakkan di: pytorch-CycleGAN-and-pix2pix/models/attn_networks.py

Berisi tiga komponen:
  1. SpatialAttention  -> Algoritma 2 paper (atensi spasial antar blok konvolusi)
  2. AttentionGate     -> Algoritma 1 paper (modul A_S / A_T penghasil s_a, s_f, s_b, s')
  3. AttnResnetGenerator -> Algoritma 3 paper (generator ResNet + skip connection + atensi)

Catatan: file ini tidak mengubah networks.py bawaan repo junyanz. Discriminator
tetap memakai NLayerDiscriminator (--netD basic, n_layers=3) yang sudah menghasilkan
peta 14x14 untuk input 128x128, persis seperti Bagian 4.1 paper.
"""

import torch
import torch.nn as nn
import functools


# ---------------------------------------------------------------------------
# 1. Spatial Attention  (Algoritma 2, Persamaan 1)
# ---------------------------------------------------------------------------
class SpatialAttention(nn.Module):
    """F_s = (sigmoid(f_7x7([AvgPool(F); MaxPool(F)]))) * F

    Pooling dilakukan sepanjang dimensi KANAL (channel-wise), menghasilkan dua
    peta H x W x 1 yang kemudian dikonkatenasi menjadi H x W x 2. Ini identik
    dengan submodul spatial attention pada CBAM.
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size harus ganjil agar padding simetris"
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)        # B x 1 x H x W
        max_out, _ = torch.max(x, dim=1, keepdim=True)      # B x 1 x H x W
        attn = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn                                     # residual-style gating


# ---------------------------------------------------------------------------
# 2. Attention Gate A_S / A_T  (Algoritma 1)
# ---------------------------------------------------------------------------
class AttentionGate(nn.Module):
    """Menghasilkan mask atensi s_a dari keluaran generator, lalu menggabungkan
    foreground hasil translasi dengan background citra rujukan.

        s_a = sigmoid(f_7x7([AvgPool(G(s)); MaxPool(G(s))]))
        s_f = s_a        * G(s)      <- foreground: diambil dari citra TERTRANSLASI
        s_b = (1 - s_a)  * ref       <- background: diambil dari citra RUJUKAN
        s'  = s_f + s_b

    PENTING - ambiguitas paper:
      Algoritma 1 baris 5 menulis  s_b = (1 - s_a) * s, yaitu background diambil
      dari citra SOURCE asli. Namun C-NMC sudah tersegmentasi (latar hitam),
      sehingga rumus itu menghasilkan citra dengan latar hitam - sementara
      Tabel 2 paper menampilkan hasil bertumpuk latar eritrosit. Karena itu
      modul ini menyediakan dua mode lewat argumen `bg_source`:
        'input'     -> setia pada Algoritma 1 (background dari citra masukan)
        'generated' -> background dari G(s), konsisten dengan Tabel 2
      Jalankan keduanya dan bandingkan secara visual serta lewat FID.
    """

    def __init__(self, kernel_size=7, bg_source='input'):
        super().__init__()
        assert bg_source in ('input', 'generated')
        self.bg_source = bg_source
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, gen_out, ref, return_logits=False):
        """gen_out : G(s), keluaran generator            (B x 3 x H x W)
           ref     : citra rujukan untuk background, yaitu s (B x 3 x H x W)
           return  : (s_a, s_translated)
        """
        avg_out = torch.mean(gen_out, dim=1, keepdim=True)
        max_out, _ = torch.max(gen_out, dim=1, keepdim=True)
        logits = self.conv(torch.cat([avg_out, max_out], dim=1))
        s_a = torch.sigmoid(logits)

        s_f = s_a * gen_out
        bg = gen_out if self.bg_source == 'generated' else ref
        s_b = (1.0 - s_a) * bg
        if return_logits:
            return s_a, s_f + s_b, logits
        return s_a, s_f + s_b


# ---------------------------------------------------------------------------
# 3. Generator  (Algoritma 3 / Bagian 4.1)
# ---------------------------------------------------------------------------
class ResnetBlock(nn.Module):
    """Blok residual standar CycleGAN (reflection pad + 2 konv 3x3)."""

    def __init__(self, dim, norm_layer, use_bias):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, bias=use_bias),
            norm_layer(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class AttnResnetGenerator(nn.Module):
    """Generator ResNet 6-blok dengan spatial attention dan skip connection.

    Arsitektur untuk masukan 128 x 128 x 3 (Bagian 4.1 paper):

        encoder   7x7x64  s1  -> 128 x 128 x 64   (+SA, skip 0)
                  3x3x128 s2  ->  64 x  64 x 128  (+SA, skip 1)
                  3x3x256 s2  ->  32 x  32 x 256  (+SA)
        transform 6 x ResnetBlock (32 x 32 x 256)
        decoder   3x3x128 s2T ->  64 x  64 x 128  (+SA, concat skip 1 -> 256)
                  3x3x64  s2T -> 128 x 128 x 64   (+SA, concat skip 0 -> 128)
                  7x7x3   s1  -> 128 x 128 x 3    (Tanh)

    Catatan ambiguitas: Algoritma 3 baris 12 menyiratkan tiga skip connection,
    tetapi peta pada resolusi 32x32 tidak punya pasangan di decoder karena blok
    residual mempertahankan resolusi. Implementasi ini memakai dua skip yang
    resolusinya cocok. Nyatakan pilihan ini di naskah Anda.
    """

    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=6,
                 norm_layer=nn.InstanceNorm2d, use_skip=True, attn_kernel=7):
        super().__init__()
        # sama seperti repo junyanz: instance-norm tidak punya parameter bias,
        # jadi konvolusi sebelumnya perlu bias sendiri
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.use_skip = use_skip

        # ---------------- encoder ----------------
        self.enc0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7, padding=0, bias=use_bias),
            norm_layer(ngf), nn.ReLU(True))
        self.sa0 = SpatialAttention(attn_kernel)

        self.enc1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 2), nn.ReLU(True))
        self.sa1 = SpatialAttention(attn_kernel)

        self.enc2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 4), nn.ReLU(True))
        self.sa2 = SpatialAttention(attn_kernel)

        # ---------------- transformer ----------------
        self.res = nn.Sequential(
            *[ResnetBlock(ngf * 4, norm_layer, use_bias) for _ in range(n_blocks)])

        # ---------------- decoder ----------------
        # self.dec0 = nn.Sequential(
        #     nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, stride=2, padding=1,
        #                        output_padding=1, bias=use_bias),
        #     norm_layer(ngf * 2), nn.ReLU(True))

        self.dec0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ngf * 4, ngf * 2, 3, stride=1, padding=1,
                bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(True),
        )
        self.sa3 = SpatialAttention(attn_kernel)

        dec1_in = ngf * 4 if use_skip else ngf * 2     # concat skip 1
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(dec1_in, ngf, 3, stride=1, padding=1, bias=use_bias),
            norm_layer(ngf), nn.ReLU(True))
        self.sa4 = SpatialAttention(attn_kernel)

        out_in = ngf * 2 if use_skip else ngf          # concat skip 0
        self.out = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(out_in, output_nc, 7, padding=0),
            nn.Tanh())

    def forward(self, x):
        s0 = self.sa0(self.enc0(x))       # 128 x 128 x 64
        s1 = self.sa1(self.enc1(s0))      #  64 x  64 x 128
        s2 = self.sa2(self.enc2(s1))      #  32 x  32 x 256

        h = self.res(s2)                  #  32 x  32 x 256

        h = self.sa3(self.dec0(h))        #  64 x  64 x 128
        if self.use_skip:
            h = torch.cat([h, s1], dim=1) #  64 x  64 x 256

        h = self.sa4(self.dec1(h))        # 128 x 128 x 64
        if self.use_skip:
            h = torch.cat([h, s0], dim=1) # 128 x 128 x 128

        return self.out(h)                # 128 x 128 x 3  (content mask)


# ---------------------------------------------------------------------------
# Helper: pembuatan generator + inisialisasi bobot ala repo junyanz
# ---------------------------------------------------------------------------
def define_attn_G(input_nc=3, output_nc=3, ngf=64, n_blocks=6, norm='instance',
                  use_skip=True, init_type='normal', init_gain=0.02):
    # init_net pada repo master menangani penempatan device sendiri;
    # parameter gpu_ids sudah dihapus dari API-nya.
    from models.networks import init_net

    if norm == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d,
                                       affine=False, track_running_stats=False)
    elif norm == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True,
                                       track_running_stats=True)
    else:
        raise NotImplementedError(f"norm '{norm}' tidak dikenali")

    net = AttnResnetGenerator(input_nc, output_nc, ngf, n_blocks,
                              norm_layer=norm_layer, use_skip=use_skip)
    return init_net(net, init_type, init_gain)


def define_attn_gate(kernel_size=7, bg_source='input',
                     init_type='normal', init_gain=0.02):
    """Modul A_S / A_T. Bobotnya ikut dioptimasi bersama generator."""
    from models.networks import init_net
    return init_net(AttentionGate(kernel_size, bg_source),
                    init_type, init_gain)
