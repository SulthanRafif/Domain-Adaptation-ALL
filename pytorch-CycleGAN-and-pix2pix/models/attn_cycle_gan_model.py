"""
Attention-Guided CycleGAN — replikasi Baydilli (2025), Biomed. Signal Process. Control 101:107159

Diletakkan di: pytorch-CycleGAN-and-pix2pix/models/attn_cycle_gan_model.py
Dipanggil dengan: --model attn_cycle_gan

Pemetaan ke persamaan paper:
  Pers. (8)  L_AGAN  : LSGAN, discriminator menerima citra BERMASK  -> self.criterionGAN
  Pers. (4)  L_cycle : ||s - s''||_1                                -> self.criterionCycle
  Pers. (5)  L_pixel : ||s - s'||_1                                 -> self.criterionPixel
  Pers. (6)  total   : lambda_gan * L_AGAN + lambda_cycle * L_cycle + lambda_pixel * L_pixel

Nilai bawaan mengikuti Algoritma 4 paper:
  lambda_gan = 0.5, lambda_cycle = 10, lambda_pixel = 1, epochs = 200
"""

import torch
import itertools

from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .attn_networks import define_attn_G, define_attn_gate


class AttnCycleGANModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_gan', type=float, default=0.5,
                                help='bobot L_AGAN (Algoritma 4: 0.5)')
            parser.add_argument('--lambda_cycle', type=float, default=10.0,
                                help='bobot L_cycle (Algoritma 4: 10)')
            parser.add_argument('--lambda_pixel', type=float, default=1.0,
                                help='bobot L_pixel (Algoritma 4: 1)')
            parser.add_argument('--bg_source', type=str, default='input',
                                choices=['input', 'generated'],
                                help="asal background pada Algoritma 1; "
                                     "'input' setia pada teks algoritma, "
                                     "'generated' konsisten dengan Tabel 2")
            parser.add_argument('--use_D_S', action='store_true',
                                help='aktifkan discriminator arah balik (D_S). '
                                     'Paper HANYA memakai D_T; flag ini untuk ablasi.')
            parser.add_argument('--no_skip', action='store_true',
                                help='matikan skip connection pada generator')
            parser.add_argument('--lambda_mask', type=float, default=0.0,
                                help='penahan kolaps mask. 0 = mati (setia pada '
                                     'paper). Aktifkan (mis. 1.0) HANYA bila '
                                     'diagnosis menunjukkan s_a kolaps.')
            parser.add_argument('--mask_min', type=float, default=0.15,
                                help='batas BAWAH proporsi wilayah yang ditandai '
                                     'mask; dipakai oleh --lambda_mask')
            parser.add_argument('--mask_max', type=float, default=0.45,
                                help='batas ATAS proporsi wilayah yang ditandai '
                                     'mask. Tanpa batas atas, s_a=1 di seluruh '
                                     'bidang memuaskan penahan secara sempurna '
                                     'dan modul atensi menjadi mati.')
            parser.add_argument('--no_mask_D', action='store_true',
                                help='discriminator melihat citra UTUH, bukan '
                                     'citra bermask. Menyimpang dari Pers. (8), '
                                     'tetapi menutup jalan pintas s_a -> 0 yang '
                                     'memadamkan gradien adversarial.')
        return parser

    # ------------------------------------------------------------------
    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = ['G', 'AGAN', 'cycle', 'pixel', 'D_T']
        if opt.isTrain and getattr(opt, 'lambda_mask', 0.0) > 0:
            self.loss_names.append('mask')
        if opt.isTrain and opt.use_D_S:
            self.loss_names.append('D_S')

        # visual: s, G(s), s' (hasil translasi), s'' (rekonstruksi), mask s_a.
        # attn_S_vis, bukan attn_S: tensor2im bawaan repo mengasumsikan rentang
        # [-1,1], sedangkan s_a adalah sigmoid di [0,1]. Tanpa penskalaan ini,
        # s_a=0 dirender abu-abu 127 dan mask sehat pun tampak pucat.
        self.visual_names = ['real_S', 'raw_ST', 'fake_T', 'rec_S', 'attn_S_vis']

        if self.isTrain:
            self.model_names = ['G_ST', 'F_TS', 'A_S', 'A_T', 'D_T']
            if opt.use_D_S:
                self.model_names.append('D_S')
        else:
            self.model_names = ['G_ST', 'A_S']

        use_skip = not getattr(opt, 'no_skip', False)
        bg_source = getattr(opt, 'bg_source', 'input')

        # --- generator & attention gate, dua arah (Algoritma 3 + Algoritma 1)
        self.netG_ST = define_attn_G(opt.input_nc, opt.output_nc, opt.ngf,
                                        n_blocks=6, norm=opt.norm, use_skip=use_skip,
                                        init_type=opt.init_type,
                                        init_gain=opt.init_gain)
        self.netF_TS = define_attn_G(opt.output_nc, opt.input_nc, opt.ngf,
                                        n_blocks=6, norm=opt.norm, use_skip=use_skip,
                                        init_type=opt.init_type,
                                        init_gain=opt.init_gain)
        self.netA_S = define_attn_gate(7, bg_source, opt.init_type,
                                        opt.init_gain)
        self.netA_T = define_attn_gate(7, bg_source, opt.init_type,
                                        opt.init_gain)

        if self.isTrain:
            # PatchGAN 70x70 bawaan repo: untuk input 128x128 menghasilkan 14x14,
            # persis seperti Bagian 4.1 paper. Tidak perlu diubah.
            self.netD_T = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm,
                                            opt.init_type, opt.init_gain)
            if opt.use_D_S:
                self.netD_S = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                                opt.n_layers_D, opt.norm,
                                                opt.init_type, opt.init_gain)

            self.fake_T_pool = ImagePool(opt.pool_size)   # buffer 50 citra
            self.fake_S_pool = ImagePool(opt.pool_size)

            # gan_mode='lsgan' -> Persamaan (8), least-squares
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionPixel = torch.nn.L1Loss()

            gen_params = itertools.chain(self.netG_ST.parameters(),
                                        self.netF_TS.parameters(),
                                        self.netA_S.parameters(),
                                        self.netA_T.parameters())
            self.optimizer_G = torch.optim.Adam(gen_params, lr=opt.lr,
                                                betas=(opt.beta1, 0.999))
            d_params = list(self.netD_T.parameters())
            if opt.use_D_S:
                d_params += list(self.netD_S.parameters())
            self.optimizer_D = torch.optim.Adam(d_params, lr=opt.lr,
                                                betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    # ------------------------------------------------------------------
    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_S = (input['A' if AtoB else 'B']).to(self.device)
        self.real_T = (input['B' if AtoB else 'A']).to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    # ------------------------------------------------------------------
    def forward(self):
        """Algoritma 4 baris 2-5."""
        # arah maju: s -> G(s) -> s'
        self.raw_ST = self.netG_ST(self.real_S)                    # G(s)
        self.attn_S, self.fake_T = self.netA_S(self.raw_ST, self.real_S)   # s_a, s'
        self.attn_S_vis = self.attn_S * 2.0 - 1.0                  # [0,1] -> [-1,1]

        # arah balik untuk cycle: s' -> F(s') -> s''
        # Catatan: Algoritma 4 baris 4-5 menulis F(t) dan A_t(F(t)), tetapi teks
        # Bagian 4 menyatakan "the fake image (s') is fed into the generator F"
        # dan background s'_b diambil dari s'. Teks yang dipakai di sini, karena
        # hanya itu yang konsisten dengan L_cycle = ||s - s''||_1.
        self.raw_TS = self.netF_TS(self.fake_T)                    # F(s')
        self.attn_T, self.rec_S = self.netA_T(self.raw_TS, self.fake_T)    # s''

        if self.isTrain and self.opt.use_D_S:
            self.raw_T_ = self.netF_TS(self.real_T)
            _, self.fake_S = self.netA_T(self.raw_T_, self.real_T)

    # ------------------------------------------------------------------
    def backward_D_T(self):
        """Persamaan (8): discriminator melihat citra yang SUDAH DIMASK.

        Mask s_a berasal dari sampel source. Mask yang sama dikenakan pada citra
        target nyata (t) maupun citra palsu (s'), sehingga D_T hanya menilai
        kemiripan domain pada wilayah sel, bukan pada latar.
        """
        fake_T = self.fake_T_pool.query(self.fake_T.detach())
        # dengan --no_mask_D, s_a -> 0 TIDAK lagi membutakan discriminator:
        # s' menjadi sama dengan s (citra source murni) yang langsung dikenali
        # palsu, sehingga gradiennya justru mendorong mask naik.
        m = 1.0 if getattr(self.opt, 'no_mask_D', False) else self.attn_S.detach()

        pred_real = self.netD_T(m * self.real_T)
        pred_fake = self.netD_T(m * fake_T)

        loss_real = self.criterionGAN(pred_real, True)
        loss_fake = self.criterionGAN(pred_fake, False)
        # dibagi 2 untuk memperlambat discriminator (praktik CycleGAN)
        self.loss_D_T = (loss_real + loss_fake) * 0.5
        self.loss_D_T.backward()

    def backward_D_S(self):
        """Hanya aktif bila --use_D_S. Tidak ada di paper; untuk ablasi."""
        s_a = self.attn_T.detach()
        fake_S = self.fake_S_pool.query(self.fake_S.detach())
        pred_real = self.netD_S(s_a * self.real_S)
        pred_fake = self.netD_S(s_a * fake_S)
        self.loss_D_S = (self.criterionGAN(pred_real, True) +
                         self.criterionGAN(pred_fake, False)) * 0.5
        self.loss_D_S.backward()

    # ------------------------------------------------------------------
    def backward_G(self):
        """Persamaan (6)."""
        l_gan = self.opt.lambda_gan
        l_cyc = self.opt.lambda_cycle
        l_pix = self.opt.lambda_pixel

        # L_AGAN: generator ingin D_T menganggap s' sebagai nyata
        no_mask = getattr(self.opt, 'no_mask_D', False)
        mT = 1.0 if no_mask else self.attn_S
        self.loss_AGAN = self.criterionGAN(self.netD_T(mT * self.fake_T), True)
        if self.opt.use_D_S:
            mS = 1.0 if no_mask else self.attn_T
            self.loss_AGAN = self.loss_AGAN + self.criterionGAN(
                self.netD_S(mS * self.fake_S), True)

        # Pers. (4) cycle-consistency, Pers. (5) pixel-loss
        self.loss_cycle = self.criterionCycle(self.rec_S, self.real_S)
        self.loss_pixel = self.criterionPixel(self.fake_T, self.real_S)

        self.loss_G = (l_gan * self.loss_AGAN +
                       l_cyc * self.loss_cycle +
                       l_pix * self.loss_pixel)

        # --- penahan kolaps mask (opsional, di luar paper) ---
        # L_pixel = ||s_a * (s - G(s))||_1, sehingga s_a -> 0 memuaskannya secara
        # trivial; pada saat yang sama D_T hanya melihat s_a*t dan s_a*s', yang
        # ikut menjadi hitam sehingga gradien adversarial padam. Kolaps ini
        # memperkuat dirinya sendiri. Suku berikut memberi penalti bila rerata
        # mask jatuh di bawah ambang, sekaligus mendorongnya mendekati 0 atau 1.
        lm = getattr(self.opt, 'lambda_mask', 0.0)
        if lm > 0:
            lo = getattr(self.opt, 'mask_min', 0.15)
            hi = getattr(self.opt, 'mask_max', 0.45)
            m = self.attn_S.mean()
            # DUA SISI. Versi sebelumnya hanya memberi lantai, sehingga s_a=1
            # di seluruh bidang memuaskannya sempurna: mask meliputi segalanya,
            # s' = G(s), dan modul atensi menjadi mati. Batas atas menutup itu.
            band = torch.relu(lo - m) ** 2 + torch.relu(m - hi) ** 2
            binarize = (self.attn_S * (1.0 - self.attn_S)).mean()
            self.loss_mask = band + 0.1 * binarize
            self.loss_G = self.loss_G + lm * self.loss_mask

        self.loss_G.backward()

    # ------------------------------------------------------------------
    def optimize_parameters(self):
        self.forward()

        # --- generator + modul atensi
        nets_D = [self.netD_T] + ([self.netD_S] if self.opt.use_D_S else [])
        self.set_requires_grad(nets_D, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # --- discriminator
        self.set_requires_grad(nets_D, True)
        self.optimizer_D.zero_grad()
        self.backward_D_T()
        if self.opt.use_D_S:
            self.backward_D_S()
        self.optimizer_D.step()