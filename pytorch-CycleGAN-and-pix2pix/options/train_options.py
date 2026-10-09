from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        # Deterministic validation, live metrics dashboard, and optional early stopping.
        parser.add_argument('--eval_source_dir', type=str, default='',
                            help='source validation images; leave empty to disable live evaluation')
        parser.add_argument('--eval_target_dir', type=str, default='',
                            help='target validation/reference images')
        parser.add_argument('--eval_source_mask_dir', type=str, default='',
                            help='optional source validation crop masks for attention IoU')
        parser.add_argument('--eval_every', type=int, default=5,
                            help='run validation every N epochs')
        parser.add_argument('--eval_max_images', type=int, default=128,
                            help='fixed deterministic validation cap per domain; 0 uses all')
        parser.add_argument('--eval_dashboard_dir', type=str, default='',
                            help='dashboard data directory; default is checkpoints/<name>/evaluation_dashboard')
        parser.add_argument('--early_stop_patience', type=int, default=0,
                            help='stop after this many validation checks without a new best; 0 disables stopping')
        parser.add_argument('--early_stop_start_epoch', type=int, default=30,
                            help='do not count early-stop patience before this epoch')
        parser.add_argument('--early_stop_min_delta', type=float, default=0.01,
                            help='minimum absolute style-score improvement to reset patience')
        parser.add_argument('--early_stop_min_dice', type=float, default=0.70,
                            help='minimum source-to-translated mask Dice for a checkpoint to be eligible as best')
        parser.add_argument('--early_stop_min_attention_iou', type=float, default=0.0,
                            help='optional minimum attention-to-crop-mask IoU; 0 disables this guard')

        self.isTrain = True
        return parser
