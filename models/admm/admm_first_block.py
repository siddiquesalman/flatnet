import torch
import torch.nn as nn
from models.admm.admm_helper_functions_torch import *
from models.admm.admm_rgb_pytorch import *
import models.admm.admm_filters_no_soft as admm_s
import logging
from sacred import Experiment
from matplotlib import pyplot as plt

from utils.tupperware import tupperware
from config import initialise
from utils.ops import roll_n

ex = Experiment("admm-model")
ex = initialise(ex)


def resize(img, factor):
    num = int(-np.log2(factor))

    for i in range(num):
        dim_x = img.shape[0]
        dim_y = img.shape[1]
        pad_x = 1 if dim_x % 2 == 1 else 0
        pad_y = 1 if dim_y % 2 == 1 else 0
        img = 0.25 * (
            img[: dim_x - pad_x : 2, : dim_y - pad_y : 2]
            + img[1::2, : dim_y - pad_y : 2]
            + img[: dim_x - pad_x : 2, 1::2]
            + img[1::2, 1::2]
        )

    return img


class ADMM_Net(nn.Module):
    def __init__(
        self,
        args,
        learning_options={"learned_vars": ["mus", "tau"]},
        le_admm_s=False,
        factor=0.25,
        denoise_model=[],
    ):
        super(ADMM_Net, self).__init__()

        cuda_device = (
            torch.device(args.device)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.iterations = args.admm_iterations  # Number of unrolled iterations
        self.batch_size = args.batch_size  # Batch size
        self.autotune = False  # Using autotune (True or False)
        self.realdata = True  # Real Data or Simulated Measurements
        self.printstats = False  # Print ADMM Variables

        self.addnoise = False  # Add noise (only if using simulated data)
        self.noise_std = 0.05  # Noise standard deviation
        self.cuda_device = cuda_device
        self.args = args

        self.l_admm_s = (
            le_admm_s
        )  # Turn on if using Le-ADMM*, otherwise should be set to False
        if le_admm_s == True:
            self.Denoiser = denoise_model

        # Leared structure options
        self.learning_options = learning_options
        logging.info(f"ADMM with learnt {learning_options['learned_vars']}")

        ## Initialize constants 1518 x 2012
        h = np.load(args.psf_mat)
        psf_crop_top = args.psf_centre_x - args.psf_crop_size_x // 2
        psf_crop_bottom = args.psf_centre_x + args.psf_crop_size_x // 2
        psf_crop_left = args.psf_centre_y - args.psf_crop_size_y // 2
        psf_crop_right = args.psf_centre_y + args.psf_crop_size_y // 2

        h = h[psf_crop_top:psf_crop_bottom, psf_crop_left:psf_crop_right]

        self.factor = factor
        h = resize(h, self.factor)
        if args.normalise_admm_psf:
            h /= np.linalg.norm(h.ravel())
        h = torch.tensor(h.copy()).float()

        self.DIMS0 = h.shape[0]  # Image Dimensions
        self.DIMS1 = h.shape[1]  # Image Dimensions

        self.PAD_SIZE0 = 0  # Pad size
        self.PAD_SIZE1 = 0  # Pad size

        # Initialize Variables
        self.initialize_learned_variables(learning_options)

        # PSF
        self.h_var = nn.Parameter(
            torch.tensor(h, dtype=torch.float32), requires_grad=False
        )

        self.h_zeros = nn.Parameter(
            torch.zeros(
                self.DIMS0 + 2 * self.PAD_SIZE0,
                self.DIMS1 + 2 * self.PAD_SIZE1,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

        self.h_complex = torch.stack(
            (pad_zeros_torch(self, self.h_var), self.h_zeros), 2
        ).unsqueeze(0)

        self.H = nn.Parameter(
            torch.fft(batch_ifftshift2d(self.h_complex).squeeze(), 2),
            requires_grad=False,
        )
        self.Hconj = nn.Parameter(
            self.H * torch.tensor([1, -1]).float(), requires_grad=False
        )
        self.HtH = nn.Parameter(
            complex_abs(complex_multiplication(self.H, self.Hconj)), requires_grad=False
        )

        self.LtL = nn.Parameter(
            torch.tensor(make_laplacian(self), dtype=torch.float32), requires_grad=False
        )

        self.resid_tol = nn.Parameter(
            torch.tensor(1.5, dtype=torch.float32), requires_grad=False
        )
        self.mu_inc = nn.Parameter(
            torch.tensor(1.2, dtype=torch.float32), requires_grad=False
        )
        self.mu_dec = nn.Parameter(
            torch.tensor(1.2, dtype=torch.float32), requires_grad=False
        )

    def initialize_learned_variables(self, learning_options):

        self.mu1 = nn.Parameter(
            torch.tensor(np.ones(self.iterations) * 1e-6, dtype=torch.float32),
            requires_grad="mus" in learning_options["learned_vars"],
        )
        self.mu2 = nn.Parameter(
            torch.tensor(np.ones(self.iterations) * 1e-5, dtype=torch.float32),
            requires_grad="mus" in learning_options["learned_vars"],
        )
        self.mu3 = nn.Parameter(
            torch.tensor(np.ones(self.iterations) * 1e-5, dtype=torch.float32).float(),
            requires_grad="mus" in learning_options["learned_vars"],
        )

        self.tau = nn.Parameter(
            torch.tensor(np.ones(self.iterations) * 1e-4, dtype=torch.float32),
            requires_grad="tau" in learning_options["learned_vars"],
        )

        # self.mu1 = nn.Parameter(
        #     torch.tensor(np.ones(self.iterations) * 1e-4, dtype=torch.float32),
        #     requires_grad="mus" in learning_options["learned_vars"],
        # )
        # self.mu2 = nn.Parameter(
        #     torch.tensor(np.ones(self.iterations) * 1e-4, dtype=torch.float32),
        #     requires_grad="mus" in learning_options["learned_vars"],
        # )
        # self.mu3 = nn.Parameter(
        #     torch.tensor(np.ones(self.iterations) * 1e-4, dtype=torch.float32).float(),
        #     requires_grad="mus" in learning_options["learned_vars"],
        # )
        #
        # self.tau = nn.Parameter(
        #     torch.tensor(np.ones(self.iterations) * 2e-4, dtype=torch.float32),
        #     requires_grad="tau" in learning_options["learned_vars"],
        # )

    def _resize(self, img, factor):
        num = int(-np.log2(factor))

        for i in range(num):
            dim_x = img.shape[2]
            dim_y = img.shape[3]
            pad_x = 1 if dim_x % 2 == 1 else 0
            pad_y = 1 if dim_y % 2 == 1 else 0
            img = 0.25 * (
                img[:, :, : dim_x - pad_x : 2, : dim_y - pad_y : 2]
                + img[:, :, 1::2, : dim_y - pad_y : 2]
                + img[:, :, : dim_x - pad_x : 2, 1::2]
                + img[:, :, 1::2, 1::2]
            )

        return img

    def forward(self, inputs):
        self.batch_size = inputs.shape[0]

        inputs = inputs * 0.5 + 0.5

        self.mu_vals = torch.stack([self.mu1, self.mu2, self.mu3, self.tau])

        self.admmstats = {
            "dual_res_s": [],
            "dual_res_u": [],
            "dual_res_w": [],
            "primal_res_s": [],
            "primal_res_u": [],
            "primal_res_w": [],
            "data_loss": [],
            "total_loss": [],
        }

        if self.autotune == True:
            self.mu_auto_list = {"mu1": [], "mu2": [], "mu3": []}

        # If using simulated data, input the raw image and run through forward model
        if self.realdata == False:
            y = crop(self, self.Hfor(pad_dim2(self, inputs)))
            if self.addnoise == True:
                y = self.gaussian_noise_layer(y, self.noise_std)

        # Otherwise, input is the normalized Diffuser Image
        else:
            y = inputs

        # Cty = pad_zeros_torch(self, y)  # Zero padded input
        Cty = pad_zeros_torch(self, y)
        CtC = pad_zeros_torch(self, torch.ones_like(y))

        if self.args.meas_crop_size_x and self.args.meas_crop_size_y:
            crop_size_x = self.args.meas_crop_size_x
            crop_size_y = self.args.meas_crop_size_y

            psf_h = self.args.psf_height
            psf_w = self.args.psf_width

            Cty = pad_zeros_torch_CtC(
                self,
                y[
                    :,
                    :,
                    int(self.factor * (psf_h - crop_size_x) // 2) : int(
                        self.factor * (psf_h + crop_size_x) // 2
                    ),
                    int(self.factor * (psf_w - crop_size_y) // 2) : int(
                        self.factor * (psf_w + crop_size_y) // 2
                    ),
                ],
            )
            CtC = pad_zeros_torch_CtC(
                self,
                torch.ones_like(
                    y[
                        :,
                        :,
                        int(self.factor * (psf_h - crop_size_x) // 2) : int(
                            self.factor * (psf_h + crop_size_x) // 2
                        ),
                        int(self.factor * (psf_w - crop_size_y) // 2) : int(
                            self.factor * (psf_w + crop_size_y) // 2
                        ),
                    ]
                ),
            )  # Zero padded ones

        # Create list of inputs/outputs
        in_vars = []
        in_vars1 = []
        in_vars2 = []
        Hsk_list = []
        a2k_1_list = []
        a2k_2_list = []

        sk = torch.zeros_like(Cty, dtype=torch.float32)
        alpha1k = torch.zeros_like(Cty, dtype=torch.float32)
        alpha3k = torch.zeros_like(Cty, dtype=torch.float32)
        Hskp = torch.zeros_like(Cty, dtype=torch.float32)

        if self.l_admm_s == True:
            Lsk_init, mem_init = self.Denoiser.forward(sk)
            alpha2k = torch.zeros_like(Lsk_init, dtype=torch.float32)
        else:

            alpha2k_1 = torch.zeros_like(sk[:, :, :-1, :], dtype=torch.float32)
            alpha2k_2 = torch.zeros_like(sk[:, :, :, :-1], dtype=torch.float32)

            a2k_1_list.append(alpha2k_1)
            a2k_2_list.append(alpha2k_2)

        mu_auto = torch.stack([self.mu1[0], self.mu2[0], self.mu3[0], self.tau[0]])

        in_vars.append(torch.stack([sk, alpha1k, alpha3k, Hskp]))

        for i in range(0, self.iterations):
            if self.l_admm_s == True:

                out_vars, alpha2k, _, symm, admmstats = admm_s.admm(
                    self, in_vars[-1], alpha2k, CtC, Cty, [], i, y
                )
                in_vars.append(out_vars)

            else:

                if self.autotune == True:
                    out_vars, a_out1, a_out2, mu_auto, symm, admmstats = admm(
                        self,
                        in_vars[-1],
                        a2k_1_list[-1],
                        a2k_2_list[-1],
                        CtC,
                        Cty,
                        mu_auto,
                        i,
                        y,
                    )

                    self.mu_auto_list["mu1"].append(mu_auto[0])
                    self.mu_auto_list["mu2"].append(mu_auto[1])
                    self.mu_auto_list["mu3"].append(mu_auto[2])

                else:
                    out_vars, a_out1, a_out2, _, symm, admmstats = admm(
                        self,
                        in_vars[-1],
                        a2k_1_list[-1],
                        a2k_2_list[-1],
                        CtC,
                        Cty,
                        [],
                        i,
                        y,
                    )

                # if torch.any(out_vars != out_vars):
                #    print('loop')

                in_vars.append(out_vars)
                a2k_1_list.append(a_out1)
                a2k_2_list.append(a_out2)

            if self.printstats == True:  # Print ADMM Variables
                self.admmstats["dual_res_s"].append(admmstats["dual_res_s"])
                self.admmstats["primal_res_s"].append(admmstats["primal_res_s"])
                self.admmstats["dual_res_w"].append(admmstats["dual_res_w"])
                self.admmstats["primal_res_w"].append(admmstats["primal_res_w"])
                self.admmstats["dual_res_u"].append(admmstats["dual_res_u"])
                self.admmstats["primal_res_u"].append(admmstats["primal_res_u"])
                self.admmstats["data_loss"].append(admmstats["data_loss"])
                self.admmstats["total_loss"].append(admmstats["total_loss"])

            x_out = crop(self, in_vars[-1][0])
            x_outn = normalize_image(x_out)
            self.in_list = in_vars

        out_h, out_w = x_outn.shape[2:]
        x_outn = x_outn[
            :,
            :,
            out_h // 2
            - int(self.args.image_height * self.factor) // 2 : out_h // 2
            + int(self.args.image_height * self.factor) // 2,
            out_w // 2
            - int(self.args.image_width * self.factor) // 2 : out_w // 2
            + int(self.args.image_width * self.factor) // 2,
        ]
        return x_outn  # , symm


def get_img_from_raw(raw):
    raw = raw / 4096.0
    img = np.zeros((1518, 2012, 3))

    img[:, :, 0] = raw[0::2, 0::2]  # r
    img[:, :, 1] = 0.5 * (raw[0::2, 1::2] + raw[1::2, 0::2])  # g
    img[:, :, 2] = raw[1::2, 1::2]  # b

    # img = (img - 0.5) * 2

    return img


@ex.config
def config():
    batch_size = 1


@ex.automain
def main(_run):
    args = tupperware(_run.config)
    from dataloader import PhaseMaskADMMTestDataset

    dataset = PhaseMaskADMMTestDataset(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    admm = ADMM_Net(args).to(device)

    # img = torch.rand(3, 1280, 1408).to(device)

    img = dataset._img_load("data/phase_images/n09193705_8254.png").float()
    out = admm(img.unsqueeze(0).to(device))

    out = out[0].permute(1, 2, 0).cpu().detach().numpy()
    out = (out - out.min()) / (out.max() - out.min())

    plt.imshow(out)
    plt.show()
