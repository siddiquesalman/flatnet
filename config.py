"""
Convention

ours/naive-fft-(fft_h-fft_w)-learn-(learn_h-learn_w)-meas-(meas_h-meas_w)-kwargs

* Phlatcam: 1518 x 2012 (post demosiacking)
"""
from pathlib import Path
import torch


def base_config():
    exp_name = "ours-fft-1280-1408-learn-1280-1408-meas-1280-1408"
    is_admm = "admm" in exp_name
    is_naive = "naive" in exp_name

    # ---------------------------------------------------------------------------- #
    # Directories
    # ---------------------------------------------------------------------------- #

    test_glob_pattern = "test_set_Jan/cap*.png"

    image_dir = Path("data")
    output_dir = Path("output_phase_mask_Feb_2020_size_384") / exp_name
    ckpt_dir = Path("ckpts_phase_mask_Feb_2020_size_384") / exp_name
    run_dir = Path("runs_phase_mask_Feb_2020_size_384") / exp_name  # Tensorboard
    test_image_dir = image_dir / "PhaseCapture_Webcam" / "saves"

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    text_file_dir = image_dir / "text_files"
    train_source_list = text_file_dir / "train_source_imagenet_384_384_Feb_19.txt"
    train_target_list = text_file_dir / "train_target.txt"

    val_source_list = text_file_dir / "val_source_imagenet_384_384_Feb_19.txt"
    val_target_list = text_file_dir / "val_target.txt"

    static_val_image = "n02980441_9185.png"
    static_test_image = "Image__2020-01-16__23-05-28.raw.png"

    test_skip_existing = True
    test_apply_gain = True

    dataset_name = "phase_mask" if not is_admm else "phase_mask_admm"

    shuffle = True
    train_gaussian_noise = 5e-3

    # ---------------------------------------------------------------------------- #
    # Image, Meas, PSF Dimensions
    # ---------------------------------------------------------------------------- #
    # PSF
    psf_mat = Path("data/phase_psf/psf.npy")

    meas_height = 1518  # original height of meas
    meas_width = 2012  # original width of meas
    meas_crop_size_x = 1280  # Meas crop. If 0, assume no crop
    meas_crop_size_y = 1408
    meas_centre_x = 808
    meas_centre_y = 965

    psf_height = 1280  # fft layer height
    psf_width = 1408  # fft layer width
    psf_crop_size_x = 1280
    psf_crop_size_y = 1408
    psf_centre_x = meas_centre_x
    psf_centre_y = meas_centre_y

    fft_gamma = 50000

    # pad meas
    pad_meas_mode = "replicate" if not is_admm else "constant"  # If none, no padding

    # Mask
    use_mask = False
    mask_path = Path("data/phase_psf/box_gaussian_1280_1408.npy")

    image_height = 384
    image_width = 384

    batch_size = 5
    num_threads = batch_size  # parallel workers

    # ---------------------------------------------------------------------------- #
    # Train Configs
    # ---------------------------------------------------------------------------- #
    # Schedules
    num_epochs = 31
    fft_epochs = num_epochs if is_naive else 0

    learning_rate = 3e-4
    fft_learning_rate = 4e-10 if not is_admm else 3e-5

    # Betas for AdamW. We follow https://arxiv.org/pdf/1704.00028
    beta_1 = 0.9  # momentum
    beta_2 = 0.999

    lr_scheduler = "cosine"  # or step

    # Cosine annealing
    T_0 = 1
    T_mult = 2
    step_size = 2  # For step lr

    # saving models
    save_filename_G = "model.pth"
    save_filename_FFT = "FFT.pth" if not is_admm else "ADMM.pth"
    save_filename_D = "D.pth"

    save_filename_latest_G = "model_latest.pth"
    save_filename_latest_FFT = "FFT_latest.pth" if not is_admm else "ADMM_latest.pth"
    save_filename_latest_D = "D_latest.pth"

    log_interval = 150  # the number of iterations (default: 10) to print at
    save_ckpt_interval = log_interval * 10
    save_copy_every_epochs = 10
    # ---------------------------------------------------------------------------- #
    # Model
    # ---------------------------------------------------------------------------- #
    # See models/get_model.py for registry
    model = "unet-128-pixelshuffle-invert"
    pixelshuffle_ratio = 2

    # admm model args
    admm_iterations = 5
    normalise_admm_psf = False

    G_finetune_layers = []  # None implies all

    num_groups = 8  # Group norm

    # ---------------------------------------------------------------------------- #
    # Loss
    # ---------------------------------------------------------------------------- #
    lambda_adversarial = 0.6
    lambda_contextual = 0.0
    lambda_perception = 1.2  # 0.006
    lambda_image = 1  # mse

    resume = True
    finetune = False  # Wont load loss or epochs

    # ---------------------------------------------------------------------------- #
    # Inference Args
    # ---------------------------------------------------------------------------- #
    inference_mode = "latest"
    assert inference_mode in ["latest", "best"]

    # ---------------------------------------------------------------------------- #
    # Distribution Args
    # ---------------------------------------------------------------------------- #
    # choose cpu or cuda:0 device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    distdataparallel = False


def ours_meas_1280_1408():
    exp_name = "ours-fft-1280-1408-learn-1280-1408-meas-1280-1408"


def ours_meas_1280_1408_unet_64():
    exp_name = "ours-fft-1280-1408-learn-1280-1408-meas-1280-1408-unet-64"
    model = "unet-64-pixelshuffle-invert"


def ours_meas_1280_1408_unet_32():
    exp_name = "ours-fft-1280-1408-learn-1280-1408-meas-1280-1408-unet-32"
    model = "unet-32-pixelshuffle-invert"


def ours_meas_1280_1408_simulated():
    exp_name = "ours-fft-1280-1408-learn-1280-1408-meas-1280-1408-simulated"
    psf_mat = Path("data/phase_psf/sim_psf.npy")


def ours_meas_990_1254():
    exp_name = "ours-fft-1280-1408-learn-1280-1408-meas-990-1254"

    meas_crop_size_x = 990
    meas_crop_size_y = 1254

    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_1280_1408_big_mask.npy")


def ours_meas_990_1254_simulated():
    exp_name = "ours-fft-1280-1408-learn-1280-1408-meas-990-1254-simulated"
    psf_mat = Path("data/phase_psf/sim_psf.npy")

    meas_crop_size_x = 990
    meas_crop_size_y = 1254

    use_mask = True


def ours_meas_864_1120():
    exp_name = "ours-fft-1280-1408-learn-1280-1408-meas-864-1120-big-mask"

    meas_crop_size_x = 864
    meas_crop_size_y = 1120

    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_1280_1408_big_mask.npy")


def ours_meas_864_1120_simulated():
    exp_name = "ours-fft-1280-1408-learn-1280-1408-meas-864-1120-simulated"
    psf_mat = Path("data/phase_psf/sim_psf.npy")

    meas_crop_size_x = 864
    meas_crop_size_y = 1120

    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_256x256_1280_1408.npy")


def ours_meas_608_864():
    exp_name = "ours-fft-1280-1408-learn-1280-1408-meas-608-864-big-mask"

    meas_crop_size_x = 608
    meas_crop_size_y = 864

    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_1280_1408_big_mask.npy")


def ours_meas_608_864_simulated():
    exp_name = "ours-fft-1280-1408-learn-1280-1408-meas-608-864-simulated-big-mask"
    psf_mat = Path("data/phase_psf/sim_psf.npy")

    meas_crop_size_x = 608
    meas_crop_size_y = 864

    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_1280_1408_big_mask.npy")


def ours_meas_512_640():
    exp_name = "ours-fft-1280-1408-learn-1280-1408-meas-512-640-big-mask"

    meas_crop_size_x = 512
    meas_crop_size_y = 640

    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_1280_1408_big_mask.npy")


def ours_meas_512_640_simulated():
    exp_name = "ours-fft-1280-1408-learn-1280-1408-meas-512-640-simulated"
    psf_mat = Path("data/phase_psf/sim_psf.npy")

    meas_crop_size_x = 512
    meas_crop_size_y = 640

    use_mask = True


def ours_meas_400_400():
    exp_name = "ours-fft-1280-1408-learn-1280-1408-meas-400-400-big-mask"

    meas_crop_size_x = 400
    meas_crop_size_y = 400

    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_1280_1408_big_mask.npy")


def ours_meas_1280_1408_finetune_dualcam_1cap():
    exp_name = (
        "ours-fft-1280-1408-learn-1280-1408-meas-1280-1408-finetune-dualcam-fullG-1cap"
    )

    num_epochs = 63
    learning_rate = 3e-6

    static_val_image = "multicap_28.png"
    static_test_image = "test_set_Jan/output_cap_Image__2020-01-16__22-56-05.raw.png"
    image_dir = Path("data")
    text_file_dir = image_dir / "text_files"
    train_source_list = text_file_dir / "train_meas_1cap_indoor_dualcam.txt"
    train_target_list = text_file_dir / "train_webcam_indoor_dualcam.txt"
    val_source_list = text_file_dir / "val_meas_3cap_indoor_dualcam.txt"
    val_target_list = text_file_dir / "val_webcam_indoor_dualcam.txt"

    # Loss
    lambda_adversarial = 0.0
    lambda_contextual = 1
    lambda_perception = 0.0  # 0.006
    lambda_image = 0.0  # mse


def ours_meas_608_864_finetune_dualcam_1cap():
    exp_name = (
        "ours-fft-1280-1408-learn-1280-1408-meas-608-864-finetune-dualcam-fullG-1cap"
    )

    num_epochs = 127

    batch_size = 6
    learning_rate = 3e-6
    meas_crop_size_x = 608
    meas_crop_size_y = 864

    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_1280_1408_big_mask.npy")

    static_val_image = "multicap_28.png"
    static_test_image = "test_set_Jan/output_cap_Image__2020-01-16__22-56-05.raw.png"
    image_dir = Path("data")
    text_file_dir = image_dir / "text_files"
    train_source_list = text_file_dir / "train_meas_1cap_indoor_dualcam.txt"
    train_target_list = text_file_dir / "train_webcam_indoor_dualcam.txt"
    val_source_list = text_file_dir / "val_meas_3cap_indoor_dualcam.txt"
    val_target_list = text_file_dir / "val_webcam_indoor_dualcam.txt"

    # Loss
    lambda_adversarial = 0.0
    lambda_contextual = 1
    lambda_perception = 0.0  # 0.006
    lambda_image = 0.0  # mse

    # Mask
    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_1280_1408_big_mask.npy")


def ours_meas_990_1254_finetune_dualcam_1cap():
    exp_name = (
        "ours-fft-1280-1408-learn-1280-1408-meas-990-1254-finetune-dualcam-fullG-1cap"
    )

    num_epochs = 127

    batch_size = 6
    learning_rate = 3e-6
    meas_crop_size_x = 990
    meas_crop_size_y = 1254

    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_1280_1408_big_mask.npy")

    static_val_image = "multicap_28.png"
    static_test_image = "test_set_Jan/output_cap_Image__2020-01-16__22-56-05.raw.png"
    image_dir = Path("data")
    text_file_dir = image_dir / "text_files"
    train_source_list = text_file_dir / "train_meas_1cap_indoor_dualcam.txt"
    train_target_list = text_file_dir / "train_webcam_indoor_dualcam.txt"
    val_source_list = text_file_dir / "val_meas_3cap_indoor_dualcam.txt"
    val_target_list = text_file_dir / "val_webcam_indoor_dualcam.txt"

    # Loss
    lambda_adversarial = 0.0
    lambda_contextual = 1
    lambda_perception = 0.0  # 0.006
    lambda_image = 0.0  # mse

    # Mask
    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_1280_1408_big_mask.npy")


def naive_meas_1280_1408():
    exp_name = "naive-fft-1280-1408-learn-1280-1408-meas-1280-1408"


def naive_meas_1280_1408_unet_64():
    exp_name = "naive-fft-1280-1408-learn-1280-1408-meas-1280-1408-unet-64"
    model = "unet-64-pixelshuffle-invert"


def naive_meas_1280_1408_unet_32():
    exp_name = "naive-fft-1280-1408-learn-1280-1408-meas-1280-1408-unet-32"
    model = "unet-32-pixelshuffle-invert"


def naive_meas_1280_1408_simulated():
    exp_name = "naive-fft-1280-1408-learn-1280-1408-meas-1280-1408-simulated"
    psf_mat = Path("data/phase_psf/sim_psf.npy")


def naive_meas_990_1254():
    exp_name = "naive-fft-1280-1408-learn-1280-1408-meas-990-1254-big-mask"

    meas_crop_size_x = 990
    meas_crop_size_y = 1254

    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_1280_1408_big_mask.npy")


def naive_meas_990_1254_simulated():
    exp_name = "naive-fft-1280-1408-learn-1280-1408-meas-990-1254-simulated"
    psf_mat = Path("data/phase_psf/sim_psf.npy")

    meas_crop_size_x = 990
    meas_crop_size_y = 1254

    use_mask = True


def naive_meas_864_1120():
    exp_name = "naive-fft-1280-1408-learn-1280-1408-meas-864-1120-big-mask"

    meas_crop_size_x = 864
    meas_crop_size_y = 1120

    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_1280_1408_big_mask.npy")


def naive_meas_864_1120_simulated():
    exp_name = "naive-fft-1280-1408-learn-1280-1408-meas-864-1120-simulated"
    psf_mat = Path("data/phase_psf/sim_psf.npy")

    meas_crop_size_x = 864
    meas_crop_size_y = 1120

    use_mask = True


def naive_meas_608_864():
    exp_name = "naive-fft-1280-1408-learn-1280-1408-meas-608-864-big-mask"

    meas_crop_size_x = 608
    meas_crop_size_y = 864

    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_1280_1408_big_mask.npy")


def naive_meas_608_864_simulated():
    exp_name = "naive-fft-1280-1408-learn-1280-1408-meas-608-864-simulated-big-mask"

    meas_crop_size_x = 608
    meas_crop_size_y = 864

    psf_mat = Path("data/phase_psf/sim_psf.npy")

    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_1280_1408_big_mask.npy")


def naive_meas_512_640():
    exp_name = "naive-fft-1280-1408-learn-1280-1408-meas-512-640-big-mask"

    meas_crop_size_x = 512
    meas_crop_size_y = 640

    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_1280_1408_big_mask.npy")


def naive_meas_512_640_simulated():
    exp_name = "naive-fft-1280-1408-learn-1280-1408-meas-512-640-simulated"

    meas_crop_size_x = 512
    meas_crop_size_y = 640

    psf_mat = Path("data/phase_psf/sim_psf.npy")

    use_mask = True


def naive_meas_400_400():
    exp_name = "naive-fft-1280-1408-learn-1280-1408-meas-400-400-big-mask"

    meas_crop_size_x = 400
    meas_crop_size_y = 400

    use_mask = True
    mask_path = Path("data/phase_psf/box_gaussian_1280_1408_big_mask.npy")


def le_admm_meas_1280_1408():
    exp_name = "le-admm-fft-1280-1408-learn-1280-1408-meas-1280-1408"


def le_admm_meas_1280_1408_unet_64():
    exp_name = "le-admm-fft-1280-1408-learn-1280-1408-meas-1280-1408-unet-64"
    model = "unet-64-pixelshuffle-invert"


def le_admm_meas_1280_1408_unet_32():
    exp_name = "le-admm-fft-1280-1408-learn-1280-1408-meas-1280-1408-unet-32"
    model = "unet-32-pixelshuffle-invert"


def le_admm_meas_1280_1408_simulated():
    exp_name = "le-admm-fft-1280-1408-learn-1280-1408-meas-1280-1408-simulated"

    psf_mat = Path("data/phase_psf/sim_psf.npy")


def le_admm_meas_990_1254():
    exp_name = "le-admm-fft-1280-1408-learn-1280-1408-meas-990-1254"

    meas_crop_size_x = 990
    meas_crop_size_y = 1254

    normalise_admm_psf = True


def le_admm_meas_990_1254_simulated():
    exp_name = "le-admm-fft-1280-1408-learn-1280-1408-meas-990-1254-simulated"
    meas_crop_size_x = 990
    meas_crop_size_y = 1254

    normalise_admm_psf = True
    psf_mat = Path("data/phase_psf/sim_psf.npy")


def le_admm_meas_864_1120():
    exp_name = "le-admm-fft-1280-1408-learn-1280-1408-meas-864-1120"

    meas_crop_size_x = 840
    meas_crop_size_y = 1120

    normalise_admm_psf = True


def le_admm_meas_864_1120_simulated():
    exp_name = "le-admm-fft-1280-1408-learn-1280-1408-meas-864-1120-simulated"
    meas_crop_size_x = 840
    meas_crop_size_y = 1120

    normalise_admm_psf = True
    psf_mat = Path("data/phase_psf/sim_psf.npy")


def le_admm_meas_608_864():
    exp_name = "le-admm-fft-1280-1408-learn-1280-1408-meas-608-864"
    meas_crop_size_x = 608
    meas_crop_size_y = 864

    normalise_admm_psf = True


def le_admm_meas_608_864_simulated():
    exp_name = "le-admm-fft-1280-1408-learn-1280-1408-meas-608-864-simulated"
    meas_crop_size_x = 608
    meas_crop_size_y = 864

    normalise_admm_psf = True
    psf_mat = Path("data/phase_psf/sim_psf.npy")


def le_admm_meas_512_640():
    exp_name = "le-admm-fft-1280-1408-learn-1280-1408-meas-512-640"
    meas_crop_size_x = 512
    meas_crop_size_y = 640

    normalise_admm_psf = True


def le_admm_meas_512_640_simulated():
    exp_name = "le-admm-fft-1280-1408-learn-1280-1408-meas-512-640-simulated"
    meas_crop_size_x = 608
    meas_crop_size_y = 864

    normalise_admm_psf = True
    psf_mat = Path("data/phase_psf/sim_psf.npy")


def le_admm_meas_400_400():
    exp_name = "le-admm-fft-1280-1408-learn-1280-1408-meas-512-640"
    meas_crop_size_x = 400
    meas_crop_size_y = 400

    normalise_admm_psf = True


named_config_ll = [
    # Ours
    ours_meas_1280_1408,
    ours_meas_1280_1408_simulated,
    ours_meas_990_1254,
    ours_meas_990_1254_simulated,
    ours_meas_864_1120,
    ours_meas_864_1120_simulated,
    ours_meas_608_864,
    ours_meas_608_864_simulated,
    ours_meas_512_640,
    ours_meas_512_640_simulated,
    ours_meas_400_400,
    # Naive
    naive_meas_1280_1408,
    naive_meas_1280_1408_simulated,
    naive_meas_990_1254,
    naive_meas_990_1254_simulated,
    naive_meas_864_1120,
    naive_meas_864_1120_simulated,
    naive_meas_608_864,
    naive_meas_608_864_simulated,
    naive_meas_512_640,
    naive_meas_512_640_simulated,
    naive_meas_400_400,
    # Le ADMM
    le_admm_meas_1280_1408,
    le_admm_meas_1280_1408_simulated,
    le_admm_meas_990_1254,
    le_admm_meas_990_1254_simulated,
    le_admm_meas_864_1120,
    le_admm_meas_864_1120_simulated,
    le_admm_meas_608_864,
    le_admm_meas_608_864_simulated,
    le_admm_meas_512_640,
    le_admm_meas_512_640_simulated,
    le_admm_meas_400_400,
    # Finetune
    ours_meas_1280_1408_finetune_dualcam_1cap,
    ours_meas_608_864_finetune_dualcam_1cap,
    ours_meas_990_1254_finetune_dualcam_1cap,
    # Unet 64
    ours_meas_1280_1408_unet_64,
    naive_meas_1280_1408_unet_64,
    le_admm_meas_1280_1408_unet_64,
    # Unet 32
    ours_meas_1280_1408_unet_32,
    naive_meas_1280_1408_unet_32,
    le_admm_meas_1280_1408_unet_32,
]


def initialise(ex):
    ex.config(base_config)
    for named_config in named_config_ll:
        ex.named_config(named_config)
    return ex


if __name__ == "__main__":
    str_named_config_ll = [str(named_config) for named_config in named_config_ll]
    print("\n".join(str_named_config_ll))
