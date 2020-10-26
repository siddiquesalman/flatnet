# FlatNet
This repo contains the train and test code for the lensless reconstruction algorithm proposed in the ICCV 2019 paper **Towards Photorealistic Reconstruction of Highly Multiplexed Lensless Images** [[PDF]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Khan_Towards_Photorealistic_Reconstruction_of_Highly_Multiplexed_Lensless_Images_ICCV_2019_paper.pdf) and the IEEE TPAMI 2020 paper 
**FlatNet: Towards Photorealistic Scene Reconstruction from Lensless Measurements**.

![Method Diagram](images/fig2_9Apr.jpg)

## Colab Notebooks

* FlatNet-Sep (separable model): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/varun19299/flatnet-gen/blob/master/explore_flatnet_gen.ipynb)

* FlatNet-Gen (general model): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/varun19299/flatnet-gen/blob/master/explore_flatnet_gen.ipynb)

## Requirements

* python 3.7+
* pytorch 1.6+
* `pip install -r requirements.txt`

# FlatNet-sep

<details>
<summary>Show all details</summary>
To run the test script, open Jupyter and use the notebook `FlatNet-separable.ipynb` to evaluate flatnet-separable on captured measurements.

Pretrained models can be found at : [[Dropbox]](https://www.dropbox.com/sh/1p9n1mclkhlx074/AADj4fLZQaFrH1y-aAnF40Bda?dl=0)

Full dataset used for the paper is available at: [[Dropbox]](https://www.dropbox.com/sh/pzmhwh1bjhn86l0/AABix6OgyENxBDGXHFuMeBSfa?dl=0) or [[G-Drive]](https://drive.google.com/drive/folders/1nyng6spi7SQRZb_1zEkOScOIPEI9DCUL?usp=sharing)

Example data is provided in the directory `example_data`. It contains some measurements along with their Tikhonov reconstructions. You can use these measurements to test the reconstruction as well without having to download the whole dataset. `fc_x.png` refers to the measurement while `rec_x.png` refers to the corresponding Tikhonov reconstruction. 


### Training From Scratch

Please run **main.py** to train from scratch

Alternatively, run the shell script **flatnet.sh** found in execs directory with desired arguments.

Please make sure your path is set properly for the dataset and saving models. For saving model, make sure the variable 'data' in main.py and for dataset, make sure the variable 'temp' in dataloader.py are changed appropriately.

### Regarding Initializations

* **Transpose Initializations:**
`flatcam_prototype2_calibdata.mat` found in the data folder contains the calibration matrices : Phi_L and Phi_R. They are named as P1 and Q1 respectively once you load the mat file. Please note that there are separate P1 and Q1 for each channel (b,gr,gb,r). For the paper, we use only one of them (P1b and Q1b) for initializing the weights (W_1 and W_2) of trainable inversion layer.


* **Random Toeplitz Initializations:**
`phil_toep_slope22.mat` and `phir_toep_slope22` found in the data folder contain the random toeplitz matrices corresponding to W_1 and W_2 of the trainable inversion layer. 

</details>


## FlatNet-Gen

Switch to the [`flatnet-gen`]() branch first.

<details>
<summary>Show all details</summary>
## Data, PSFs and Checkpoint

* Download data as [imagenet_caps_384_12bit_Feb_19](https://drive.google.com/open?id=1TTiQbIX_a880slUk4US32wovfqhsIYpd&authuser=ee16b068%40smail.iitm.ac.in&usp=drive_fs) and place under `data` (or symlink it).
* Download Point Spread Function(s) and Mask(s) as [phase_psf](https://drive.google.com/open?id=1BbotgTN4I2kGanWV130dLWwxalODo-FG&authuser=ee16b068%40smail.iitm.ac.in&usp=drive_fs) and place under `data` (or symlink it).
* Download checkpoints from [ckpts_phase_mask_Feb_2020_size_384](https://drive.google.com/open?id=159MsGGakny59MSXuynHMYSWiaq73o4af&authuser=ee16b068%40smail.iitm.ac.in&usp=drive_fs) and place as `ckpts_phase_mask_Feb_2020_size_384`.

You should then have the following directory structure:

```bash
.
|-- ckpts_phase_mask_Feb_2020_size_384
|   |-- ours-fft-1280-1408-learn-1280-1408-meas-1280-1408
|   `-- le-admm-fft-1280-1408-learn-1280-1408-meas-1280-1408
|-- data
|   |-- imagenet_caps_384_12bit_Feb_19
|   `-- phase_psf
```

## Streamlit Server

Run as: `streamlit run test_streamlit.py`

Streamlit is an actively developed package, and while we install the latest version in this project, please note that backward compatibility may break in upcoming months. 
Nevertheless, we shall try to keep `test_streamlit.py` updated to reflect these changes.

## Train Script

Run as:

```bash
python train.py with ours_meas_1280_1408 -p
```

Here, `ours_meas_1280_1408` is a config function, defined in `config.py`, where you can also find an exhaustive list of other configs available.
For a multi-gpu version (we use pytorch's `distdataparallel`):

```bash
python -m torch.distributed.launch --nproc_per_node=3 --use_env train.py with ours_meas_1280_1408 distdataparallel=True -p
```

## Val Script

```bash
python val.py with ours_meas_1280_1408 -p
```

Metrics and Outputs are writen at `output_dir/exp_name/`.

## Configs

See `config.py` for exhaustive set of config options. Create a new function to add a configuration.

|       Model      |                      Calibrated PSF Config                     |                  Simulated PSF Config                 |
|:----------------:|:--------------------------------------------------------------:|:-----------------------------------------------------:|
|      _Ours_      |            ours_meas_{sensor_height}_{sensor_width}            |   ours_meas_{sensor_height}_{sensor_width}_simulated  |
| _Ours Finetuned_ | ours_meas_{sensor_height}_{sensor_width}_finetune_dualcam_1cap |                           NA                          |

Here, (sensor_height,sensor_width) can be (1280, 1408), (990, 1254), (864, 1120), (608, 864), (512, 640), (400, 400).

Finetuned refers to finetuning with contextual loss on indoor measurements (see our paper for more details).

</details>


## Citation

If you use this code, please cite our work:
```
@inproceedings{khan2019towards,
  title={Towards photorealistic reconstruction of highly multiplexed lensless images},
  author={Khan, Salman S and Adarsh, VR and Boominathan, Vivek and Tan, Jasper and Veeraraghavan, Ashok and Mitra, Kaushik},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={7860--7869},
  year={2019}
}
```

And our more recent TPAMI paper:

```

```

## Contact Us

In case of any queries, please reach out to salmansiddique.khan@gmail.com
