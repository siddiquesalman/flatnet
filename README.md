# flatnet-separable
This repo contains the train and test code for the lensless reconstruction algorithm (Proposed-T and Proposed-R) proposed in ICCV 2019 paper **Towards Photorealistic Reconstruction of Highly Multiplexed Lensless Images** [[PDF]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Khan_Towards_Photorealistic_Reconstruction_of_Highly_Multiplexed_Lensless_Images_ICCV_2019_paper.pdf)

The best way to make sure all dependencies are installed is by installing Anaconda followed by PyTorch. 
It has been tested on PyTorch version 0.4.0 and above.

Once the dependencies are installed, to run the test script, open Jupyter and use the notebook **FlatNet-separable.ipynb** to evaluate flatnet on captured measurements. 

Pretrained models can be found at : [[Link]](https://www.dropbox.com/sh/1p9n1mclkhlx074/AADj4fLZQaFrH1y-aAnF40Bda?dl=0)

Full dataset used for the paper is available at: [[Dropbox]](https://www.dropbox.com/sh/pzmhwh1bjhn86l0/AABix6OgyENxBDGXHFuMeBSfa?dl=0) or [[G-Drive]](https://drive.google.com/drive/folders/1nyng6spi7SQRZb_1zEkOScOIPEI9DCUL?usp=sharing)

Example data is provided in the directory **example_data**. It contains some measurements along with their Tikhonov reconstructions. You can use these measurements to test the reconstruction as well without having to download the whole dataset. 'fc_x.png' refers to the measurement while 'rec_x.png' refers to the corresponding Tikhonov reconstruction. 


**TRAINING FROM SCRATCH**

Please run **main.py** to train from scratch

Alternatively run the shell script **flatnet.sh** found in execs directory with desired arguments.

Please make sure your path is set properly for the dataset and saving models. For saving model, make sure the variable 'data' in main.py is appropriately changed. For dataset, make sure the variable 'temp' in dataloader.py is changed appropriately.

**REGARDING INITIALIZATIONS**

Transpose Initializations:
'flatcam_prototype2_calibdata.mat' found in the data folder contains the calibration matrices : Phi_L and Phi_R. They are named as P1 and Q1 respectively once you load the mat file. Please note that there are separate P1 and Q1 for each channel (b,gr,gb,r). For the paper, we use only one of them (P1b and Q1b) for initializing the weights (W_1 and W_2) of trainable inversion layer.


Random Toeplitz Initializations:
'phil_toep_slope22.mat' and 'phir_toep_slope22' found in the data folder contain the random toeplitz matrices corresponding to W_1 and W_2 of the trainable inversion layer. 

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
In case of any queries, please reach out to me at salmansiddique.khan@gmail.com
