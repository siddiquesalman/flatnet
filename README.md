# flatnet-separable
This repo contains the test code for the lensless reconstruction algorithm (Proposed-T and Proposed-R) proposed in ICCV 2019 paper **Towards Photorealistic Reconstruction of Highly Multiplexed Lensless Images**[PDF](http://openaccess.thecvf.com/content_ICCV_2019/papers/Khan_Towards_Photorealistic_Reconstruction_of_Highly_Multiplexed_Lensless_Images_ICCV_2019_paper.pdf)

The best way to make sure all dependencies are installed is by installing Anaconda followed by PyTorch. 
It has been tested on PyTorch version 0.4.0 and above.

Once the dependencies are installed, open Jupyter and use the notebook **FlatNet-separable.ipynb** to evaluate flatnet on captured measurements. 

Pretrained models can be found at :[Link](https://www.dropbox.com/sh/1p9n1mclkhlx074/AADj4fLZQaFrH1y-aAnF40Bda?dl=0)

Full dataset used for the paper is available at:[Link](https://www.dropbox.com/sh/pzmhwh1bjhn86l0/AABix6OgyENxBDGXHFuMeBSfa?dl=0)

Example data is provided in the directory **example_data**. It contains some measurements along with their Tikhonov reconstructions. You can use these measurements to test the reconstruction as well without having to download the whole dataset. 'fc_x.png' refers to the measurement while 'rec_x.png' refers to the corresponding Tikhonov reconstruction. 


