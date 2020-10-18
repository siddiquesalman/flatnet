import numpy as np
import skimage
import torch
from PIL import Image
import torchvision
import torch.nn.functional as F
from io import BytesIO


def demosaic_raw(meas):
	tform = skimage.transform.SimilarityTransform(rotation=0.00174)
	X = meas.numpy()[0,:,:]
	X = X/65535.0
	X=X+0.003*np.random.randn(X.shape[0],X.shape[1])
	im1=np.zeros((512,640,4))
	im1[:,:,0]=X[0::2, 0::2]#b
	im1[:,:,1]=X[0::2, 1::2]#gb
	im1[:,:,2]=X[1::2, 0::2]#gr
	im1[:,:,3]=X[1::2, 1::2]#r
	im1=skimage.transform.warp(im1,tform)
	im=im1[6:506,10:630,:]      
	rowMeans = im.mean(axis=1, keepdims=True)
	colMeans = im.mean(axis=0, keepdims=True)
	allMean = rowMeans.mean()
	im = im - rowMeans - colMeans + allMean
	im = im.astype('float32')
	meas = torch.from_numpy(np.swapaxes(np.swapaxes(im,0,2),1,2)).unsqueeze(0)
	return meas[0,:,:,:]
def demosaic_raw_fcfd(meas):
	tform = skimage.transform.SimilarityTransform(rotation=0.0008726)
	X = meas.numpy()[0,:,:]
	X = X/65535.0
	X=X+0.003*np.random.randn(X.shape[0],X.shape[1])
	im1=np.zeros((512,640,4))
	im1[:,:,0]=X[0::2, 0::2]#b
	im1[:,:,1]=X[0::2, 1::2]#gb
	im1[:,:,2]=X[1::2, 0::2]#gr
	im1[:,:,3]=X[1::2, 1::2]#r
	im1=skimage.transform.warp(im1,tform)
	im=im1[6:506,10:630,:]      
	rowMeans = im.mean(axis=1, keepdims=True)
	colMeans = im.mean(axis=0, keepdims=True)
	allMean = rowMeans.mean()
	im = im - rowMeans - colMeans + allMean
	im = im.astype('float32')
	meas = torch.from_numpy(np.swapaxes(np.swapaxes(im,0,2),1,2)).unsqueeze(0)
	return meas[0,:,:,:]

def demosaic_raw_basler(meas):
	tform = skimage.transform.SimilarityTransform(rotation=0.00174)
	X = meas.numpy()[0,:,:]
	X = X/4095.0
	X=X+0.003*np.random.randn(X.shape[0],X.shape[1])
	im1=np.zeros((1518,2012,4))
	im1[:,:,0]=X[1::2, 1::2]#b
	im1[:,:,1]=X[0::2, 1::2]#gb
	im1[:,:,2]=X[1::2, 0::2]#gr
	im1[:,:,3]=X[0::2, 0::2]#r
	im=im1[9:1509,256:1756,:]      
	rowMeans = im.mean(axis=1, keepdims=True)
	colMeans = im.mean(axis=0, keepdims=True)
	allMean = rowMeans.mean()
	im = im - rowMeans - colMeans + allMean
	im = im.astype('float32')
	meas = torch.from_numpy(np.swapaxes(np.swapaxes(im,0,2),1,2)).unsqueeze(0)
	return meas[0,:,:,:]


def demosaic_jpeg(meas,qf):
	tform = skimage.transform.SimilarityTransform(rotation=0.00174)
	X = meas.numpy()[0,:,:]
	# print(X.min())
	X = X/65535.0
	X=X+0.003*np.random.randn(X.shape[0],X.shape[1])
	im1=np.zeros((512,640,4))
	im1[:,:,0]=X[0::2, 0::2]#b
	im1[:,:,1]=X[0::2, 1::2]#gb
	im1[:,:,2]=X[1::2, 0::2]#gr
	im1[:,:,3]=X[1::2, 1::2]#r
	im1=skimage.transform.warp(im1,tform)
	im=im1[6:506,10:630,:]
	meas1 = Image.fromarray(np.uint8(im[:,:,0]*255))
	meas2 = Image.fromarray(np.uint8(im[:,:,1]*255))
	meas3 = Image.fromarray(np.uint8(im[:,:,2]*255))
	meas4 = Image.fromarray(np.uint8(im[:,:,3]*255))
	meas1 = toJPEG(meas1,qf)
	meas2 = toJPEG(meas2,qf)
	meas3 = toJPEG(meas3,qf)
	meas4 = toJPEG(meas4,qf)
	meas1 = torchvision.transforms.ToTensor()(meas1)[0,:,:]
	meas2 = torchvision.transforms.ToTensor()(meas2)[0,:,:]
	meas3 = torchvision.transforms.ToTensor()(meas3)[0,:,:]
	meas4 = torchvision.transforms.ToTensor()(meas4)[0,:,:]
	# print(meas1.unsqueeze(2).shape)
	measo = torch.cat((meas1.unsqueeze(2),meas2.unsqueeze(2),meas3.unsqueeze(2),meas4.unsqueeze(2)),2)
	im = measo.numpy()
	# print(im.shape)
	rowMeans = im.mean(axis=1, keepdims=True)
	colMeans = im.mean(axis=0, keepdims=True)
	allMean = rowMeans.mean()
	im = im - rowMeans - colMeans + allMean
	im = im.astype('float32')
	meas = torch.from_numpy(np.swapaxes(np.swapaxes(im,0,2),1,2)).unsqueeze(0)
	return meas[0,:,:,:]


class DatasetFromFilenames:

	def __init__(self, filenames_loc_meas, filenames_loc_orig):
		self.filenames_meas = filenames_loc_meas
		self.paths_meas = get_paths(self.filenames_meas)
		self.filenames_orig = filenames_loc_orig
		self.paths_orig = get_paths(self.filenames_orig)
		self.num_im = len(self.paths_meas)
		self.totensor = torchvision.transforms.ToTensor()
		self.resize = torchvision.transforms.Resize((256,256))
		

	def __len__(self):
		return len(self.paths_meas)

	def __getitem__(self, index):
		# obtain the image paths
#         print(index)
		im_path = self.paths_orig[index % self.num_im]
		meas_path = self.paths_meas[index % self.num_im]
		# load images (grayscale for direct inference)
		im = Image.open(im_path)
		im = im.convert('RGB')
		im = self.resize(im)
		# print(im.size)
		im = self.totensor(im)

		meas = Image.open(meas_path)
		meas = self.totensor(meas)
		# print(meas.shape)
		meas = demosaic_raw(meas)

		# print(im_label.max())
		# print(torch.max(im_label))
		# print(meas.shape)
		# print(im.shape)
		return meas,im

class DatasetFromFilenamesFCFD:

	def __init__(self, filenames_loc_meas, filenames_loc_orig):
		self.filenames_meas = filenames_loc_meas
		self.paths_meas = get_paths(self.filenames_meas)
		self.filenames_orig = filenames_loc_orig
		self.paths_orig = get_paths(self.filenames_orig)
		self.num_im = len(self.paths_meas)
		self.totensor = torchvision.transforms.ToTensor()
		self.resize = torchvision.transforms.Resize((256,256))
		

	def __len__(self):
		return len(self.paths_meas)

	def __getitem__(self, index):
		# obtain the image paths
#         print(index)
		im_path = self.paths_orig[index % self.num_im]
		meas_path = self.paths_meas[index % self.num_im]
		# load images (grayscale for direct inference)
		im = Image.open(im_path)
		im = im.convert('RGB')
		im = self.resize(im)
		# print(im.size)
		im = self.totensor(im)

		meas = Image.open(meas_path)
		meas = self.totensor(meas)
		# print(meas.shape)
		meas = demosaic_raw_fcfd(meas)

		# print(im_label.max())
		# print(torch.max(im_label))
		# print(meas.shape)
		# print(im.shape)
		return meas,im


class DatasetFromFilenamesTikFCFD:

	def __init__(self, filenames_loc_meas, filenames_loc_orig):
		self.filenames_meas = filenames_loc_meas
		self.paths_meas = get_paths(self.filenames_meas)
		self.filenames_orig = filenames_loc_orig
		self.paths_orig = get_paths(self.filenames_orig)
		self.num_im = len(self.paths_meas)
		self.totensor = torchvision.transforms.ToTensor()
		self.resize = torchvision.transforms.Resize((256,256))
		

	def __len__(self):
		return len(self.paths_meas)

	def __getitem__(self, index):
		# obtain the image paths
#         print(index)
		im_path = self.paths_orig[index % self.num_im]
		meas_path = self.paths_meas[index % self.num_im]
		# load images (grayscale for direct inference)
		im = Image.open(im_path)
		im = im.convert('RGB')
		im = self.resize(im)
		# print(im.size)
		im = self.totensor(im)

		meas = Image.open(meas_path)
		meas = meas.convert('RGB')
		meas = self.totensor(meas)
		# print(meas.shape)

		# print(im_label.max())
		# print(torch.max(im_label))
		# print(meas.shape)
		# print(im.shape)
		return meas,im
class DatasetFromFilenamesBasler:

	def __init__(self, filenames_loc_meas, filenames_loc_orig):
		self.filenames_meas = filenames_loc_meas
		self.paths_meas = get_paths(self.filenames_meas)
		self.filenames_orig = filenames_loc_orig
		self.paths_orig = get_paths(self.filenames_orig)
		self.num_im = len(self.paths_meas)
		self.totensor = torchvision.transforms.ToTensor()
		self.resize = torchvision.transforms.Resize((256,256))
		

	def __len__(self):
		return len(self.paths_meas)

	def __getitem__(self, index):
		# obtain the image paths
#         print(index)
		im_path = self.paths_orig[index % self.num_im]
		meas_path = self.paths_meas[index % self.num_im]
		# load images (grayscale for direct inference)
		im = Image.open(im_path)
		im = im.convert('RGB')
		im = self.resize(im)
		# print(im.size)
		im = self.totensor(im)

		meas = Image.open(meas_path)
		meas = self.totensor(meas)
		# print(meas.shape)
		meas = demosaic_raw_basler(meas)

		# print(im_label.max())
		# print(torch.max(im_label))
		# print(meas.shape)
		# print(im.shape)
		return meas,im

def JPEGcompression(image,qf):
	# qf = 99
	outputIoStream = BytesIO()
	image = torchvision.transforms.ToTensor()(image)
	image = image.numpy()
	image = (image/65535.0)
	image = image.astype('float32')
	image = torch.tensor(image)
	image = torchvision.transforms.ToPILImage()(image)
	image.convert('L').save(outputIoStream, "JPEG", quality=qf, optimize=True)
	outputIoStream.seek(0)
	return Image.open(outputIoStream)

def toJPEG(image,qf):
	# qf = 99
	outputIoStream = BytesIO()
	image = torchvision.transforms.ToTensor()(image)
	# image = image.unsqueeze(0)
	image = image.numpy()
	# image = (image/65535.0)
	image = image.astype('float32')
	image = torch.tensor(image)
	image = torchvision.transforms.ToPILImage()(image)
	image.convert('L').save(outputIoStream, "JPEG", quality=qf, optimize=True)
	outputIoStream.seek(0)
	return Image.open(outputIoStream)

	

class DatasetFromFilenamesJPEG:

	def __init__(self, filenames_loc_meas, filenames_loc_orig,qf):
		self.filenames_meas = filenames_loc_meas
		self.paths_meas = get_paths(self.filenames_meas)
		self.filenames_orig = filenames_loc_orig
		self.paths_orig = get_paths(self.filenames_orig)
		self.num_im = len(self.paths_meas)
		self.totensor = torchvision.transforms.ToTensor()
		self.resize = torchvision.transforms.Resize((256,256))
		self.toJPEG = torchvision.transforms.Lambda(JPEGcompression)
		self.qf = qf

	def __len__(self):
		return len(self.paths_meas)

	def __getitem__(self, index):
		# obtain the image paths
#         print(index)
		im_path = self.paths_orig[index % self.num_im]
		meas_path = self.paths_meas[index % self.num_im]
		# load images (grayscale for direct inference)
		im = Image.open(im_path)
		im = im.convert('RGB')
		im = self.resize(im)
		# print(im.size)
		im = self.totensor(im)

		meas = Image.open(meas_path)
		# meas = self.toJPEG(meas,qf)
		meas = self.totensor(meas)
		# print(meas.shape)
		meas = demosaic_jpeg(meas,self.qf)
		# print(measo.max())
		# print(measo.min())
		return meas,im

def get_paths(fname):
	paths = []
	with open(fname, 'r') as f:
		for line in f:
			temp = '/root/data/'+str(line).strip()
			paths.append(temp)
	return paths

