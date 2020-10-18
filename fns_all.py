import torch
import scipy.io as sio
import numpy as np
import os
from skimage.color import rgb2gray
import skimage.io
import random
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize as rsz
import torch.optim as optim
import os
from models import*
from torch_vgg import Vgg16


#make train within train for gen and dis
def train_discriminator_epoch(gen, dis, optim_dis, criterion, train_loader, epochs, disc_err,device):
	for param in gen.parameters():
		param.requires_grad = False
	for param in dis.parameters():
		param.requires_grad = True

	for ep in range(epochs):
		for X_train, Y_train in train_loader:
			batchsize = X_train.shape[0]
			target_real = Variable(torch.rand(batchsize,1)*0.5 + 0.7).to(device)
			target_fake = Variable(torch.rand(batchsize,1)*0.3).to(device)
			X_train, Y_train = X_train.to(device), Y_train.to(device)
			optim_dis.zero_grad()
			dis.train()
			high_res_real = Variable(Y_train)
			high_res_fake,_ = gen(X_train)

			dis_loss = criterion(dis(high_res_real), target_real) + criterion(dis(Variable(high_res_fake.data)), target_fake)
			dis_loss.backward()
			optim_dis.step()
			disc_err.append(dis_loss.item())
			print('At Epoch:'+str(ep)+', Pretraining Dis Loss:'+str(dis_loss.item()))
	return disc_err





def validate(gen, dis, vgg, wts, val_loader, gen_criterion, dis_criterion,device):
	k = 0
	tloss = 0
	gen.eval()
	with torch.no_grad():
		for X_val, Y_val in val_loader:			
			batchsize = X_val.shape[0]
			ones_const = Variable(torch.ones(batchsize, 1)).to(device)
			# X_val, Y_val = batchGenerator(1, arr[i], h, phase_mask_fwd,device,pathstr)
			X_val, Y_val = X_val.to(device), Y_val.to(device)
			# print(Y_val.shape)
			X_valout,_ = gen(X_val)
			valfeatures_y = vgg(Y_val)
			valfeatures_x = vgg(X_valout)
			if k == 5:
				op = X_valout
			tloss += (wts[0]*(gen_criterion(Y_val, X_valout)+(wts[1]*gen_criterion(valfeatures_y.relu2_2, valfeatures_x.relu2_2))+(wts[1]*gen_criterion(valfeatures_y.relu4_3, valfeatures_x.relu4_3)))+wts[2]*dis_criterion(dis(X_valout), ones_const)).item()
			k += 1
		tloss = tloss/len(val_loader)
	return op, tloss


def train_full_epoch(gen, dis, vgg, wts, optim_gen, optim_dis, train_loader, val_loader, gen_criterion, dis_criterion, device, vla, e, savedir, train_error, val_error, disc_err,ss,valFreq):
	i = 0
	
	for X_train, Y_train in train_loader:
		X_train, Y_train = X_train.to(device), Y_train.to(device)
		batchsize = X_train.shape[0]

		#Train discriminator
		ones_const = Variable(torch.ones(batchsize, 1)).to(device)
		target_real = Variable(torch.rand(batchsize,1)*0.5 + 0.7).to(device)
		target_fake = Variable(torch.rand(batchsize,1)*0.3).to(device)

		for param in gen.parameters():
			param.requires_grad = False
			
		for param in dis.parameters():
			param.requires_grad = True

		optim_dis.zero_grad()
		dis.train()
		high_res_real = Variable(Y_train)
		high_res_fake,_ = gen(X_train)
		dis_loss = dis_criterion(dis(high_res_real), target_real) + dis_criterion(dis(Variable(high_res_fake.data)), target_fake)
		dis_loss.backward()
		optim_dis.step()
		disc_err.append(dis_loss.item())

		#Train generator
		for param in gen.parameters():
			param.requires_grad = True
			
		for param in dis.parameters():
			param.requires_grad = False

		optim_gen.zero_grad()
		gen.train()
		Xout,_ = gen(X_train)
		features_y = vgg(Y_train)
		features_x = vgg(Xout)
		loss = wts[0]*gen_criterion(Y_train, Xout)+(wts[1]*gen_criterion(features_y.relu2_2, features_x.relu2_2))+(wts[1]*gen_criterion(features_y.relu4_3, features_x.relu4_3))+wts[2]*dis_criterion(dis(Xout), ones_const)
		loss.backward()
		optim_gen.step()
		train_error.append(loss.item())
		if i % valFreq == 0:
			Xvalout, vloss= validate(gen, dis, vgg, wts, val_loader, gen_criterion, dis_criterion, device)
			val_error.append(vloss)
			if vloss < vla:
				vla = vloss
				Xvalout = Xvalout.cpu()
				ims = Xvalout.detach().numpy()
				ims = ims[0, :, :, :]
				ims = np.swapaxes(np.swapaxes(ims,0,2),0,1)
				ims = (ims-np.min(ims))/(np.max(ims)-np.min(ims))
				skimage.io.imsave(savedir+'/best.png', ims)
				dict_save = {
				'gen_state_dict': gen.state_dict(),
				'dis_state_dict': dis.state_dict(),
				'optimizerG_state_dict': optim_gen.state_dict(),
				'optimizerD_state_dict': optim_dis.state_dict(),
				'train_err': train_error,
				'val_err': val_error,
				'disc_err': disc_err,
				'last_finished_epoch': e}
				torch.save(dict_save, savedir+'/best.tar')
				print('Saved best')
		print('Epoch and Iterations::'+str(e)+','+str(i))
		print('Train and Val Loss:'+str(loss.item())+','+str(vloss))
		ss.flush()
		i += 1
	return train_error, val_error, disc_err, vla, Xvalout






