import os
import numpy as np

from torchvision.utils import save_image

import torch.nn as nn
import torch
from mods import *
from scipy import stats

n_imgs = 64
img_size = 3*64*64
img_shape = (3, 64, 64)
ldim = 6

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        edim = 6000

        self.decoder = nn.Sequential(
            nn.BatchNorm1d(ldim, 0.8),
            nn.Linear(ldim, edim, bias = False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.BatchNorm1d(edim, 0.8),
            nn.Linear(edim, img_size, bias = False),
            nn.Tanh()
        )
		
    def forward(self, z):
        img = self.decoder(z)
        return img

d = "../datasets/hf64_rgb.npy"

# Loss function
loss_func = torch.nn.MSELoss()

# Initialize generator
decoder =  Decoder() #torch.load("models/decoder_20") #

# Optimizers
optim_d = torch.optim.Adam(decoder.parameters(), lr = 0.0001)

imgs = torch.tensor(get_faces(0, n_imgs)).float()
imgs = (2*imgs.reshape(imgs.shape[0], img_size))-1

img_base_zs = torch.load("img_base_zs")

# ----------
#  Training
# ----------
batch = 10
start = 0
end = batch
round = 0

def get_zs():
    global imgs
    c_imgs = []
    zs = []
    for i in range(n_imgs):
        path = f"img_zs/img{i}"
        z_names = os.listdir(path)
        try:
            z_names.remove("avg_z")
        except ValueError:
            continue
        nzs = len(z_names)

        if nzs == 0:
            continue
        
        czi = np.random.randint(0,nzs)
        cz_name = z_names[czi]
        z = torch.load(path+"/"+cz_name)[0].numpy()

        c_imgs.append(imgs[i].numpy())
        zs.append(z)
    
    c_imgs = torch.tensor(c_imgs).float()
    zs = torch.tensor(zs).float()
    return c_imgs, zs

def get_zs2():
    global imgs
    c_imgs = []
    zs = []
    for i in range(n_imgs):
        path = f"img_zs/img{i}/avg_z"
        try:
            tzs = torch.load(path)
        except FileNotFoundError:
            continue

        c_imgs.append(imgs[i].numpy())
        zs.append(tzs.numpy())
    
    c_imgs = torch.tensor(c_imgs).float()
    zs = torch.tensor(zs).float()
    return c_imgs, zs

for epoch in range(100000000):
    '''
    # Sample noise as generator input
    optim_d.zero_grad()
    gen_imgs1 = decoder(img_base_zs)
    d_loss = loss_func(gen_imgs1, imgs)
    d_loss.backward()
    optim_d.step()
    '''

    optim_d.zero_grad()

    c_imgs, zs = get_zs()

    gen_imgs2 = decoder(zs)
    d_loss = loss_func(gen_imgs2, c_imgs)
    d_loss.backward()

    optim_d.step()


    optim_d.zero_grad()

    c_imgs, zs = get_zs2()

    gen_imgs3 = decoder(zs)
    d_loss = loss_func(gen_imgs3, c_imgs)
    d_loss.backward()

    optim_d.step()

    start += batch
    end += batch
    if start >= n_imgs:        
        start = 0
        end = batch
        round += 1

    if epoch % 50 == 0:
        Printer(f"{epoch = }, {d_loss.item() = }")

        save_image(imgs.reshape(imgs.shape[0], *img_shape), "images/real_imgs.png", nrow=5, normalize=True)

        #save_image(gen_imgs1[:64].reshape(64, *img_shape), "images/gen_imgs1.png", nrow=5, normalize=True)
        save_image(gen_imgs2[:64].reshape(64, *img_shape), "images/gen_imgs2.png", nrow=5, normalize=True)
        save_image(gen_imgs3[:64].reshape(64, *img_shape), "images/gen_imgs3.png", nrow=5, normalize=True)

        zs = torch.abs(torch.tensor(np.random.normal(0,1,(25,ldim))).float())
        gen_imgs = decoder(zs)
        save_image(gen_imgs.reshape(gen_imgs.shape[0], *img_shape), "images/gen_imgs4.png", nrow=5, normalize=True)

        

        torch.save(decoder, "models/decoder_20")






