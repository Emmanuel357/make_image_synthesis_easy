import os
import numpy as np

from torchvision.utils import save_image

import torch.nn as nn
import torch
from mods import *

n_imgs = 202599
img_size = 3*64*64
img_shape = (3, 64, 64)
ldim = 512

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        edim = img_size

        self.decoder = nn.Sequential(
            nn.BatchNorm1d(ldim, 0.8),
            nn.Linear(ldim, edim, bias = False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.BatchNorm1d(edim, 0.8),
            nn.Linear(edim, img_size, bias = False),
            nn.Tanh()
        )
		
    def forward(self, z):
        gen_img = self.decoder(z)
        return gen_img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        edim = img_size

        self.model = nn.Sequential(
            nn.Linear(img_size, edim, bias = False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(edim, 1, bias = False),
            #nn.Sigmoid()
        )
		
    def forward(self, img):
        pred = self.model(img)
        return pred

d = "../datasets/hf64_rgb.npy"

test_img = torch.tensor(np.load("emma.npy")).float().reshape(1, img_size)

tz = torch.tensor(get_zs2(0, 1)).float()

img1 = torch.tensor(get_faces(0, 1)).float()
save_image(img1.reshape(1, *img_shape), "images/first_imgs.png", nrow=5, normalize=True)

# Loss function
loss_func = torch.nn.MSELoss()

# Initialize generator
generator =  torch.load("models/eight_generator") #Generator() # /
discriminator =  torch.load("models/eight_discriminator") #Discriminator() #

# Optimizers
optim_g = torch.optim.Adam(generator.parameters(), lr = 0.0001)
optim_d = torch.optim.Adam(discriminator.parameters(), lr = 0.0001)

# ----------
#  Training
# ----------
start = 0
end = 64

round = 0
for epoch in range(100000000):

    imgs = torch.tensor(get_faces(start, end)).float()
    imgs = (2*imgs.reshape(imgs.shape[0], img_size))-1

    # Sample noise as generator input
    zs = torch.tensor(np.random.normal(0,1,(imgs.shape[0],ldim))).float()

    optim_g.zero_grad()
    gen_imgs = generator(zs)
    gd_preds = discriminator(gen_imgs)
    g_loss = loss_func(gd_preds, torch.ones((gd_preds.shape)))
    g_loss.backward()
    optim_g.step()
	
    optim_d.zero_grad()
    gd_preds = discriminator(gen_imgs.detach())
    d_preds = discriminator(imgs)
    d_loss = (loss_func(gd_preds, torch.zeros((gd_preds.shape))) + loss_func(d_preds, torch.ones((d_preds.shape))))/2
    d_loss.backward()
    optim_d.step()

    start += 64
    end += 64
    if start >= 512:
        print(f"\nround {round} done.")
        start = 0
        end = 64

        round += 1

    if epoch % 50 == 0:
        Printer(f"{epoch = }, {g_loss.item() = }, {d_loss.item() = }")

        save_image(imgs.reshape(imgs.shape[0], *img_shape), "images/real_imgs.png", nrow=5, normalize=True)

        save_image(gen_imgs.reshape(gen_imgs.shape[0], *img_shape), "images/gen_imgs.png", nrow=5, normalize=True)

        #print(test_img.shape)
        #im = torch.cat((test_img, imgs[0].unsqueeze(0)))
        #timg = autoencoder(im)
        #save_image(timg.reshape(2, *img_shape), "images/test_recr_face.png", nrow=5, normalize=True)

        #gen_img = decoder(torch.tensor(np.random.normal(0,1,(2,ldim))).float())
        #save_image(gen_img.reshape(2, *img_shape), "images/gen_img.png", nrow=5, normalize=True)

        #z = (new_z[0] + new_z[1])/2
        #bimg = generator(z)
        #save_image(bimg.reshape(1, *img_shape), "images/blended_img.png", nrow=5, normalize=True)
        
        torch.save(generator, "models/eight_generator")
        torch.save(discriminator, "models/eight_discriminator")






