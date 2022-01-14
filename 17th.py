import os
import numpy as np

from torchvision.utils import save_image

import torch.nn as nn
import torch
from mods import *
from scipy import stats

n_imgs = 202599
img_size = 3*64*64
img_shape = (3, 64, 64)
ldim = 512

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.BatchNorm1d(img_size),
            nn.Linear(img_size, ldim, bias = False),
            nn.LeakyReLU(0.5, inplace=True)
        )
		
    def forward(self, img):
        z = self.encoder(img)
        return z

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        edim = 1024

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

test_img = torch.tensor(np.load("emma.npy")).float().reshape(1, img_size)

img1 = torch.tensor(get_faces(0, 1)).float()
save_image(img1.reshape(1, *img_shape), "images/first_imgs.png", nrow=5, normalize=True)

# Loss function
loss_func = torch.nn.MSELoss()

# Initialize generator
encoder =  Encoder() #torch.load("models/encoder_17") #
decoder =  Decoder() #torch.load("models/decoder_17") #

# Optimizers
optim_e = torch.optim.Adam(encoder.parameters(), lr = 0.0001)
optim_d = torch.optim.Adam(decoder.parameters(), lr = 0.0001)

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
    optim_e.zero_grad()
    optim_d.zero_grad()

    zs = encoder(imgs)
    gen_imgs = decoder(zs)
    d_loss = loss_func(gen_imgs, imgs)
    d_loss.backward()

    optim_d.step()
    optim_e.step()

    optim_d.zero_grad()

    mean, std = stats.norm.fit(zs.detach().numpy())
    tzs = torch.tensor(stats.norm.pdf(zs.detach().numpy(), mean, std)).float()

    gen_imgs2 = decoder(tzs)
    d_loss = loss_func(gen_imgs2, imgs)
    d_loss.backward()

    optim_d.step()

    start += 64
    end += 64
    if start >= 512:        
        start = 0
        end = 64
        round += 1

    if epoch % 50 == 0:
        Printer(f"{epoch = }, {d_loss.item() = }")

        save_image(imgs.reshape(imgs.shape[0], *img_shape), "images/real_imgs.png", nrow=5, normalize=True)

        save_image(gen_imgs.reshape(gen_imgs.shape[0], *img_shape), "images/gen_imgs.png", nrow=5, normalize=True)
        save_image(gen_imgs2.reshape(gen_imgs.shape[0], *img_shape), "images/gen_imgs2.png", nrow=5, normalize=True)

        zs = torch.tensor(np.random.normal(mean,std,(25,ldim))).float()
        gen_imgs = decoder(zs)
        save_image(gen_imgs.reshape(gen_imgs.shape[0], *img_shape), "images/gen_imgs3.png", nrow=5, normalize=True)

        noise_imgs = torch.tensor(np.random.uniform(0,1,(25,img_size))).float()
        noise_imgs = torch.cat((test_img, noise_imgs))
        zs = encoder(noise_imgs)
        gen_imgs = decoder(zs)
        save_image(gen_imgs.reshape(gen_imgs.shape[0], *img_shape), "images/gen_imgs4.png", nrow=5, normalize=True)

        #print(test_img.shape)
        #im = torch.cat((test_img, imgs[0].unsqueeze(0)))
        #timg = autoencoder(im)
        #save_image(timg.reshape(2, *img_shape), "images/test_recr_face.png", nrow=5, normalize=True)

        #gen_img = decoder(torch.tensor(np.random.normal(0,1,(2,ldim))).float())
        #save_image(gen_img.reshape(2, *img_shape), "images/gen_img.png", nrow=5, normalize=True)
        '''
        decoder.eval()
        z = (mzs[0] + mzs[1])/2
        bimg = decoder(z.unsqueeze(0))
        save_image(bimg.reshape(1, *img_shape), "images/blended_img.png", nrow=5, normalize=True)
        decoder.train()
        '''

        torch.save(encoder, "models/encoder_17")
        torch.save(decoder, "models/decoder_17")






