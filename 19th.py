import os
import numpy as np

from torchvision.utils import save_image

import torch.nn as nn
import torch
from mods import *
from scipy import stats

n_imgs = 64

n_zs = {}
for i in range(n_imgs):
    try:
        os.makedirs(f"img_zs/img{i}")
    except:
        None
    
    n_zs[str(i)] = 0

img_size = 3*64*64
img_shape = (3, 64, 64)
ldim = 6

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

d = "../datasets/hf64_rgb.npy"

# Initialize encoder
encoder =  Encoder() #torch.load("models/encoder_19") #
torch.save(encoder,"models/encoder_19")

imgs = torch.tensor(get_faces(0, n_imgs)).float()
imgs = (2*imgs.reshape(imgs.shape[0], img_size))-1

img_base_zs = torch.abs(encoder(imgs))
torch.save(img_base_zs,"img_base_zs")

#np_img_base_zs = normalize_pattern(img_base_zs.detach())

counter = 0
while True:
    ndz = torch.abs(torch.tensor(np.random.normal(0,1,(1,ldim))).float())

    '''
    ######### Uses pattern similarity ##########
    nzp = normalize_pattern(ndz)

    vals = torch.sum(torch.abs(np_img_base_zs - nzp), 1)
    
    ci = torch.min(vals)

    index = (vals == ci).nonzero(as_tuple=True)[0].tolist()[0]
    
    print(index)
    '''

    ######### Uses cosine similarity ########
    ndz = ndz.expand((64,ldim))
    sv = nn.CosineSimilarity()(img_base_zs, ndz)
    ci = torch.max(sv)
    index = (sv == ci).nonzero(as_tuple=True)[0].tolist()[0]
    #print(index)

    '''
    ########## Uses Magnitude similarity #########
    vals = torch.sum(torch.abs(img_base_zs - ndz), 1)

    ci = torch.min(vals)

    index = (vals == ci).nonzero(as_tuple=True)[0].tolist()[0]
    print(index)
    quit()
    ##########
    '''
    
    if n_zs[str(index)] < 512:
        n_zs[str(index)] += 1
        torch.save(ndz, f"img_zs/img{index}/z{counter}")

        if n_zs[str(index)] == 512:
            path = f"img_zs/img{index}"
            z_names = os.listdir(path)
            nzs = len(z_names)
            
            tzs = 0
            for z_name in z_names:
                z = torch.load(path+"/"+z_name)[0].numpy()
                tzs += z
            
            tzs = tzs/nzs
            tzs = torch.tensor(tzs).float()

            torch.save(tzs, path+"/avg_z")

        counter += 1
    
    if counter == 100000:
        break


