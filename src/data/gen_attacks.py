from PIL import Image
from advertorch.attacks import LinfPGDAttack, FGSM, JSMA
#from advertorch.attacks import FGSM

import matplotlib.pyplot as plt

from advertorch.utils import NormalizeByChannelMeanStd
from torchvision.models import resnet50, vgg16
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import numpy as np

import os, sys

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

device = 'cuda'


class MyDataLoader(Dataset):
    def __init__(self,paths):
        self.paths = paths

    def __len__(self): return len(self.paths)

    def __getitem__(self,i):
        label = int(self.paths[i].split('/')[-2][:4])
        img_pil = Image.open(self.paths[i]).convert('RGB').resize((256,256))
        img_torch = torch.tensor(np.array(img_pil,dtype=np.float32).transpose((2,0,1)))/255
        return img_torch, self.paths[i], label
#        if self.with_fname: return img_torch, self.paths[i]
#        else: img_torch

try:
    sub_path = sys.argv[1]
    data_dir = f'../../data/{sub_path}/'
    print(data_dir)
    assert os.path.exists(data_dir)

except:
    print('pass the data sub-directory as the second arg')

input_dir = '../../data/'

tvt_fnames = list()

for t in ['train','val','test']:
    with open(data_dir + f'{t}_files.txt') as f:
        tvt_fnames.append([input_dir + fname for fname in f.read().split('\n')])


tvt_maps = [MyDataLoader(l) for l in tvt_fnames]

tvt_loaders = [DataLoader(t,batch_size=32,shuffle=False) for t in tvt_maps]





normalize = NormalizeByChannelMeanStd(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

step_sizes = [1,2,3]
n_steps = [8,12,16]


base_models = [resnet50, vgg16]


#attacks = list()
#attack_names = list()


first_iter = True




for b_model in [resnet50, vgg16]:

    model = b_model(pretrained=True)
    model.eval()
    model = nn.Sequential(normalize, model).to(device)


    attacks = list()
    attack_names = list()

    #attacks += [JSMA(model,num_classes=1000),]
    #attack_names += [f'{b_model.__name__}_JSMA',]

    attacks += [LinfPGDAttack(model,eps=ss*ns/255.0,eps_iter=ss/255.0,nb_iter=ns) for ss in step_sizes for ns in n_steps]
    attack_names += [f'{b_model.__name__}_PGD_ns_{ns}_ss_{ss}' for ss in step_sizes for ns in n_steps]

    attacks += [FGSM(model,eps=ss/255.0) for ss in step_sizes]
    attack_names += [f'{b_model.__name__}_FGSM_ss_{ss}' for ss in step_sizes]

    #attacks += [JSMA(model,num_classes=1000),]
    #attack_names += [f'{b_model.__name__}_JSMA',]


    for attack,attack_name in zip(attacks,attack_names):
        for t_name, dg in zip(['train','val','test'],tvt_loaders):

            for X, fnames, labels in tqdm(dg):
                #print(labels)
                adv_labels = (labels + torch.randint(1,1000,labels.shape)) % 1000
                #print(adv_labels)
                #quit()
               
                #l = attack._get_predicted_label(X.to(device))
                #print(l)
                #quit()
                f_tails = ['/'.join(os.path.splitext(f)[0].split('/')[-2:]) for f in fnames]

                X_adv = attack.perturb(X.to(device),adv_labels.to(device))

                for Xi, tail, new_label in zip(X_adv,f_tails,adv_labels):

                    X_np = (255*np.array(Xi.cpu()).transpose((1,2,0))).astype(np.uint8)


                    out_path = f'{data_dir}{attack_name}/{t_name}/{tail}_new_label_{new_label:04d}.png'
                    if not os.path.exists(os.path.dirname(out_path)):
                        os.makedirs(os.path.dirname(out_path))
                    Image.fromarray(X_np).save(out_path)


                if first_iter == True:
                    for Xi, tail in zip(X,f_tails):
                        X_np = (255*np.array(Xi.cpu()).transpose((1,2,0))).astype(np.uint8)
                        out_path = f'{data_dir}original_resize/{t_name}/{tail}.png'
                        if not os.path.exists(os.path.dirname(out_path)):
                            os.makedirs(os.path.dirname(out_path))
                        Image.fromarray(X_np).save(out_path)



            first_iter = False





