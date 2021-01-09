import torch
from torchvision.models import resnet50
from torchvision import transforms
from glob import glob
from data_gen import Laplace, CoOccur, Normalize, ToTensor

import torch.nn as nn
from tqdm import tqdm
import numpy as np


data_dir = '../../data/data_med/'





#label2name = [


groups = [f'FGSM_ss_{n}' for n in [1,3]]
groups += [f'PGD_ns_{n}_ss_{s}' for n in [8,16] for s in [1,3]]

label2name = ['original_resized',] + [m + g for m in ['resnet50_','vgg16_'] for g in groups]




def get_files_and_labels(tvt):
    
    all_files = [glob(data_dir + 'original_resized/' + f'{tvt}/*/*.png'),]
    groups = [f'FGSM_ss_{n}/' for n in [1,3]]
    groups += [f'PGD_ns_{n}_ss_{s}/' for n in [8,16] for s in [1,3]]
    
    all_files += [glob(data_dir + m + g + f'{tvt}/*/*.png') for m in ['resnet50_','vgg16_'] for g in groups]

    

    print([len(l) for l in all_files])
    label_file_pairs = [(i,f) for i,l in enumerate(all_files) for f in l]
    labels, files = zip(*label_file_pairs)
    return files, labels


def get_all_files(tvt):
    all_files = [glob(data_dir + 'original_resized/' + f'{tvt}/*/*.png'),]
    groups = [f'FGSM_ss_{n}/' for n in [1,2,3]]
    groups += [f'PGD_ns_{n}_ss_{s}/' for n in [8,12,16] for s in [1,2,3]]
    all_files += [glob(data_dir + m + g + f'{tvt}/*/*.png') for m in ['resnet50_','vgg16_'] for g in groups]
    print([len(l) for l in all_files])
    all_files = sum(all_files,[])
    return all_files



def load_model(model_type,n_classes,weights_path=None):
    if model_type == 'laplace':
        pre_proc = transforms.Compose([Laplace(),Normalize(),ToTensor()])
    elif model_type == 'co_occur':
        pre_proc = transforms.Compose([CoOccur(),Normalize(),ToTensor()])
    elif model_type == 'direct':
        pre_proc = transforms.Compose([Normalize(),ToTensor()])

    model = resnet50(pretrained=True)

    if model_type == 'co_occur':
        conv1_weights = model.conv1.weight.clone()
        new_weights = torch.cat((conv1_weights,)*2,dim=1)
        model.conv1 = nn.Conv2d(6,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
        model.conv1.weight.data = new_weights

    fc_weights = model.fc.weight.clone()
    fc_bias = model.fc.bias.clone()
    model.fc = nn.Linear(2048,n_classes,bias=True)
    model.fc.weight.data = fc_weights[:n_classes]
    model.fc.bias.data = fc_bias[:n_classes]


    if weights_path: model.load_state_dict(torch.load(weights_path))

    return pre_proc, model


# optimizer set to None is for eval, providing an optimizer sets to train
def run_model(model,device,dg,optimizer=None,use_pbar=True):

    train = True if optimizer is not None else False
    bs = dg.batch_size
    n_classes = model.fc.weight.shape[0]
    if train: model.train()
    else: model.eval()

    if use_pbar: pbar = tqdm(total=len(dg))

    cum_loss = 0
    cum_acc = 0

    predictions = np.empty((len(dg.dataset.fnames),n_classes),dtype=np.float32)
    gts = np.empty(len(dg.dataset.fnames))

    for i,(X,y) in enumerate(dg):
        X, y = X.to(device), y.to(device)
        if train: optimizer.zero_grad()
        output = model(X)
        loss = nn.CrossEntropyLoss()(output,y)
        if train: loss.backward()
        if train: optimizer.step()
        cum_loss += float(loss)
        cum_acc += float(torch.mean((torch.argmax(output,1)==y).float()))
        predictions[bs*i:bs*(i+1)] = output.detach().cpu().numpy()
        gts[bs*i:bs*(i+1)] = y.detach().cpu().numpy()
        if use_pbar:
            pbar.set_description(f'{"TRAIN" if train else "EVAL "} - Loss: {cum_loss/(i+1):0.3f}, Acc: {cum_acc/(i+1):0.3f}')
            pbar.update(1)

    if use_pbar: pbar.close()

    return cum_loss/len(dg), cum_acc/len(dg), predictions, gts





