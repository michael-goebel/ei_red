from torchvision.models import resnet50
from torchvision import transforms
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
#from data_gen import CoOccur, Laplace, Normalize, ToTensor, REDDataset
#from glob import glob
import os, sys
from tqdm import tqdm
from time import time
import numpy as np

assert len(sys.argv) == 3, 'Run with dataset as first arg and model type as second, ex: python train.py data_med laplace'

model_type = sys.argv[2]
assert model_type in {'laplace','co_occur'}


model_dir = '../../models/' + sys.argv[1]
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
model_path = os.path.join(model_dir,'model.pt')
model_hist = os.path.join(model_dir,'hist.txt')

debug = True
use_pbar = True

data_dir = '../../data/data_med/'

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def get_files_and_labels(tvt):
    dirs = ['original_resized/']
    for model in ['resnet50','vgg16']:
        dirs += [f'{model}_PGD_ns_{n}_ss_{s}/' for n in [8,12,16] for s in [1,3]]
    all_files = [glob(data_dir + d + f'{tvt}/*/*.png') for d in dirs]
    print([len(l) for l in all_files])
    label_file_pairs = [(i,f) for i,l in enumerate(all_files) for f in l]
    labels, files = zip(*label_file_pairs)
    return files, labels




bs = 32
n_cpu = 4

if model_type == 'laplace':
    pre_proc = transforms.Compose([Laplace(),Normalize(),ToTensor()])
elif model_type == 'co_occur':
    pre_proc = transforms.Compose([CoOccur(),Normalize(),ToTensor()])


dataset_tr = REDDataset(*get_files_and_labels('train'),pre_proc,debug)
dataset_va = REDDataset(*get_files_and_labels('val'),pre_proc,debug)
dataset_te = REDDataset(*get_files_and_labels('test'),pre_proc,debug)

dg_kwargs = {'shuffle': True, 'batch_size': bs, 'num_workers': n_cpu}
dg_tr = torch.utils.data.DataLoader(dataset_tr,**dg_kwargs)
dg_va = torch.utils.data.DataLoader(dataset_va,**dg_kwargs)
dg_te = torch.utils.data.DataLoader(dataset_te,**dg_kwargs)

n_classes = dataset_tr.n_classes
model = resnet50(pretrained=True)

if model_type == 'co_occur':
    conv1_weights = model.conv1.weight.cpu()
    new_weights = torch.cat((conv1_weights,)*2,dim=1)
    model.conv1 = nn.Conv2d(6,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
    #print(conv1_weights.shape,new_weights.shape)
    model.conv1.weight.data = new_weights


model.load_state_dict(torch.load(model_path))


model = model.to(device)


def run_model(model,device,dg,use_pbar):

    if use_pbar: pbar = tqdm(total=len(dg_tr))

    cum_loss = 0
    cum_acc = 0

    predictions = np.empty((len(dg.dataset.fnames),n_classes),dtype=np.float32)
    gts = np.empty(len(dg.dataset.fnames))

    for i,(X,y) in enumerate(dg):
        X, y = X.to(device), y.to(device)
        output = model(X)
        loss = nn.CrossEntropyLoss()(output,y)
        cum_loss += float(loss)
        cum_acc += float(torch.mean((torch.argmax(output,1)==y).float()))
        
        predictions[bs*i:bs*(i+1)] = output.detach().cpu().numpy()[:,:n_classes]
        gts[bs*i:bs*(i+1)] = y.detach().cpu().numpy()

        if use_pbar: 
            pbar.set_description(f'Loss: {cum_loss/(i+1):0.3f}, Acc: {cum_acc/(i+1):0.3f}')
            pbar.update(1)

    if use_pbar: pbar.close()

    return cum_loss/len(dg), cum_acc/len(dg), predictions, gts




for dg, name in [(dg_tr,'train'),(dg_va,'val'),(dg_te,'test')]:


    loss, acc, preds, gts = run_model(model,device,dg,use_pbar=True)



    this_dir = os.path.join(model_dir,name)
    if not os.path.exists(this_dir): os.mkdir(this_dir)

    np.save(os.path.join(this_dir,'outputs.npy'),preds)
    np.save(os.path.join(this_dir,'gts.npy'),gts)
    with open(os.path.join(this_dir,'fnames.txt'),'w+') as f:
        f.write('\n'.join(dg.dataset.fnames))





