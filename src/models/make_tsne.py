import torch
import os, sys
from time import time
import numpy as np
from utils import get_files_and_labels, load_model, run_model, get_all_files
from data_gen import REDDataset
from torch.utils.data import DataLoader

debug = False
use_pbar = True

bs = 32
n_cpu = 4


assert len(sys.argv) == 4, 'Run with dataset as first arg and model type as second and gpu as third, ex: python train.py data_med laplace 0'

model_type = sys.argv[2]
assert model_type in {'laplace','co_occur', 'direct'}

os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[3]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir = '../../models/' + sys.argv[1]
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
model_path = os.path.join(model_dir,'model.pt')
model_hist = os.path.join(model_dir,'hist.txt')


n_classes = 13

files_te = get_all_files('test')



pre_proc, model = load_model(model_type,n_classes,weights_path=model_path)

dg_kwargs = {'shuffle': True, 'batch_size': bs, 'num_workers': n_cpu}

dg_te = DataLoader(REDDataset(files_te,[0,]*len(files_te),pre_proc,debug),**dg_kwargs)

model.to(device)


for dg, name in [(dg_te,'test')]:

    _, _, preds, _ = run_model(model,device,dg,use_pbar=True)

    this_dir = os.path.join(model_dir,name)
    if not os.path.exists(this_dir): os.mkdir(this_dir)

    np.save(os.path.join(this_dir,'outputs.npy'),preds)
    with open(os.path.join(this_dir,'fnames.txt'),'w+') as f:
        f.write('\n'.join(dg.dataset.fnames))


