from torchvision.models import resnet50
from torchvision import transforms
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim



from data_gen import CoOccur, Laplace, Normalize, ToTensor, REDDataset
from glob import glob
import os, sys
from tqdm import tqdm

#assert len(sys.argv) == 2, 'Pass the name for the experiment as second argument'

assert len(sys.argv) == 3, 'Run with dataset as first arg and model type as second, ex: python train.py data_med laplace'

model_type = sys.argv[2]
assert model_type in {'laplace','co_occur'}


model_dir = '../../models/' + sys.argv[1]
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
model_path = os.path.join(model_dir,'model.pt')



#data_dir = '../../data/data_med/'
data_dir = '../../data/data2_small/'

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_epochs = 10


def get_files_and_labels(tvt):
    dirs = ['original_resize/']
    for model in ['resnet50','vgg16']:
        dirs += [f'{model}_PGD_ns_{n}_ss_{s}/' for n in [8,12,16] for s in [1,3]]
    all_files = [glob(data_dir + d + f'{tvt}/*/*.png') for d in dirs]
    print([data_dir + d + f'{tvt}/*/*.png' for d in dirs])
    print([len(l) for l in all_files])
    label_file_pairs = [(i,f) for i,l in enumerate(all_files) for f in l]
    labels, files = zip(*label_file_pairs)
    return files, labels




#train_dirs = ['original_resize/']

#for model in ['resnet50','vgg16']:
#    train_dirs += [f'{model}_PGD_ns_{n}_ss_{s}/' for n in [1,10] for s in [1,3]]


bs = 32
n_cpu = 4

#print(train_dirs)


#all_files = [glob(data_dir + d + 'train/*/*.png') for d in train_dirs]



#label_file_pairs = [(i,f) for i,l in enumerate(all_files) for f in l]

#labels, files = zip(*label_file_pairs)

if model_type == 'laplace':
    pre_proc = transforms.Compose([Laplace(),Normalize(),ToTensor()])
elif model_type == 'co_occur':
    pre_proc = transforms.Compose([CoOccur(),Normalize(),ToTensor()])


#dg_tr = torch.utils.data.DataLoader(REDDataset(files,labels,pre_proc),shuffle=True,batch_size=bs,num_workers=n_cpu)
#dg_va = torch.utils.data.

#get_files_and_labels


dataset_tr = REDDataset(*get_files_and_labels('train'),pre_proc)
dataset_va = REDDataset(*get_files_and_labels('val'),pre_proc)


#dg_tr = torch.utils.data.DataLoader(REDDataset(*get_files_and_labels('train'),pre_proc),shuffle=True,batch_size=bs,num_workers=n_cpu)
#dg_va = torch.utils.data.DataLoader(REDDataset(*get_files_and_labels('val'),pre_proc),shuffle=True,batch_size=bs,num_workers=n_cpu)

dg_kwargs = {'shuffle': True, 'batch_size': bs, 'num_workers': n_cpu}
dg_tr = torch.utils.data.DataLoader(dataset_tr,**dg_kwargs)
dg_va = torch.utils.data.DataLoader(dataset_va,**dg_kwargs)


n_classes = dataset_tr.n_classes
model = resnet50(pretrained=True).to(device)

if model_type == 'co_occur':
    conv1_weights = model.conv1.weight
    new_weights = torch.stack((conv1_weights,)*2,dim=1)
    model.conv1 = nn.Linear(6,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
    model.conv1.weight = new_weights



#last_layer_weight = model.fc.weight
#last_layer_b



#model







#(conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#(fc): Linear(in_features=2048, out_features=1000, bias=True)

print(model)



#quit()

def train(model,device,dg_tr,dg_va,optimizer):
    model.train()
    pbar = tqdm(total=len(dg_tr))
    cum_loss_tr = 0
    cum_acc_tr = 0
    for i,(X, y) in enumerate(dg_tr):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = nn.CrossEntropyLoss()(output,y)
        loss.backward()
        optimizer.step()
        cum_loss_tr += float(loss)
        cum_acc_tr += float(torch.mean((torch.argmax(output,1)==y).float()))
        pbar.set_description(f'TRAIN - Loss: {cum_loss_tr/(i+1):0.3f}, Acc: {cum_acc_tr/(i+1):0.3f}')
        pbar.update(1)

    pbar.close()

    model.eval()
    pbar = tqdm(total=len(dg_tr))
    cum_loss_va = 0
    cum_acc_va = 0
    for i,(X, y) in enumerate(dg_tr):
        X, y = X.to(device), y.to(device)
        output = model(X)
        loss = nn.CrossEntropyLoss()(output,y)
        cum_loss_va += float(loss)
        cum_acc_va += float(torch.mean((torch.argmax(output,1)==y).float()))
        pbar.set_description(f'VAL   - Loss: {cum_loss_va/(i+1):0.3f}, Acc: {cum_acc_va/(i+1):0.3f}')
        pbar.update(1)

    pbar.close()

    return cum_loss_tr/len(dg_tr), cum_acc_tr/len(dg_tr), cum_loss_va/len(dg_va), cum_acc_va/len(dg_va)



#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

optimizer = optim.Adam(model.parameters())


best_val_loss = float('inf')
for i in range(n_epochs):
    print(f'\nEPOCH {i+1} of {n_epochs}')
    _,_,val_loss,_ = train(model,device,dg_tr,dg_va,optimizer)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(),model_path)








