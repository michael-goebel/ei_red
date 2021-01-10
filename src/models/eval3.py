import numpy as np
from utils import label2name
import scipy.ndimage as ndi


name2label = {v:i for i,v in enumerate(label2name)}


input_dir = '../../models/med_co/test/'

with open(input_dir + 'fnames.txt') as f: fnames = f.read().split('\n')

outputs = np.load(input_dir + 'outputs.npy')

names = [f.split('/')[-4] for f in fnames]
labels = np.array([name2label[n] if n in name2label else -1 for n in names])






def get_acc(outputs,names,labels):
    mask = labels >= 0
    outputs_filt = outputs[mask]
    labels_filt = labels[mask]
    y_est = np.argmax(outputs_filt,1)
    return np.mean(y_est==labels_filt)
    
    

def get_stats(model,attack,param,outputs,names,labels):

    if attack == 'FGSM':
        names_list = [f'{model}_FGSM_ss_{i}' for i in [1,2,3]]
    else:
        names_list = [f'{model}_PGD_ns_{n}_ss_{s}' for n in [8,12,16] for s in [1,2,3]]

    names_set = set(names_list)

    mask = np.array([n in names_set for n in names])
    label_mask = np.array([l in names_set for l in label2name])

    ind = -1 if param == 'ss' else -3

    gt_vals = np.array([int(n.split('_')[ind]) for n,v in zip(names,mask) if v])
    vals = np.array([int(n.split('_')[ind]) if v else 0 for n,v in zip(label2name,label_mask)])

    outputs_filt = outputs[mask][:,label_mask]
    vals_filt = vals[label_mask]

    outputs_filt = np.exp(outputs_filt) / np.exp(outputs_filt).sum(1)[:,None]
    vals_est = (outputs_filt * vals_filt[None,:]).sum(1)
    name2arr_idx = {n:i for i,n in enumerate(names_list)}
    ndi_labels = np.array([name2arr_idx[n] for n,m in zip(names,mask) if m])
    index = np.arange(len(names_list))
    avg_vals = ndi.mean(vals_est,ndi_labels,index)
    mse_vals = ndi.mean((vals_est-gt_vals)**2,ndi_labels,index)
    
    return names_list, avg_vals, mse_vals








#print(get_fgsm_stats('vgg16',outputs,names,labels))
#print(get_pgd_ss_stats('vgg16',outputs,names,labels))

#print(label2name)


for model in ['vgg16','resnet50']:
    for args in [[model,'FGSM','ss'],[model,'PGD','ss'],[model,'PGD','ns']]:
        print(args)
        print(get_stats(*args,outputs,names,labels))


#print(get_stats('vgg16','FGSM','ss',outputs,names,labels))

#print(get_stats('vgg16','PGD','ss',outputs,names,labels))



#print(get_stats('vgg16','PGD','ns',outputs,names,labels))

#for model in ['resnet50', 'pgd']:
    


#print(names[::1000])
#print(labels[::1000])




#print(label2name)
#print(fnames[::1000])






#for model in ['resnet50','vgg16']:

#    pass






