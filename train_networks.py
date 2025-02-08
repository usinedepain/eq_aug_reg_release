import torch
import numpy as np
import sys,os
from opt_utils import epoch
from networks_final import eqvConvNet,omegaCNN


from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from tqdm import tqdm



LOAD_ROOT = 'data' # Root for loading dataset
SAVE_ROOT = 'results' # Root for saving data


gamma = int(sys.argv[1])

# Training hyper-parameters
EPOCHS = 10
BATCH_SIZE = 10
LEARNING_RATE = 1e-3

# Load MNIST
mnist = MNIST(root=LOAD_ROOT,train=True,download=True,transform = ToTensor())
mnistloader = torch.utils.data.DataLoader(mnist,batch_size=BATCH_SIZE)

for TASK_ID in tqdm(range(30), desc='runs',leave = False):
    # Set up architectures
    omega = torch.ones(3,3) # support
    enet = eqvConvNet(omega,lowest_im_dim=7,out_dim=10,hidden=(16,16)) # equivariant net
    net = omegaCNN(omega,hidden=(16,16))                               # non-equivariant net, to be trained without augmentation
    anet = omegaCNN(omega,hidden=(16,16))                              # non-equivariant net, to be trained with augmentation


    # initialize non-equivariant nets, perturb
    net.load_weights(enet)
    net.perturb_weights(1e-3)
    anet.load_weights(enet)
    anet.perturb_weights(1e-3)

    net.to('cuda')
    anet.to('cuda')
    enet.to('cuda')


    # initialize optimizers
    opt = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)
    eopt = torch.optim.SGD(enet.parameters(),lr=LEARNING_RATE)
    aopt = torch.optim.SGD(anet.parameters(),lr=LEARNING_RATE)




    for k in tqdm(range(EPOCHS), desc='epochs',leave = False):
        print('epoch started')
        diff, adiff,projerror,aprojerror,los,elos,alos = epoch(net,anet,enet,10**gamma,torch.nn.CrossEntropyLoss(),mnistloader,opt,eopt,aopt)
        filename = os.path.join(SAVE_ROOT,str(TASK_ID)+'_'+str(gamma)+'_'+str(k))
        np.savez_compressed(filename, diff = diff, adiff = adiff,projerror=projerror,aprojerror=aprojerror,loss=los,eloss=elos,aloss=alos)

