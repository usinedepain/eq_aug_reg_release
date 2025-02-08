import torch
import numpy as np
import os
from networks_final import compare_nets,eqvConvNet,omegaCNN

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

"""
    Functions for handling the training of the models
"""


# augment data - each element in batch is rotated uniformly randomly.

def rotate_batch(x):
    
    output = torch.zeros_like(x)
    with torch.no_grad():
        b,d,h,w = x.shape
        
        indices = torch.randint(low=0, high=4, size=(x.shape[0],))

        for k in range(x.shape[0]):
            output[k,:]=torch.rot90(x[k],indices[k],dims=(-2,-1))
    


    return output


# main training loop
def epoch(model,amodel,emodel,gamma,lossf,loader,opt,eopt,aopt,device='cuda'):
    n = len(loader)
    K = loader.batch_size


    # arrays to save drifts from equivariant model 
    diff = np.zeros((n,3))
    adiff = np.zeros((n,3))
    
    # arrays to save projection erros. 
    projerror = np.zeros((n,3))
    projerrora = np.zeros((n,3))

    #arrays to save losses
    losses = np.zeros(n)
    elosses = np.zeros(n)
    alosses = np.zeros(n)


    for k,(x,label) in enumerate(loader):
        d0,d1,d2 = compare_nets(model,emodel)
        diff[k,0] = d0.cpu().detach().numpy()
        diff[k,1] = d1.cpu().detach().numpy()
        diff[k,2] = d2.cpu().detach().numpy()

 

        ad0,ad1,ad2 = compare_nets(amodel,emodel)
        adiff[k,0] = ad0.cpu().detach().numpy()
        adiff[k,1] = ad1.cpu().detach().numpy()
        adiff[k,2] = ad2.cpu().detach().numpy()
        #print(ad0)

        x = x.to(device)
        label=label.to(device)

        pred = model(x)
        epred,_ = emodel(x)

        xrot = rotate_batch(x) 
        apred = amodel(xrot)
        apred_nonrot = amodel(x)


        reg0,reg1,reg2 = model.projection_error()
        projerror[k,0] = reg0.cpu().detach().numpy()
        projerror[k,1] = reg1.cpu().detach().numpy()
        projerror[k,2] = reg2.cpu().detach().numpy()

        areg0,areg1,areg2 = amodel.projection_error()
        projerrora[k,0] = areg0.cpu().detach().numpy()
        projerrora[k,1] = areg1.cpu().detach().numpy()
        projerrora[k,2] = areg2.cpu().detach().numpy()

   
        loss= lossf(pred,label)/K +gamma/2*(reg0+reg1+reg2)
        aloss= lossf(apred,label)/K +gamma/2*(areg0+areg1+areg2)
        eloss = lossf(epred,label)/K

        losses[k] = lossf(pred,label).cpu().detach().numpy()/K
        elosses[k] = lossf(epred,label).cpu().detach().numpy()/K

      
       
        alosses[k] = lossf(apred_nonrot,label).cpu().detach().numpy()/K
        loss.backward()
        eloss.backward()
        aloss.backward()

        opt.step()
        opt.zero_grad()
        eopt.step()
        eopt.zero_grad()
        aopt.step()
        aopt.zero_grad()
    return diff,adiff,projerror,projerrora,losses,elosses,alosses



if __name__ == "__main__":
    omega = torch.ones(3,3)

    mnist = MNIST(root='data',train=False,download=True,transform = ToTensor())
    mnistloader = torch.utils.data.DataLoader(mnist,batch_size=10)


    enet = eqvConvNet(omega,lowest_im_dim=7,out_dim=10,hidden=(16,16))
    net = omegaCNN(omega,hidden=(16,16))
    anet = omegaCNN(omega,hidden=(16,16))
    net.load_weights(enet)
    net.perturb_weights(1e-3)
    anet.load_weights(enet)
    anet.perturb_weights(1e-3)

    #print(compare_nets(net,enet))
    #print(compare_nets(anet,enet))

    opt = torch.optim.SGD(net.parameters(), lr=1e-3)
    eopt = torch.optim.SGD(enet.parameters(),lr=1e-3)
    aopt = torch.optim.SGD(anet.parameters(),lr=1e-3)
    fig1,ax1 = plt.subplots()
    fig2,ax2 = plt.subplots()
    fig3,ax3 = plt.subplots()


    for k in range(5):
        diff, adiff,pe,pa,los,elos,alos = epoch(net,anet,enet,1e2,torch.nn.CrossEntropyLoss(),mnistloader,opt,eopt,aopt)
        N = len(diff)
        ax1.plot(np.arange(k*N,k*N+N),los,label='nom')
        ax1.plot(np.arange(k*N,k*N+N),elos,label='eq')
        ax1.plot(np.arange(k*N,k*N+N),alos,label='aug')




        ax2.plot(np.arange(k*N,k*N+N),diff-pe,label='dif',linestyle=':', color='blue')
        ax2.plot(np.arange(k*N,k*N+N),adiff-pa,label='difa',linestyle=':', color='red')


        ax2.legend()





        ax3.plot(np.arange(k*N,k*N+N),pe,label='eqerr',color='blue')
        ax3.plot(np.arange(k*N,k*N+N),pa,label='eqerra',color='red')





    plt.show()

