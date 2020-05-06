################################# DCGAN-train ###############################################

import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
import torch.optim as optim
import torchvision.utils as vutils
import torch.utils.data
import torchvision.datasets as dset

import model_DCGAN as model

############################################ DATA LOADING(start) ############################################
# data-root
dataroot = "C:/Users/nparab/Desktop/CS230/project/finalproject/cs230-project/dataset/celeba"
dataroot = '/myData/img_align_celeba/img_align_celeba'

#number of workers for dataloader
workers = 2

# training batch-size
batch_size = 128


#Load dataset and resize
dataset = dset.ImageFolder( root=dataroot,
                            transform=transforms.Compose([
                              transforms.Resize(model.image_size),
                              transforms.CenterCrop(model.image_size),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
#Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=True, num_workers=workers)


############################################ DATA LOADING (end) #############################################



########################################### TRAINING (start) ################################################

# batch of input latent vectors
fixed_noise = torch.randn(model.ngv, model.gvSize, 1, 1, device=model.device)


####### Hyperparameter tuning params (start)######

# learning-rate
learning_rate = 0.01

# adam optimizer
beta1 = 0.5

# number of epochs
num_epochs = 5


####### Hyperparameter tuning params (end) ######


####### LOSS Function and Optimizer(start) ######

# BCELoss function
lossF = nn.BCELoss()

# Setup Adam optimizers for both G and D
optD = optim.Adam(model.netD.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optG = optim.Adam(model.netG.parameters(), lr=learning_rate, betas=(beta1, 0.999))

####### LOSS Function and Optimizer(end) ######


####### 3. Training Loop ######
img_list = []
G_losses = []
D_losses = []
iters = 0

# labels
real_label = 1
fake_label = 0

print("Start Trainng...")
for epoch in range(num_epochs):
    print("Start Epoch..." + str(epoch))
    for i, data in enumerate(dataloader):
        
        real = data[0].to(model.device)
        b_size = real.size(0)
        labels_real = torch.full((b_size,), real_label, device=model.device)
        labels_fake = torch.full((b_size,), fake_label, device=model.device)
    
        ## **Train real data through Discriminator** ##
        model.netD.zero_grad()
        #forward prop
        outputD_real = model.netD(real)
        #loss
        lossD_real = lossF(outputD_real, labels_real)
        # backward prop
        lossD_real.backward()
        #stats (D)
        D_x = outputD_real.mean().item()

        
        ## **Train fake data through Discriminator** ##
        #forward prop
        noise = torch.randn(b_size, model.gvSize, 1, 1, device=model.device)
        fake = model.netG(noise)
        outputD_fake = model.netD(fake.detach())
        #loss
        lossD_fake = lossF(outputD_fake,labels_fake)
        # backward prop
        lossD_fake.backward()
        #stats
        D_G_z1 = outputD_fake.mean().item()
        
        #total discriminator loss (real + fake)
        lossD = lossD_real + lossD_fake
        #update
        optD.step()

        
        ## **Update Generator** ##
        model.netG.zero_grad()
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = model.netD(fake)
        #loss
        lossG = lossF(output, labels_real)
        # backward pass (D)
        lossG.backward()
        #update
        optG.step()
       
        #stats (D)
        D_G_z2 = output.mean().item()
        
        
        #### LOG Training stats ####
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                     lossD.item(), lossG.item(), D_x, D_G_z1, D_G_z2))

        # # Save Losses for plotting later
        G_losses.append(lossG.item())
        D_losses.append(lossD.item())

        # # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = model.netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

########################################### TRAINING (end) ################################################

#RESULTS
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(model.device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
