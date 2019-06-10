from model import Generator, Discriminator
import torch.nn as nn
import torch.optim as optim
import torch
import utils

# %%


class CGAN():

    def __init__(self, lr, beta1, input_dim, hidden_dim, output_dim, num_layers):
        self.netG = Generator(input_dim, hidden_dim, output_dim, num_layers)
        self.netD = Discriminator()
        self.criterion = nn.MSELoss()
        self.optimizerD = optim.Adam(
            self.netD.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizerG = optim.Adam(
            self.netG.parameters(), lr=lr, betas=(beta1, 0.999))
        self.real_label = 1
        self.fake_label = 0

    def update(self, realData, condition):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        self.netD.zero_grad()
        # Format batch
        b_size = realData.size(0)
        label = torch.full((b_size,), self.real_label)
        output = self.netD(realData, condition).view(-1)
        errD_real = self.criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # 这里应该修改，噪声应该为打乱的二级结构组合
        noise = torch.rand(250, 3)*5+realData
        fake = self.netG(noise, condition)[0]
        label.fill_(self.fake_label)
        output = self.netD(fake.detach(), condition).view(-1)
        errD_fake = self.criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        self.optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        self.netG.zero_grad()
        label.fill_(self.real_label)
        output = self.netD(fake, condition).view(-1)
        errG = self.criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        self.optimizerG.step()

        print('Loss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
              % (errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        return output
