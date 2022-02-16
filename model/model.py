import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self,
                 generator,
                 global_discriminator,
                 local_discriminator,
                 criterion,
                 vgg,
                 vgg_loss,
                 lr=0.0001, beta=0.5):
        super().__init__()
        self._netG = generator
        self._netD_global = global_discriminator
        self._netD_local = local_discriminator
        self.vgg = vgg
        self.vgg_loss = vgg_loss

        self.optimizer_G = torch.optim.Adam(self._netG.parameters(), lr=lr, betas=(beta, 0.999))
        self.optimizerD_global = torch.optim.Adam(self._netD_global.parameters(), lr=lr, betas=(beta, 0.999))
        self.optimizerD_local = torch.optim.Adam(self._netD_local.parameters(), lr=lr, betas=(beta, 0.999))

        self._criterion = criterion

        self.lr = lr
        self.default_lr = lr

    def forward(self, input, attention_map):
        output = self._netG.forward(input, attention_map)
        return output

    def loss_G(self, img_dark, img_normal_light, img_generated, patches_dark_img, patches_generated_img):
        loss_G = 0
        prediction_generated_img = self._netD_global.forward(img_generated)
        prediction_normal_img = self._netD_global.forward(img_normal_light)

        # global discriminator loss for generator
        loss_G_global = self._criterion.compute(prediction_normal_img - torch.mean(prediction_generated_img), False) \
                        + self._criterion.compute(prediction_generated_img - torch.mean(prediction_normal_img), True)
        loss_G_global /= 2
        loss_G += loss_G_global

        # local discriminator loss for generator
        loss_G_local = 0
        for patch in patches_generated_img:
            prediction_patch = self._netD_local.forward(patch)
            loss_G_local += self._criterion.compute(prediction_patch, True)

        loss_G_local /= len(patches_generated_img)
        loss_G += loss_G_local

        # VGG perceptual loss
        vgg_loss = 0

        vgg_loss += self.vgg_loss.compute(self.vgg, img_generated, img_dark)

        vgg_patch_loss = 0
        for i in range(len(patches_generated_img)):
            vgg_patch_loss += self.vgg_loss.compute(self.vgg, patches_generated_img[i], patches_dark_img[i])

        vgg_loss += vgg_patch_loss / len(patches_generated_img)

        loss_G += vgg_loss
        return loss_G, loss_G_global, loss_G_local, vgg_loss

    def loss_D(self, net, normal_img, generated_img, hybrid_loss):
        loss_D = 0

        prediction_normal_img = net.forward(normal_img)
        prediction_generated_img = net.forward(generated_img)
        if hybrid_loss:
            loss_D += self._criterion.compute(prediction_normal_img - torch.mean(prediction_generated_img),
                                              True) + self._criterion.compute(
                prediction_generated_img - torch.mean(prediction_normal_img), False)
        else:
            loss_D += self._criterion.compute(prediction_normal_img, True)
            loss_D += self._criterion.compute(prediction_generated_img, False)
        loss_D /= 2
        return loss_D

    def loss_D_local(self, patches_normal_img, patches_generated_img):
        D_loss = 0
        for i in range(len(patches_generated_img)):
            D_loss += self.loss_D(self._netD_local, patches_normal_img[i], patches_generated_img[i], False)
        D_loss /= len(patches_generated_img)
        return D_loss

    def loss_D_global(self, normal_img, generated_img):
        D_loss = self.loss_D(self._netD_global, normal_img, generated_img, True)
        return D_loss

    def update_learning_rate(self, niter):
        lr_decay = self.default_lr / niter
        lr = self.lr - lr_decay
        for param_group in self.optimizerD_global.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizerD_local.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        self.lr = lr
