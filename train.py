import torch
import os

from model.generators.enlighten_gan_generator import GeneratorEnlightenGAN
from model.discriminators.enlighten_gan_global_discriminator import GlobalDiscriminatorEnlightenGAN
from model.discriminators.enlighten_gan_local_discriminator import LocalDiscriminatorEnlightenGAN
from model.model import Model
from model.attention_map import AttentionMapGray
from model.loses.gan_loss import GanLoss
from model.vgg import Vgg16
from model.loses.vgg_loss import VGGLoss
from utils import patches
from data_loader import ABDatasetLoader

if __name__ == '__main__':
    netG = GeneratorEnlightenGAN()
    netD_global = GlobalDiscriminatorEnlightenGAN()
    netD_local = LocalDiscriminatorEnlightenGAN()
    attention_map = AttentionMapGray()
    criterion = GanLoss()
    vgg_loss = VGGLoss()

    netVgg = Vgg16()
    netVgg.device("cuda")
    model_dir = '../../saved_models'
    netVgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))
    netVgg = torch.nn.DataParallel(netVgg, device_ids=[0])

    model = Model(netG, netD_global, netD_local, criterion, attention_map, netVgg, vgg_loss)

    niter = 100
    niter_decay = 100
    dataset_root = ''

    dataset = ABDatasetLoader(dataset_root)
    for epoch in range(niter + niter_decay + 1):
        for i, data in enumerate(dataset):
            img_dark = data['dark']
            img_normal = data['normal']

            img_generated = model.forward(img_dark)

            patches_dark_img = patches.create_patch(img_dark)
            patches_normal_img = patches.create_patch(img_normal)
            patches_generated_img = patches.create_patch(img_generated)

            model.optimizer_G.zero_grad()
            model.loss_G(img_dark, img_normal, img_generated, patches_dark_img, patches_generated_img).backward()
            model.optimizer_G.step()

            model.optimizer_D_global.zero_grad()
            model.loss_D_global(img_normal, img_generated).backward()

            model.optimizer_D_local.zero_grad()
            model.loss_D_local(patches_normal_img, patches_generated_img).backward()

            model.optimizer_D_global.step()
            model.optimizer_D_local.step()

        if epoch > niter:
            model.update_learning_rate(niter)
