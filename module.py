import os

import numpy as np
from torch import optim, nn
import torch.nn.functional as F
import pytorch_lightning as L
import matplotlib.pyplot as plt

from components.discriminator import MunitDiscriminator, disc_hinge_loss

from components.network import ResnetGenerator
from helpers.utils import mutual_information, LambdaLR, weights_init_normal


class CycleGAN(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False

        self.generatorBtoA = ResnetGenerator().apply(weights_init_normal)
        self.generatorAtoB = ResnetGenerator().apply(weights_init_normal)

        self.discA = MunitDiscriminator()
        self.discB = MunitDiscriminator()

        self.lambda_cyc = 10
        self.lambda_idt = 5
        self.lambda_adv = 1

        self.mi_A = []
        self.mi_B = []

    def disc_loss(self, disc_out, disc_update, real):
        return disc_hinge_loss(disc_out, disc_update, real)

    def training_step_generators(self, batch):
        gen_optimzer, _ = self.optimizers()
        self.toggle_optimizer(gen_optimzer)

        A, B = batch.values()

        """I. Generator from A to B"""
        # i. Cyclic Forward
        B_hat = self.generatorAtoB(A)
        A_hat = self.generatorBtoA(B_hat)
        cycAB_loss = F.l1_loss(A_hat, A)
        dscAB_loss = self.disc_loss(self.discB(B_hat), disc_update=False, real=True)

        mid_B_hat = B_hat.detach()

        # ii. Identity Forward
        B_hat = self.generatorAtoB(B)
        idtAB_loss = F.l1_loss(B_hat, B)

        AB_loss = self.lambda_cyc * cycAB_loss + self.lambda_adv * dscAB_loss + self.lambda_idt * idtAB_loss

        """II. Generator from B to A"""
        # i. Cyclic Forward
        A_hat = self.generatorBtoA(B)
        B_hat = self.generatorAtoB(A_hat)
        cycBA_loss = F.l1_loss(B_hat, B)
        dscBA_loss = self.disc_loss(self.discA(A_hat), disc_update=False, real=True)

        mid_A_hat = A_hat.detach()

        # ii. Identity Forward
        A_hat = self.generatorBtoA(A)
        idtBA_loss = F.l1_loss(A_hat, A)

        BA_loss = self.lambda_cyc * cycBA_loss + self.lambda_adv * dscBA_loss + self.lambda_idt * idtBA_loss

        """III. Backprop"""
        loss = AB_loss + BA_loss
        self.log("cyc_AB", cycAB_loss, on_step=True, prog_bar=True)
        self.log("cyc_BA", cycBA_loss, on_step=True, prog_bar=True)
        self.log("dsc_AB", dscAB_loss, on_step=True, prog_bar=True)
        self.log("dsc_BA", dscBA_loss, on_step=True, prog_bar=True)
        gen_optimzer.zero_grad()
        self.manual_backward(loss)
        gen_optimzer.step()
        self.untoggle_optimizer(gen_optimzer)

        return mid_A_hat, mid_B_hat

    def training_step_discriminator(self, batch, A_hat, B_hat):
        _, dsc_optimizer = self.optimizers()
        self.toggle_optimizer(dsc_optimizer)

        A, B = batch.values()

        """I. Discriminator A"""
        dscA_fake_loss = self.disc_loss(self.discA(A_hat.detach()), disc_update=True, real=False)
        dscA_real_loss = self.disc_loss(self.discA(A), disc_update=True, real=True)

        dscA_loss = 0.5 * (dscA_fake_loss + dscA_real_loss)

        """II. Discriminator B"""
        dscB_fake_loss = self.disc_loss(self.discB(B_hat.detach()), disc_update=True, real=False)
        dscB_real_loss = self.disc_loss(self.discB(B), disc_update=True, real=True)

        dscB_loss = 0.5 * (dscB_fake_loss + dscB_real_loss)

        self.log("d_A_fake", dscA_fake_loss, on_step=True, prog_bar=True)
        self.log("d_A_real", dscA_real_loss, on_step=True, prog_bar=True)
        self.log("d_B_fake", dscB_fake_loss, on_step=True, prog_bar=True)
        self.log("d_B_real", dscB_real_loss, on_step=True, prog_bar=True)

        """III. Backprop"""
        loss = dscA_loss + dscB_loss
        dsc_optimizer.zero_grad()
        self.manual_backward(loss)
        dsc_optimizer.step()
        self.untoggle_optimizer(dsc_optimizer)

    def on_train_epoch_end(self):
        gen_scheduler, dsc_scheduler = self.lr_schedulers()
        gen_scheduler.step()
        dsc_scheduler.step()

    def training_step(self, batch, batch_idx):
        A_hat, B_hat = self.training_step_generators(batch)
        self.training_step_discriminator(batch, A_hat, B_hat)

    def validation_step(self, batch, batch_idx):
        A, B = batch.values()
        B_hat = self.generatorAtoB(A)
        A_hat = self.generatorBtoA(B)

        B_idt = self.generatorAtoB(B)
        A_idt = self.generatorBtoA(A)

        """ MI Calculation """
        self.mi_A.append(mutual_information(B.flatten().cpu().numpy(), B_hat.flatten().cpu().numpy()))
        self.mi_B.append(mutual_information(A.flatten().cpu().numpy(), A_hat.flatten().cpu().numpy()))

        """Sample logging"""
        if np.random.random() < 0.1:
            for i in range(batch['A'].shape[2]):
                plt.figure(figsize=(16, 16))
                fig, axs = plt.subplots(2, 3)

                axs[0, 0].imshow(batch['A'].cpu().numpy()[0][0][i], cmap="gray")
                axs[0, 0].axis('off')
                axs[0, 0].set_title('A')

                axs[0, 1].imshow(B_hat.cpu().numpy()[0][0][i], cmap="gray")
                axs[0, 1].axis('off')
                axs[0, 1].set_title('B_hat')

                axs[0, 2].imshow(B_idt.cpu().numpy()[0][0][i], cmap="gray")
                axs[0, 2].axis('off')
                axs[0, 2].set_title('B_idt')

                axs[1, 0].imshow(batch['B'].cpu().numpy()[0][0][i], cmap="gray")
                axs[1, 0].axis('off')
                axs[1, 0].set_title('B')

                axs[1, 1].imshow(A_hat.cpu().numpy()[0][0][i], cmap="gray")
                axs[1, 1].axis('off')
                axs[1, 1].set_title('A_hat')

                axs[1, 2].imshow(A_idt.cpu().numpy()[0][0][i], cmap="gray")
                axs[1, 2].axis('off')
                axs[1, 2].set_title('A_idt')

                os.makedirs('logs/samples/', exist_ok=True)
                plt.savefig(f'logs/samples/fig_{self.current_epoch}_{batch_idx}_{i}')
                plt.close('all')

    def on_validation_epoch_end(self):
        mi_A_score, mi_B_score = np.average(self.mi_A), np.average(self.mi_B)
        self.log("mi_A", mi_A_score)
        self.log("mi_B", mi_B_score)

        print(f"\nEpoch {self.current_epoch} Score --- MI A: {mi_A_score} MI B: {mi_B_score}\n")

        self.log("LR", self.optimizers()[0].param_groups[0]['lr'], prog_bar=True, on_epoch=True)

        self.mi_A, self.mi_B = [], []

    def configure_optimizers(self):
        gen_parameters = list(self.generatorAtoB.parameters()) + list(self.generatorBtoA.parameters())
        gen_optimizer = optim.Adam(gen_parameters, lr=2e-4, betas=(0.5, 0.999))

        dsc_parameters = list(self.discA.parameters()) + list(self.discB.parameters())
        dsc_optimizer = optim.Adam(dsc_parameters, lr=2e-4, betas=(0.5, 0.999))

        # Scheduler
        gen_scheduler = optim.lr_scheduler.LambdaLR(
            gen_optimizer, lr_lambda=LambdaLR(100, 50).step
        )
        dsc_scheduler = optim.lr_scheduler.LambdaLR(
            dsc_optimizer, lr_lambda=LambdaLR(100, 50).step
        )
        return (
            {
                "optimizer": gen_optimizer,
                "lr_scheduler": gen_scheduler,
            },
            {
                "optimizer": dsc_optimizer,
                "lr_scheduler": dsc_scheduler
            }
        )
