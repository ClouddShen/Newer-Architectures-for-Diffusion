import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet, UNet_BigGAN_Residual, UNet_anti_aliasing
from module_collections.DDPM_RDN import DDPM_RDN
from module_collections.DDPM_4ResBlocks import DDPM_4ResBlocks
from module_collections.DDPM_AdditionalResLink import DDPM_AdditionalResLink
from module_collections.DDPM_Deep import DDPM_Deep
from module_collections.DDPM_ConvNeXt import DDPM_ConvNeXt
import logging
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=32, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.fixed_noise = None

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
    
    # New sampling function to ensure the sampled images come from the same noise
    def sample_fixed_noise(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            if self.fixed_noise is None:
                self.fixed_noise = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            x = self.fixed_noise
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def sample_and_save(self, run_name, model, n, batchsize, epochs):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        num_chunk = n // batchsize
        for chunk_ind in range(num_chunk):
            with torch.no_grad():
                x = torch.randn((batchsize, 3, self.img_size, self.img_size)).to(self.device)
                for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                    t = (torch.ones(batchsize) * i).long().to(self.device)
                    predicted_noise = model(x, t)
                    alpha = self.alpha[t][:, None, None, None]
                    alpha_hat = self.alpha_hat[t][:, None, None, None]
                    beta = self.beta[t][:, None, None, None]
                    if i > 1:
                        noise = torch.randn_like(x)
                    else:
                        noise = torch.zeros_like(x)
                    x = 1 / torch.sqrt(alpha) * (
                            x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                        beta) * noise
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
            for index in range(x.shape[0]):
                img = x[index].permute(1, 2, 0).cpu().numpy()
                img = Image.fromarray(img)
                os.makedirs(os.path.join("./rds/samples", run_name, epochs), exist_ok=True)
                filepath = os.path.join("./rds/samples", run_name, epochs, f"{batchsize * chunk_ind + index}.jpg")
                img.save(filepath)

def sample_4096_for_models(run_name, net):
    torch.cuda.empty_cache()
    device = "cuda"
    model = net().to(device)
    epoch = 100
    ckpt = torch.load(os.path.join(f"./rds/models", run_name, f"{epoch}epochs_ckpt.pt"))
    model.load_state_dict(ckpt)
    diffusion = Diffusion(img_size=32, device=device)
    diffusion.sample_and_save(run_name, model, 4096, 128, f"{epoch}epochs")

def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet_anti_aliasing().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("./rds/runs", args.run_name))
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % args.save_per_num_epochs == args.save_per_num_epochs-1:
            sampled_images = diffusion.sample(model, n=images.shape[0])
            save_images(sampled_images, os.path.join("./rds/results", args.run_name, f"{epoch}.jpg"))
            torch.save(model.state_dict(), os.path.join("./rds/models", args.run_name, f"ckpt_{epoch}.pt"))


def launch(run_name, net):
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = run_name
    args.epochs = 100
    args.batch_size = 128
    args.image_size = 32
    args.dataset_path = r"./rds/datasets"
    args.device = "cuda"
    args.lr = 2e-4
    args.save_per_num_epochs = 50
    args.net = net()
    train(args)


if __name__ == '__main__':
    launch("DDPM_ConvNeXt", DDPM_ConvNeXt)
    # sample_4096_for_models("DDPM_CovNeXt", DDPM_ConvNeXt)
