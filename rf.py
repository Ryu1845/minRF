# implementation of Rectified Flow for simple minded people like me.
import argparse
from pathlib import Path
import sys
sys.path.append("./BigVGAN")

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import librosa

from meldataset import MelDataset


class RF:
    def __init__(self, model, ln=True):
        self.model = model
        self.ln = ln

    def forward(self, x, cond):
        b = x.size(0)
        if self.ln:
            nt = torch.randn((b,)).to(x.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,)).to(x.device)
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
        zt = (1 - texp) * x + texp * z1
        vtheta = self.model(zt, t, cond)
        batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]
        return batchwise_mse.mean(), ttloss

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)

            vc = self.model(z, t, cond)
            if null_cond is not None:
                vu = self.model(z, t, null_cond)
                vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)
        return images


class AudioMNISTMel(MelDataset):
    def __init__(self, set_file: str, *args, **kwargs):
        with open(set_file) as set_file:
            training_files = []
            labels = []
            for line in set_file:
                path, label = line.strip().split(',')
                training_files.append(path)
                labels.append(int(label))
        super().__init__(training_files, *args, **kwargs)
        self.labels = labels
    
    def __getitem__(self, idx):
        mel, audio, filename, mel_loss = super().__getitem__(idx)
        mel = torch.nn.functional.pad(mel, (0, 100-mel.shape[1]))
        label = self.labels[idx]
        return mel, torch.tensor(label)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

if __name__ == "__main__":
    # train class conditional RF on mnist.
    import numpy as np
    import torch.optim as optim
    from tqdm import tqdm

    import wandb
    from dit import DiT_Llama
    import json

    with open('BigVGAN/configs/bigvgan_22khz_80band.json') as config_f:
        config=json.load(config_f)
    h = AttrDict(config)

    training_filelist = "train.csv"
    val_filelist = "val.csv"
    test_filelist = "test.csv"
    trainset = AudioMNISTMel(
        training_filelist,
        h,
        h.segment_size,
        h.n_fft,
        h.num_mels,
        h.hop_size,
        h.win_size,
        h.sampling_rate,
        h.fmin,
        h.fmax,
        shuffle=False if h.num_gpus > 1 else True,
        fmax_loss=h.fmax_for_loss,
        device=torch.device("cuda"),
        is_seen=True,
        split=False,
    )
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=256, num_workers=8)

    model = DiT_Llama(
            80, 32, dim=2048, n_layers=16, n_heads=16, num_classes=10
        ).cuda()

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {model_size}, {model_size / 1e6}M")

    rf = RF(model)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = torch.nn.MSELoss()

    #wandb.init(project=f"rf_{dataset_name}")
    epoch = 0
    glob=Path('./').glob('*.pt')
    ckpts = sorted(list(glob))
    if ckpts:
        ckpt = ckpts[-1]
        print(f"Loading {ckpt}")
        ckpt = torch.load(ckpt)
        state_dict = ckpt["state"]
        epoch = ckpt["epoch"]
        model.load_state_dict(state_dict)

    for epoch in range(epoch, 100):
        rf.model.eval()
        with torch.no_grad():
            cond = torch.arange(0, 16).cuda() % 10
            uncond = torch.ones_like(cond) * 10

            init_noise = torch.randn(16, 80, 100).cuda()
            images = rf.sample(init_noise, cond, uncond)
            # image sequences to gif
            gif = []
            for idx, image in enumerate(images):
                fig, ax = plt.subplots()
                img = librosa.display.specshow(image[0].cpu().numpy(),  x_axis='time', y_axis='log', ax=ax)
                ax.set(title='Using a logarithmic frequency axis')
                fig.colorbar(img, ax=ax, format="%+2.f dB")
                fig.savefig(f'contents/sample_{epoch}_{idx}.png')
                plt.close()
        rf.model.train()
        lossbin = {i: 0 for i in range(10)}
        losscnt = {i: 1e-6 for i in range(10)}
        for i, (x, c) in tqdm(enumerate(dataloader)):
            x, c = x.cuda(), c.cuda()
            optimizer.zero_grad()
            loss, blsct = rf.forward(x, c)
            loss.backward()
            optimizer.step()

            #wandb.log({"loss": loss.item()})
            print(loss.item())

            # count based on t
            for t, l in blsct:
                lossbin[int(t * 10)] += l
                losscnt[int(t * 10)] += 1

        # log
        for i in range(10):
            print(f"Epoch: {epoch}, {i} range loss: {lossbin[i] / losscnt[i]}")

        #wandb.log({f"lossbin_{i}": lossbin[i] / losscnt[i] for i in range(10)}
        print(f"Saving checkpoint_{epoch}.pt")
        torch.save(dict(state=model.state_dict(), epoch=epoch), f'checkpoint_{epoch}.pt')
