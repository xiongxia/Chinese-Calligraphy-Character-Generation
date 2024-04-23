# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision
from tqdm import tqdm
from torch import optim
import copy
import argparse
import json
from diffusers import AutoencoderKL
from unet import UNetModel
import wandb
import os
from tool import save_single_images,dataset_path,get_val_char,save_images,setup_logging,_extract_into_tensor,mean_flat,load_specific_dict,latent2image,image2latent,getFileList
import random
import numpy
from skimage import morphology

MAX_CHARS = 28
OUTPUT_MAX_LEN = MAX_CHARS
vocab_size = 518
char_classes = vocab_size
full_dict = {}

with open("char2compont.json", 'r', encoding='gbk') as f:
    char2compont =json.load(f)

def label_padding(labels):
    compont = char2compont[labels]
    if compont == None:
        print(labels)
    compont = compont.split(',')
    ll = np.array(compont)
    ll = list(ll)
    num = OUTPUT_MAX_LEN - len(ll)
    if not num == 0:
        ll.extend([0] * num)  # replace PAD_TOKEN
    return ll

class IAMDataset(Dataset):
    def __init__(self, full_dict, image_path, writer_dict, font_path, transforms=None):
        self.data_dict = full_dict
        self.image_path = image_path
        self.writer_dict = writer_dict
        self.transforms = transforms
        self.font_path = font_path
        self.indices = list(full_dict.keys())
        self.sty_id = writer_dict.keys()
        self.sty2image = {}

        for key,value in writer_dict.items():
            id = key
            image_name = []
            for i in full_dict.keys():
                if full_dict[i]['s_id'] == id:
                    image_name.append(full_dict[i]['image'])
            self.sty2image[id] = image_name

    def get_sty_image(self,sty_id):
        return random.choice(self.sty2image[sty_id])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image_name = self.data_dict[self.indices[idx]]['image']
        label = self.data_dict[self.indices[idx]]['label']
        wr_id = self.data_dict[self.indices[idx]]['s_id']
        sty_image = self.get_sty_image(wr_id)
        wr_id = torch.tensor(self.writer_dict[wr_id]).to(torch.int64)

        img_path = os.path.join(self.image_path, "sty_%d" % (wr_id), sty_image)
        image = Image.open(img_path).convert('RGB')
        sty_image = self.transforms(image)

        img_path = os.path.join(self.image_path, "sty_%d" % (wr_id), image_name)

        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)

        img_path = os.path.join(self.font_path, "sty_%d" % (wr_id), image_name)
        img_path = os.path.join(self.font_path, "sty_%d" % (wr_id), image_name)
        font_image = Image.open(img_path).convert('RGB')
        font_image = self.transforms(font_image)

        word_embedding = label_padding(label)
        word_embedding = np.array(word_embedding, dtype="int64")
        word_embedding = torch.from_numpy(word_embedding).long()

        return image, word_embedding, wr_id, font_image, label, sty_image

#EMA
class EMA:
    '''
    EMA is used to stabilize the training process of diffusion models by
    computing a moving average of the parameters, which can help to reduce
    the noise in the gradients and improve the performance of the model.
    '''

    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

#扩散模型
class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=(256, 256), args=None):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(args.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.sqrt_alphas_cumprod = np.sqrt(self.alpha_hat.cpu())
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alpha_hat.cpu())

        self.img_size = img_size
        self.device = args.device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sampling(self, model, vae, n, x_text, labels, args, font_images=None, sty_images=None):
        model.eval()
        tensor_list = []

        with torch.no_grad():
            words = [x_text] * n
            for word in words:
                transcript = label_padding(word)  # self.transform_text(transcript)
                word_embedding = np.array(transcript, dtype="int64")
                word_embedding = torch.from_numpy(word_embedding).long()  # float()
                tensor_list.append(word_embedding)

            text_features = torch.stack(tensor_list)
            text_features = text_features.to(args.device)
            labels = torch.tensor([labels], dtype=torch.long).to(args.device)
            sty_images = sty_images.unsqueeze(0).to(args.device)

            if font_images is not None:
                font_images = font_images.unsqueeze(0).to(args.device)

            if args.latent == True:
                font_images = image2latent(vae, font_images)

            x = torch.randn((n, 4, self.img_size[0] // 8, self.img_size[1] // 8)).to(args.device)

            # atts = []

            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                # predicted_noise,att = model(x, font_images, t, text_features, labels, original_context=x_text, sty_image=sty_images)
                predicted_noise = model(x, font_images, t, text_features, labels, original_context=x_text,
                                             sty_image=sty_images)

                # atts.append(att)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        if args.latent == True:
            images = latent2image(vae,x)

        model.train()

        return images


    def samplle_x0(self, predicted_noise, x_t, t):
        alpha = self.alpha[t][:, None, None, None]
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        beta = self.beta[t][:, None, None, None]

        noise = torch.randn_like(x_t)

        x = 1 / torch.sqrt(alpha) * (x_t - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        return x


def train(diffusion, model, ema, ema_model, vae, optimizer, mse_loss, loader, args,transforms):
    model.train()

    print('Training started....')

    with open(dataset_path + "/test.gt.filter", 'r', encoding='gbk') as f:
        val_data = f.read().split("\n")
    words = []
    all_font_img = []
    all_sty_img = []
    all_targt_img = []
    labels = []
    for item in val_data[:10]:
        item = item.split(';')
        words.append(item[2])
        sty_id = item[1].split('_')[0]
        image_path = os.path.join(dataset_path, 'font/test/sty_%s' % (sty_id), item[1])
        font_image = Image.open(image_path).convert('RGB')
        font_image = transforms(font_image)
        all_font_img.append(font_image)
        image_path = os.path.join(dataset_path, 'gt/test/sty_%s' % (sty_id), item[1])
        targt_image = Image.open(image_path).convert('RGB')
        targt_image = transforms(targt_image)
        all_targt_img.append(targt_image)
        image_path = os.path.join(dataset_path, 'gt/train/sty_%s' % (sty_id))
        image_path = getFileList(image_path, [], 'jpg')
        image_path = np.random.choice(image_path, size=1)[0]
        sty_image = Image.open(image_path).convert('RGB')
        sty_image = transforms(sty_image)
        all_sty_img.append(sty_image)
        labels.append(int(sty_id))

    steps = 0
    for epoch in range(args.resume_epochs+1,args.epochs):
        print('Epoch:', epoch)
        pbar = tqdm(loader)
        for i, (images, word, s_id, font_image, char, sty_image) in enumerate(pbar):
            images = images.to(args.device)
            text_features = word.to(args.device)
            font_image = font_image.to(args.device)
            sty_image = sty_image.to(args.device)
            s_id = s_id.to(args.device)
            if args.latent == True:
                images = image2latent(vae,images)
                font_image = image2latent(vae,font_image)

            t = diffusion.sample_timesteps(images.shape[0]).to(args.device)
            x_t, noise = diffusion.noise_images(images, t)

            predicted_noise = model(x_t, font_image, timesteps=t, context=text_features, y=s_id,original_context=char,sty_image=sty_image)
            predicted_x0 = diffusion.samplle_x0(predicted_noise,x_t,t)
            loss_latent = mse_loss(images, predicted_x0)
            loss = mse_loss(noise, predicted_noise)
            total_loss = loss + args.gradient_scale*loss_latent
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)
            pbar.set_postfix(MSE=loss.item(),loss_latent=loss_latent.item(), total_loss=total_loss.item())
            steps += 1

            if args.wandb_log == True:
                wandb.log({"Train Loss": total_loss.item()})

        if (epoch+1) % 50 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_path, "models", "ckpt_%d.pt" % (epoch)))
            torch.save(ema_model.state_dict(), os.path.join(args.save_path, "models", "ema_ckpt_%d.pt" % (epoch)))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, "models", "optim_%d.pt" % (epoch)))
            idx = 0
            for x_text in words:
                ema_sampled_images = diffusion.sampling(ema_model, vae,n=1, x_text=x_text, labels=labels[idx], font_images=all_font_img[idx], sty_images=all_sty_img[idx], args=args)
                all_targt_img[idx] = all_targt_img[idx].to(args.device)
                save_image = torch.cat([all_targt_img[idx].unsqueeze(0), ema_sampled_images], dim=0)
                sampled_ema = save_images(save_image,os.path.join(args.save_path, 'images', f"{x_text}_{idx}_{epoch}.jpg"))
                idx += 1
                if args.wandb_log == True:
                    wandb_sampled_ema = wandb.Image(sampled_ema, caption=f"{x_text}_{epoch}")
                    wandb.log({f"Sampled images": wandb_sampled_ema})
        torch.save(model.state_dict(), os.path.join(args.save_path, "models", "ckpt_last.pt"))
        torch.save(ema_model.state_dict(), os.path.join(args.save_path, "models", "ema_ckpt_last.pt"))
        torch.save(optimizer.state_dict(), os.path.join(args.save_path, "models", "optim_last.pt"))

def main():
    '''Main function'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--img_size', type=int, default=(256, 256))
    parser.add_argument('--iam_path', type=str, default=dataset_path + '/gt/train', help='path to iam dataset')
    parser.add_argument('--gt_train', type=str, default=dataset_path + '/tr_va.gt.filter')
    parser.add_argument('--gt_test', type=str, default=dataset_path + '/test.gt.filter')
    parser.add_argument('--font_path', type=str, default=dataset_path + '/font/train')
    parser.add_argument('--id', type=str, default='0')
    # UNET parameters
    parser.add_argument('--channels', type=int, default=8, help='if latent is True channels should be 4, else 3')
    parser.add_argument('--out_channels', type=int, default=4, help='if latent is True channels should be 4, else 3')
    parser.add_argument('--emb_dim', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./output')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--wandb_log', type=bool, default=False)
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--writer_dict', type=str, default=dataset_path + '/writers_dict.json')
    parser.add_argument('--dataset_dict', type=str, default=dataset_path + '/dataset_dict.json')
    parser.add_argument('--stable_dif_path', type=str, default=r'./vae', help='path to stable diffusion')
    parser.add_argument('--sty_pretrained', type=str,default=r'./sty_encoder.pth')
    parser.add_argument('--resume_epochs', type=int, default=0)
    parser.add_argument('--gradient_scale', type=int, default=2)

    args = parser.parse_args()
    if args.wandb_log == True:
        wandb.init(project='DIFFUSION_CAILL', name=f'{args.save_path+args.id}', config=args)
        wandb.config.update(args)
    print(args)
    args.save_path = os.path.join(args.save_path, args.id)
    # create save directories
    setup_logging(args)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 0.5)
    ])

    with open(args.dataset_dict, "r") as f:
        full_dict = json.load(f)
    with open(args.writer_dict, "r") as f:
        wr_dict = json.load(f)

    train_ds = IAMDataset(full_dict, args.iam_path, wr_dict, args.font_path, transforms=transforms)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    unet = UNetModel(image_size=args.img_size, in_channels=args.channels, model_channels=args.emb_dim,
                     out_channels=args.out_channels, num_res_blocks=args.num_res_blocks, attention_resolutions=(1, 1),
                     channel_mult=(1, 1), num_heads=args.num_heads, context_dim=args.emb_dim,args=args).to(args.device)

    optimizer = optim.AdamW(unet.parameters(), lr=0.0001)

    mse_loss = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, args=args)

    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)

    ### sty_encoder
    if len(args.sty_pretrained) > 0:
        checkpoint = torch.load(args.sty_pretrained, map_location='cpu')
        unet.sty_encoder.load_state_dict(checkpoint, strict=False)

    # frozen sty_encoder
    for p in unet.sty_encoder.parameters():
        p.requires_grad = False

    if args.resume_epochs > 0:
        print("loading pre-trained model...")
        unet.load_state_dict(torch.load(f'{args.save_path}/models/ckpt_%d.pt'%(args.resume_epochs)))
        optimizer.load_state_dict(torch.load(f'{args.save_path}/models/optim_%d.pt'%(args.resume_epochs)))
        ema_model.load_state_dict(torch.load(f'{args.save_path}/models/ema_ckpt_%d.pt'%(args.resume_epochs)))
    else:
        pass


    if args.latent == True:
        print('Latent is true - Working on latent space')
        vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
        vae = vae.to(args.device)

        # Freeze vae and text_encoder
        vae.requires_grad_(False)
    else:
        print('Latent is false - Working on pixel space')
        vae = None

    if args.wandb_log == True:
        wandb.watch(unet, log="all")

    train(diffusion, unet, ema, ema_model, vae, optimizer, mse_loss, train_loader, args, transforms)


if __name__ == "__main__":
    main()


