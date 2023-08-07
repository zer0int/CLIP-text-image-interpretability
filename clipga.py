# -*- coding: utf-8 -*-
"""Ascending CLIPtext.ipynb

2023 GPT-4 & zer0int -- Twitter: @zer0int1
Adaptation of the original notebook by advadnoun, used with explicit permission to publish

# Original Author: Twitter @advadnoun ~ 2021:
Closed Test Ascending CLIPtext.ipynb
This is a notebook for determining descriptions that maximally match an image per CLIP using gradient ascent.

# Top
"""
###	SET 	clipmodel, training_iterations, batchsize 	below, depending on your hardware and preferences:

# VRAM use, batch size = 16, 	in GB:	   batch size = 4, GB:
#
# ViT-B/32 			 4.5			 3.8
# ViT-B/16			 9.5			 8.7
# ViT-L/14			N/A*			N/A*
# ViT-L/14@336px		N/A*			N/A*
#
# N/A*: CUDA OOM, >>24 GB VRAM needed, I don't have that.

clipmodel = 'ViT-B/32'
# available models = ['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'] -- if not, upgrade from git:OpenAI/CLIP

training_iterations = 200    # <50 will yield awfully imprecise results, >600 doesn't improve reasonably. Recommended 100-400.

batchsize = 16



# You can ignore what follows, as it doesn't typically need adjusting.


import imageio
import torchvision
import PIL.Image
from IPython import display
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
checkin_step = training_iterations - 1
import os
import sys
import clip
import kornia
import torch
import torch.nn.functional as F
import random
clip.available_models()
import numpy as np
import argparse
import glob
from multiprocessing import cpu_count
from ldmutil import parallel_data_prefetch
from tqdm import tqdm
from torchvision.transforms import Resize
import warnings
warnings.filterwarnings('ignore')
def get_clip_dimensions(clipmodel):
    model, preprocess = clip.load(clipmodel, jit=True)
    model = model.eval()
    for transform in preprocess.transforms:
        if isinstance(transform, Resize):
            input_dims = transform.size
            return input_dims
perceptor, preprocess = clip.load(clipmodel, jit=True)
perceptor = perceptor.eval()
input_dims = get_clip_dimensions(clipmodel)
parser = argparse.ArgumentParser(description="CLIP Gradient Ascent")
parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
args = parser.parse_args()


"""# Def"""

def displ(img, pre_scaled=True):
  img = np.array(img)[:,:,:]
  img = np.transpose(img, (1, 2, 0))
  if not pre_scaled:
    img = scale(img, 48*4, 32*4)
  imageio.imwrite(str(3) + '.png', np.array(img))
  return display.Image(str(3)+'.png')

"""# Internal tweaks"""

def clip_encode_text(gobble, text):
  x = torch.matmul(text, gobble.token_embedding.weight)  # [batch_size, n_ctx, d_model]

  x = x + gobble.positional_embedding
  x = x.permute(1, 0, 2)  # NLD -> LND

  x = gobble.transformer(x)
  x = x.permute(1, 0, 2)  # LND -> NLD
  x = gobble.ln_final(x)

  x = x[torch.arange(x.shape[0]), many_tokens + len(prompt) + 2] @ gobble.text_projection

  return x

"""# Settings"""

import warnings
warnings.filterwarnings('ignore')

batch_size = batchsize # You will want to change this unless you have massive VRAM. Try adjusting for a perfect fit with regard to your selected CLIP model and available VRAM.
many_tokens = 4 # You can also change this = number of predicted tokens.

# a prompt to use before the learned tokens/words
prompt = clip.tokenize('''''').numpy().tolist()[0]
prompt = [i for i in prompt if i != 0 and i != 49406 and i != 49407]

sideX = input_dims # was 288 RN50x4 and 224 for VIT-L/14 and 336 for VIT@336 and 372 for RN50x16
sideY = input_dims # was 288

# set the image to use
img_path = args.image_path

import os
img_name = os.path.splitext(os.path.basename(img_path))[0]

im = torch.tensor(imageio.imread(img_path).copy()).cuda().unsqueeze(0).permute(0, 3, 1, 2) / 255 # 0,3,1,2 . 255
im = F.interpolate(im, (sideX, sideY))

"""
# Setup parameters"""

torch.cuda.empty_cache()

class Pars(torch.nn.Module):
    def __init__(self):
        super(Pars, self).__init__()
        
        st = torch.zeros(batch_size, many_tokens, 49408).normal_()
        self.normu = torch.nn.Parameter(st.cuda())
        self.much_hard = 1000

        self.start = torch.zeros(batch_size, 1, 49408).cuda()
        self.start[:, :, 49406] = 1

        ptt = prompt

        self.prompt = torch.zeros(batch_size, len(ptt), 49408).cuda()
        for jk, pt in enumerate(ptt):
          self.prompt[:, jk, pt] = 1 
        
        self.pad = torch.zeros(batch_size, 77 - (many_tokens + len(prompt) + 1), 49408).cuda()
        self.pad[:, :, 49407] = 1

        
    def forward(self):
      self.soft = F.gumbel_softmax(self.normu, tau=self.much_hard, dim=-1, hard=True)
      fin = torch.cat([self.start, self.prompt, self.soft, self.pad], 1)
      return fin


lats = Pars().cuda()
mapper = [lats.normu]
optimizer = torch.optim.Adam([{'params': mapper, 'lr': 5}])
eps = 0

nom = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

augs = torch.nn.Sequential(
    kornia.augmentation.RandomAffine(degrees=10, translate=.1, p=.8).cuda(),
).cuda()

tok = clip.simple_tokenizer.SimpleTokenizer()

bests = {1000:'None', 1001:'None', 1002:'None', 1003:'None', 1004:'None'}

torch.argmax(lats(), 2)[0].clone().detach().cpu().numpy()

"""# Train"""

import warnings
warnings.filterwarnings('ignore')

def augment(into):
  into = augs(into)
  return into


def ascend_txt():
  global im
  iii = nom(augment(im[:,:3,:,:].expand(64, -1, -1, -1)))
  iii = perceptor.encode_image(iii).detach()
  lll = lats()
  tx = clip_encode_text(perceptor, lll)
  return -100*torch.cosine_similarity(tx.unsqueeze(0), iii.unsqueeze(1), -1).view(-1, batch_size).T.mean(1), lll

def train():
  loss1, lll = ascend_txt()
  loss = loss1.mean()
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  return loss1, lll

def checkin(loss, lll):
    unique_tokens = set()

    these = [tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist()).replace('', '').replace('', '') for kj in range(lll.shape[0])]

    for kj in range(lll.shape[0]):
        if loss[kj] < sorted(list(bests.keys()))[-1]:
            # Remove non-printable characters and replace them with a space
            cleaned_text = ''.join([c if c.isprintable() else ' ' for c in these[kj]])
            bests[loss[kj]] = cleaned_text
            bests.pop(sorted(list(bests.keys()))[-1], None)

    for j, k in zip(list(bests.values())[:5], list(bests.keys())[:5]):
        j = j.replace('<|startoftext|>', '')
        j = j.replace('<|endoftext|>', '')
        j = j.replace('\ufffd', '')
        j = j.replace('.', '')
        j = j.replace(';', '')
        j = j.replace('?', '')
        j = j.replace('!', '')
        j = j.replace('_', '')
        j = j.replace('-', '')
        j = j.replace('\\', '')
        j = j.replace('\'', '')
        j = j.replace('"', '')
        j = j.replace('^', '')
        j = j.replace('&', '')
        j = j.replace('#', '')
        j = j.replace(')', '')
        j = j.replace('(', '')
        j = j.replace('*', '')
        j = j.replace(',', '')

        #print(j, ' ') # not printing them as emojis etc. are non-printable characters in the console
        tokens = j.split()
        unique_tokens.update(tokens)

    with open(f"TOK/tokens_{img_name}.txt", "w", encoding='utf-8') as f:
        f.write(" ".join(unique_tokens))


def loop():
  for i in range(training_iterations):
    loss, lll = train()
    if i % checkin_step == 0:
      checkin(loss, lll)

loop()