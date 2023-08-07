# CLIP text-image interpretability
## Visual Interpretability / XAI Tool for CLIP ViT (Vision Transformer) models

![banner](https://github.com/zer0int/CLIP-text-image-interpretability/assets/132047210/35cfed98-3eed-42a6-84bd-31e27c33de2c)


## Credits & Prerequisites

- **CLIP Model**: Install this: [OpenAI/CLIP](https://github.com/openai/CLIP)
- **Transformer-MM-Explainability**: Install this: [hila-chefer/Transformer-MM-Explainability](https://github.com/hila-chefer/Transformer-MM-Explainability)
- **Original CLIP Gradient Ascent Script**: Used with permission by Twitter / X: [@advadnoun](https://twitter.com/advadnoun)
  
## Overview

In simple terms: Feeds an image to a CLIP ViT vision transformer to obtain "a CLIP opinion" / words (text tokens) about the image (gradient ascent), then uses the [token] + [image] pair to visualize what CLIP is "looking at" (attention visualization), producing an overlay "heatmap" image.

## Setup 

1. **Install OpenAI/CLIP and hila-chefer/Transformer-MM-Explainability**
2. **Put the contents of this repo into the "/Transformer-MM-Explainability" folder**
3. **Execute "python runall.py" from the command line, follow instructions**
4. **Or run the individual scripts separately, check runall.py for details**
5. *You should have most requirements from the prequisite installs (1.), except maybe kornia ("pip install kornia")*
6. *Requires a minimum amount of 4 GB VRAM (CLIP ViT-B/32). Check clipga.py and adjust batch size if you get a CUDA OOM, or to use a different model*


## What does a vision transformer "see"?

- Find out what CLIP's attention is on for a given image, explore bias as well as sophistication and broad concepts learned by the AI
- Use CLIP's "opinion" + heatmap image verification, then try to prompt your favorite text-to-image AI with those tokens. YES! Even the "crazy tokens"; after all, it's a CLIP steering the image towards your prompt inside a text-to-image AI system!

### Examples:

![what-clip-sees](https://github.com/zer0int/CLIP-text-image-interpretability/assets/132047210/8a54441b-15c1-4472-8218-f626483e6e30)

![attention-guided](https://github.com/zer0int/CLIP-text-image-interpretability/assets/132047210/646fe6eb-09f6-4481-b570-37c309955329)

![interoperthunderbirds](https://github.com/zer0int/CLIP-text-image-interpretability/assets/132047210/92fc4b11-f1ad-4278-8f73-bc016cce4afa)

---
