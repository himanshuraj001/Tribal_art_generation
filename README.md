# 🎨 Tribal Art Generation with Fine-Tuned Stable Diffusion (DreamBooth)

This project fine-tunes **Stable Diffusion** using **DreamBooth** to generate tribal art styles:
- **Gond**
- **Kangra**
- **Kerala Mural**
- **Warli**

Our contributions:
- ✅ Introduced **style loss**, **perceptual loss**, and **latent MSE loss** for improved fidelity.  
- ✅ Enabled **mixed-style generation** (e.g., tree in Gond style, person in Kerala mural).  
- ✅ Used a **ResNet-based classifier** to validate style correctness.  
- ✅ Leveraged **BLIP** for captioning and measured **style/content correctness** using similarity metrics.  

---

## 📂 Hugging Face Models

We provide fine-tuned models via [🤗 Hugging Face Diffusers](https://huggingface.co/docs/diffusers/index):

- 🖼️ **Gond Fine-Tuned SD Model** → [Hugging Face Link](https://huggingface.co/himanshu0510/gond_style_loss-art-fine-tuned)  
- 🖼️ **Kangra Fine-Tuned SD Model** → [Hugging Face Link](https://huggingface.co/himanshu0510/kangra_style_loss-art-fine-tuned)  
- 🖼️ **Kerala Mural Fine-Tuned SD Model** → [Hugging Face Link](https://huggingface.co/himanshu0510/kerla_style_loss-art-fine-tuned)
- 🖼️ **Warli Fine-Tuned SD Model** → [Hugging Face Link](https://huggingface.co/himanshu0510/warli_style_loss-art-fine-tuned)  

*(replace the links above with actual HF model URLs once uploaded)*

---

## 📊 Methodology

### 1. Training Objective

We fine-tuned Stable Diffusion’s U-Net with a composite objective function:

\[
\mathcal{L}_{total} = \lambda_{recon}\mathcal{L}_{recon} + \lambda_{style}\mathcal{L}_{style} + \lambda_{perc}\mathcal{L}_{perc} + \lmbda_{latent}\mathcal{L}_{latent}
\]

- **Reconstruction Loss (\(\mathcal{L}_{recon}\))**: Standard diffusion reconstruction objective.  
- **Style Loss (\(\mathcal{L}_{style}\))**: Gram matrix matching of VGG-19 feature maps.  
\[
\mathcal{L}_{style} = \sum_{l} \| G_l(F(x)) - G_l(F(y_{style})) \|_2^2
\]
- **Perceptual Loss (\(\mathcal{L}_{perc}\))**: L2 distance in VGG feature space.  
\[
\mathcal{L}_{perc} = \| \phi(x) - \phi(y_{style}) \|_2^2
\]
- **Latent MSE Loss (\(\mathcal{L}_{latent}\))**: Mean squared error in the latent space of Stable Diffusion’s encoder.  

### 2. Mixed-Style Conditioning

We allow conditioning tokens for different objects in the prompt.  
For example:  
