from diffusers import DiffusionPipeline
import torch.nn as nn
import torch
from torch.cuda.amp import custom_bwd, custom_fwd
import einops

from scipy.spatial import Delaunay
import numpy as np
from torch.nn import functional as nnf
from torchvision import transforms
def get_data_augs():
    augmentations = []
    augmentations.append(transforms.RandomPerspective(
        fill=1, p=1.0, distortion_scale=0.5))
    augmentations.append(transforms.RandomResizedCrop(
        size=(256,256), scale=(0.4, 1), ratio=(1.0, 1.0)))
    augment_trans = transforms.Compose(augmentations)
    return augment_trans


# =============================================
# ===== Helper function for SDS gradients =====
# =============================================
class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


# ========================================================
# ===== Basic class to extend with SDS loss variants =====
# ========================================================
class SDSLossBase(nn.Module):

    _global_pipe = None

    def __init__(self, caption, model_name, device="cuda", reuse_pipe=True):
        super(SDSLossBase, self).__init__()

        self.caption = caption
        self.model_name = model_name
        self.device = device

        # initiate a diffusion pipeline if we don't already have one / don't want to reuse it for both paths
        self.maybe_init_pipe(reuse_pipe) 

        self.alphas = self.pipe.scheduler.alphas_cumprod.to(self.device)
        self.sigmas = (1 - self.pipe.scheduler.alphas_cumprod).to(self.device)


        self.text_embeddings = self.embed_text(self.caption)

        del self.pipe.tokenizer
        del self.pipe.text_encoder

    def maybe_init_pipe(self, reuse_pipe):
        model_name=self.model_name
        if reuse_pipe:
            if SDSLossBase._global_pipe is None:
                SDSLossBase._global_pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16, variant="fp16")
                SDSLossBase._global_pipe = SDSLossBase._global_pipe.to(self.device)
            self.pipe = SDSLossBase._global_pipe
        else:
            self.pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16, variant="fp16")
            self.pipe = self.pipe.to(self.device)

    def embed_text(self, caption):
        # tokenizer and embed text
        text_input = self.pipe.tokenizer(caption, padding="max_length",
                                         max_length=self.pipe.tokenizer.model_max_length,
                                         truncation=True, return_tensors="pt")
        uncond_input = self.pipe.tokenizer([""], padding="max_length",
                                         max_length=text_input.input_ids.shape[-1],
                                         return_tensors="pt")
        with torch.no_grad():
            text_embeddings = self.pipe.text_encoder(text_input.input_ids.to(self.device))[0]
            uncond_embeddings = self.pipe.text_encoder(uncond_input.input_ids.to(self.device))[0]
            
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        text_embeddings = text_embeddings.repeat_interleave(1, 0)

        return text_embeddings

        
    def prepare_latents(self, x):

        data_augs = get_data_augs()
        augmented_pair = data_augs(x.squeeze(0))
        x_aug = augmented_pair.unsqueeze(0)

        x = x_aug * 2. - 1. # encode rendered image, values should be in [-1, 1]
        
        with torch.cuda.amp.autocast():
            batch_size, num_frames, channels, height, width = x.shape # [1, K, 3, 256, 256], for K frames
            x = x.reshape(batch_size * num_frames, channels, height, width) # [K, 3, 256, 256], for the VAE encoder
            init_latent_z = (self.pipe.vae.encode(x).latent_dist.sample()) # [K, 4, 32, 32]
            frames, channel, h_, w_ = init_latent_z.shape
            init_latent_z = init_latent_z[None, :].reshape(batch_size, num_frames, channel, h_, w_).permute(0, 2, 1, 3, 4) # [1, 4, K, 32, 32] for the video model
            
        latent_z = self.pipe.vae.config.scaling_factor * init_latent_z  # scaling_factor * init_latents

        return latent_z

    def add_noise_to_latents(self, latent_z, timestep, return_noise=True, eps=None):
        
        # sample noise if not given some as an input
        if eps is None:
            eps = torch.randn_like(latent_z)

        # zt = alpha_t * latent_z + sigma_t * eps
        noised_latent_zt = self.pipe.scheduler.add_noise(latent_z, eps, timestep)

        if return_noise:
            return noised_latent_zt, eps

        return noised_latent_zt

    def drop_nans(self, grads):
        if not torch.isfinite(grads).all():
            print("Disturbed: Grads contain non-finite values. Replacing NaN values. Try another seed.")
            assert False
        return torch.nan_to_num(grads.detach().float(), 0.0, 1.0, -1.0)

    def get_grad_weights(self, timestep):
        return (1 - self.alphas[timestep])

    def sds_grads(self, latent_z, **sds_kwargs):
        with torch.no_grad():
            # sample timesteps
            timestep = torch.randint(
                low=200,
                high=400,  # avoid highest timestep | diffusion.timesteps=1000
                size=(latent_z.shape[0],),
                device=self.device, dtype=torch.long)

            # add noise
            noised_latent_zt, eps = self.add_noise_to_latents(latent_z, timestep, return_noise=True)

            # denoise
            z_in = torch.cat([noised_latent_zt] * 2)  # expand latents for classifier free guidance
            timestep_in = torch.cat([timestep] * 2)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                eps_t_uncond, eps_t = self.pipe.unet(z_in, timestep_in, encoder_hidden_states=self.text_embeddings).sample.float().chunk(2)
            
            eps_t = eps_t_uncond + 100 * (eps_t - eps_t_uncond)

            w = self.get_grad_weights(timestep)
            grad_z = w * (eps_t - eps)

            grad_z = self.drop_nans(grad_z)

        return grad_z

# =======================================
# =========== Basic SDS loss  ===========
# =======================================
class SDSVideoLoss(SDSLossBase):
    def __init__(self, caption, model_name, device="cuda", reuse_pipe=True):
        super(SDSVideoLoss, self).__init__(caption, model_name, device, reuse_pipe=reuse_pipe)

    def forward(self, x, grad_scale=1.0):
        latent_z = self.prepare_latents(x)

        grad_z = grad_scale * self.sds_grads(latent_z)

        sds_loss = SpecifyGradient.apply(latent_z, grad_z)

        return sds_loss    
