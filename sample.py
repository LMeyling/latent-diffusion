from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from tqdm.auto import tqdm
from PIL import Image
import torch
class Sampler:
    def __init__(self, device="cpu"):

        self.device = device
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").to(device)
        self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    def sample(self, prompt, height=512, width=512, steps=128, guidance=7.5, seed=0, return_steps=False, x_steps=10):

        generator = torch.manual_seed(seed)
        inter_res = []

        # Prompt processing to text embeddings
        text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True,return_tensors="pt")
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer([""] * 1, padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # initial latent generation
        latents = torch.randn((1, self.unet.in_channels, height // 8, width // 8),generator= generator)
        latents = latents.to(self.device)
        self.scheduler.set_timesteps(steps)
        latents = latents * self.scheduler.init_noise_sigma
        c = 0

        # denoising loop
        for t in tqdm(self.scheduler.timesteps):

            c+=1
            # predict noise
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # apply guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance * (noise_pred_text - noise_pred_uncond)

            # save current image from latent
            if return_steps and c % x_steps == 0:
                with torch.no_grad():
                    image = self.vae.decode(1 / 0.18215 * (latents-noise_pred)).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
                images = (image * 255).round().astype("uint8")
                pil_images = [Image.fromarray(image) for image in images]
                inter_res.append(pil_images[0])

            # apply denosing
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample



        # decode latent to image
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]


        return pil_images[0],inter_res