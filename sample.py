import os
from typing import Any
import math

import hydra
import torch
from pytorch_lightning.lite import LightningLite

import omegaconf
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate

from torchvision.utils import save_image, make_grid

from models.autoencoder.vqgan import VQEncoderInterface, VQDecoderInterface
from utils.helpers import ensure_path_join, denormalize_to_zero_to_one

import sys
sys.path.insert(0, 'idiff-face-iccv2023-code/')


class DiffusionSamplerLite(LightningLite):
    def run(self, cfg) -> Any:
        train_cfg_path = os.path.join(cfg.checkpoint.path, '.hydra', 'config.yaml')
        train_cfg = omegaconf.OmegaConf.load(train_cfg_path)

        # do not set seed to get different samples from each device
        self.seed_everything(cfg.sampling.seed * (1 + self.global_rank))

        # instantiate stuff from restoration config
        diffusion_model = instantiate(train_cfg.diffusion)

        # registrate model in lite
        diffusion_model = self.setup(diffusion_model)

        # load state dicts from checkpoint
        if cfg.checkpoint.global_step is not None:
            checkpoint_path = os.path.join(cfg.checkpoint.path, 'checkpoints', f'ema_averaged_model_{cfg.checkpoint.global_step}.ckpt')
        if cfg.checkpoint.use_non_ema:
            checkpoint_path = os.path.join(cfg.checkpoint.path, 'checkpoints', f'model.ckpt')
        else:
            checkpoint_path = os.path.join(cfg.checkpoint.path, 'checkpoints', 'ema_averaged_model.ckpt')

        diffusion_model.module.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

        # sample size
        size = (train_cfg.constants.input_channels, train_cfg.constants.image_size, train_cfg.constants.image_size)

        if train_cfg.latent_diffusion:
            # create VQGAN encoder and decoder for training in its latent space
            latent_encoder = VQEncoderInterface(
                first_stage_config_path=os.path.join(".", "models", "autoencoder",
                                                     "first_stage_config.yaml"),
                encoder_state_dict_path=os.path.join(".", "models", "autoencoder",
                                                     "first_stage_encoder_state_dict.pt")
            )

            size = latent_encoder(torch.ones([1, *size])).shape[-3:]
            del latent_encoder

            latent_decoder = VQDecoderInterface(
                first_stage_config_path=os.path.join(".", "models", "autoencoder",
                                                     "first_stage_config.yaml"),
                decoder_state_dict_path=os.path.join(".", "models", "autoencoder",
                                                     "first_stage_decoder_state_dict.pt")
            )
            latent_decoder = self.setup(latent_decoder)
            latent_decoder.eval()
        else:
            latent_decoder = None

        # build path to save the samples


        assert cfg.sampling.contexts_file is not None

        contexts = torch.load(cfg.sampling.contexts_file)
        assert len(contexts) >= cfg.sampling.n_contexts

        if type(contexts) == dict:
            input_contexts_name = cfg.sampling.contexts_file.split("/")[-1].split(".")[0]
            model_name = cfg.checkpoint.path.split("/")[-1]
            context_ids = list(contexts.keys())[:cfg.sampling.n_contexts]
        else:
            exit(1)

        if cfg.checkpoint.use_non_ema:
            model_name += "_non_ema"
        elif cfg.checkpoint.global_step is not None:
            model_name += f"_{cfg.checkpoint.global_step}"

        samples_dir = ensure_path_join("samples", model_name, input_contexts_name)


        length_before_filter = len(context_ids)
        context_ids = list(filter(lambda i: not os.path.isfile(os.path.join(samples_dir, f"{i}.png")), context_ids))
        print(f"Skipped {length_before_filter - len(context_ids)} context ids, because for them files already seem to "
              f"exist!")
        context_ids = self.split_across_devices(context_ids)

        if self.global_rank == 0:
            with open(ensure_path_join(f"{samples_dir}.yaml"), "w+") as f:
                OmegaConf.save(config=cfg, f=f.name)

        for id_name in context_ids:
            prefix = id_name

            context = torch.from_numpy(contexts[id_name])
            context = context.repeat(cfg.sampling.batch_size, 1).cuda()

            self.perform_sampling(
                diffusion_model=diffusion_model,
                n_samples=cfg.sampling.n_samples_per_context,
                size=size,
                batch_size=cfg.sampling.batch_size,
                samples_dir=samples_dir,
                prefix=prefix,
                context=context,
                latent_decoder=latent_decoder
            )

    @staticmethod
    def perform_sampling(
            diffusion_model, n_samples, size, batch_size, samples_dir,
            prefix: str = None, context: torch.Tensor = None,
            latent_decoder: torch.nn.Module = None):

        n_batches = math.ceil(n_samples / batch_size)

        samples_for_grid = []

        if context is not None:
            assert prefix is not None

        with torch.no_grad():
            for _ in range(n_batches):

                batch_samples = diffusion_model.sample(batch_size, size, context=context)

                with torch.no_grad():
                    if latent_decoder:
                        batch_samples = latent_decoder(batch_samples).cpu()

                batch_samples = denormalize_to_zero_to_one(batch_samples)

                samples_for_grid.append(batch_samples)

            samples = torch.cat(samples_for_grid, dim=0)[:n_samples]
            grid = make_grid(samples, nrow=4, padding=0)
            save_image(grid, ensure_path_join(samples_dir, f"{prefix}.png"))

    def split_across_devices(self, L):
        if type(L) is int:
            L = list(range(L))

        chunk_size = math.ceil(len(L) / self.world_size)
        L_per_device = [L[idx: idx + chunk_size] for idx in range(0, len(L), chunk_size)]
        while len(L_per_device) < self.world_size:
            L_per_device.append([])

        return L_per_device[self.global_rank]

    @staticmethod
    def spherical_interpolation(value, start, target):
        start = torch.nn.functional.normalize(start)
        target = torch.nn.functional.normalize(target)
        omega = torch.acos((start * target).sum(1))
        so = torch.sin(omega)
        res = (torch.sin((1.0 - value) * omega) / so).unsqueeze(1) * start + (
                torch.sin(value * omega) / so).unsqueeze(1) * target
        return res


@hydra.main(config_path='configs', config_name='sample_config', version_base=None)
def sample(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    sampler = DiffusionSamplerLite(devices="auto", accelerator="auto")
    sampler.run(cfg)


if __name__ == "__main__":

    sample()
