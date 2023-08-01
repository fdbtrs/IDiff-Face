# IDiff-Face: Synthetic-based Face Recognition through Fizzy Identity-conditioned Diffusion Models (Official)

[![arXiv](https://img.shields.io/badge/cs.CV-arXiv%3A1234.12345-009d81.svg)](todo)
[![ICCV2023](https://img.shields.io/badge/ICCV-2023-0083cc.svg)](todo)

**[Fadi Boutros](https://www.igd.fraunhofer.de/en/institute/employees/fadi-boutros.html), [Jonas Henry Grebe](https://www.linkedin.com/in/jonasgrebe/?originalSubdomain=de), [Naser Damer](https://www.igd.fraunhofer.de/de/institut/mitarbeitende/naser-damer.html) & [Arjan Kuijper](https://www.igd.fraunhofer.de/de/institut/mitarbeitende/arjan-kuijper.html)**

## Abstract

<img align="right" src="etc/idiff-face-overview.png" width=40%>

The availability of large-scale authentic face databases has been crucial to the significant advances made in face recognition research over the past decade. However, recent legal and ethical concerns led to the retraction of many of these databases by their creators, raising questions about the continuity of future face recognition research without one of its key resources. Synthetic face datasets have emerged as a promising alternative to privacy-sensitive authentic data for face recognition development. However, recent synthetic datasets that are used to train face recognition models suffer either from limitations in intra-class diversity or cross-class (identity) discrimination, leading to less optimal verification accuracies, far away from the accuracies achieved by models trained on authentic data. This paper targets this issue by proposing IDiff-Face, a novel approach based on conditional latent diffusion models for synthetic identity generation with realistic identity variations for face recognition training. Through extensive evaluations, our proposed synthetic-based face recognition approach pushed the limits of state-of-the-art performances, achieving, for example, 98.00% accuracy on the LFW benchmark, far ahead from the recent synthetic-based face recognition solutions with 95.40% and closing the gap to authentic-based face recognition with 99.82% accuracy.

## TODO: IDiff-Face models

Download links for the pre-generated 10K identities x 50 images datasets from the paper:
- **IDiff-Face Synthetic Uniform (25% CPD)**
- **IDiff-Face Synthetic Two-Stage (50% CPD)**

Download links for the pre-trained IDiff-Face diffusion model weights:
- **IDiff-Face Synthetic Uniform (25% CPD)**
- **IDiff-Face Synthetic Two-Stage (50% CPD)**

Download links for the pre-trained autoencoder weights that originally come from the `fhq256` LDM from [Rombach et al.](https://github.com/CompVis/latent-diffusion/blob/main/models/ldm/ffhq256/config.yaml). Specifically, the VQModelInterface submodule is manually extracted and split into its encoder and decoder models, since the encoder is only used during training and the decoder is only needed for sampling. The resulting .pt files are then expected to be saved under `models/autoencoder/first_stage_encoder_state_dict.pt` and `models/autoencoder/first_stage_decoder_state_dict.pt`, respectively.

## Results

The following table shows the verification benchmark performances achieved by methods from related work in comparison to the ones of our proposed IDiff-Face approach. They have been reported by the respective authors themselves. For face recognition training on our synthetic (uniform) data, we used a ResNet-50 with an ElasticFace loss, which is exactly the same setup that has been used in the SFace and USynthFace works. More detailed results are presented in the paper.

<table>
<tr>
<td colspan=2 align=center><b>Method</b>  <td colspan=3 align=center><b>Dataset</b>  <td colspan=6 align=center> <b> Verification Benchmarks</b>
<tr>
<td align=center>name <td align=center>year <td align=center>identities <td align=center>N per ID <td align=center>images<td align=center>LFW <td align=center>Age-DB-30 <td align=center>CA-LFW <td align=center>CFP-FP <td align=center>CP-LFW <td align=center>Average
<tr>
<td align=center>SynFace <td align=center> 2021 <td align=center> 10,000 <td align=center> 50 <td align=center>500K<td align=center>91.93 <td align=center> 61.63 <td align=center> 74.73 <td align=center> 75.03 <td align=center> 70.43 <td align=center> 74.75
    
<tr>
<td align=center>SFace <td align=center> 2021 <td align=center> 10,575 <td align=center> 60 <td align=center>634K<td align=center>91.87 <td align=center> 71.68 <td align=center> 77.93 <td align=center> 73.86 <td align=center> 73.20 <td align=center> 77.71
    
<tr>
<td align=center>USynthFace <td align=center> 2022 <td align=center> 400,000 <td align=center> 1 <td align=center>400K<td align=center>92.23 <td align=center> 71.62 <td align=center>77.05 <td align=center> 78.56 <td align=center> 72.03 <td align=center> 79.30
    
<tr>
<td align=center>DigiFace-1M <td align=center> 2023 <td align=center> 10,000 <td align=center> 50 <td align=center>500K<td align=center>95.40 <td align=center> 76.97 <td align=center>78.62 <td align=center> <b>87.40</b> <td align=center> 78.87 <td align=center> 83.45

<tr>
<td align=center>IDiff-Face (Ours) <td align=center> 2023 <td align=center> 5,000 <td align=center> 16 <td align=center>80K<td align=center>94.37 <td align=center> 76.70 <td align=center>85.02 <td align=center> 70.31 <td align=center> 72.18 <td align=center> 79.72
<tr>
<td align=center colspan=2> <td align=center> 5,000 <td align=center> 32 <td align=center>160K<td align=center>96.23 <td align=center> 80.50 <td align=center>88.30 <td align=center> 73.87 <td align=center> 75.42 <td align=center> 82.86
<tr>
<td align=center colspan=2> <td align=center> 10,000 <td align=center> 16 <td align=center>160K<td align=center>96.72 <td align=center> 81.52 <td align=center>88.83 <td align=center> 75.43 <td align=center> 76.85 <td align=center> 83.87
<tr>
<td align=center colspan=2> <td align=center> 10,000 <td align=center> 50 <td align=center>500K<td align=center>97.68 <td align=center> 84.63 <td align=center>90.58 <td align=center> 82.39 <td align=center> 79.70 <td align=center> 87.00
<tr>
<td align=center colspan=2>+ RandAugment(4, 16) <td align=center> 10,000 <td align=center> 50 <td align=center>500K<td align=center><b>98.00</b> <td align=center> <b>86.43</b> <td align=center><b>90.65</b> <td align=center> 85.47 <td align=center> <b>80.45</b> <td align=center> <b>88.20</b>
</table>

This fair comparison of the existing synthetic-based face recognition methods clearly demonstrates that IDiff-Face outperforms the GAN-based ones by a large margin and the rendering-based DigiFace-1M as well, except for one benchmark (the difficult CFP-FP).

---
## TODO: Examples
### Authentic ID to Image

<img align="center" src="etc/authentic_variations.png" width=90%>

### Synthetic ID to Image

<img align="center" src="etc/synthetic_variations.png" width=90%>

---

## How to use the code?

This is a reduced anonymized version of the implementation used for the main submission. It includes the main scripts used for training and evaluating the IDiff-Face models. All experiments of this project were conducted within a Docker container, whose `Dockerfile` is included in this archive. However, the scripts are itself not depending on Docker and thus the commands provided in this README to run the scripts are kept basic and thus might have to be slightly altered to match the specific environment of the user. You also might have to alter the root paths under `configs/paths/gpuc_cluster.yaml`.

### Setup
Download the FFHQ dataset (128x128) and put the `70.000` unlabelled images under `data/ffhq_128/`. The training embeddings used as contexts during training are NOT provided under `data/embeddings_elasticface_128.npy` due to the size limitation. They have to be extracted using the `extract_face_embeddings_from_dir.py` script. For that, the pre-trained ElasticFace-Arc model weights have to be downloaded from the [official ElasticFace repository](https://github.com/fdbtrs/ElasticFace) and placed under `utils/Elastic_R100_295672backbone.pth`. The [pre-trained autoencoder](https://ommer-lab.com/files/latent-diffusion/ffhq.zip) for the latent diffusion training is obtained from the pre-trained `fhq256` LDM from [Rombach et al.](https://github.com/CompVis/latent-diffusion/blob/main/models/ldm/ffhq256/config.yaml). Specifically, the VQModelInterface submodule is manually extracted and split into its encoder and decoder models, since the encoder is only used during training and the decoder is only needed for sampling. The resulting .pt files are then expected to be saved under `models/autoencoder/first_stage_encoder_state_dict.pt` and `models/autoencoder/first_stage_decoder_state_dict.pt`, respectively.

---
### Training a new model
In order to train the model with 25% CPD make sure that the option `model: unet_cond_ca_cpd25` is set in the `configs/train_config.yaml`. The CPD probability can easily be changed by creating a new model specification in the `configs/model/` subconfiguration folder. In addition to that, it has to be ensured that the `dataset: ffhq_folder` option is set and that the paths in the corresponding subconfiguration `configs/dataset/ffhq_folder.yaml` are pointing to the training images and pre-extracted embeddings. The model training can be initiated by executing:
    
    python main.py
     

**Naming the trained model**: After the model is trained, the model output directory content under `outputs/DATE/TIME/` can be copied to another folder e.g. `trained_models/unet-cond-ca-bs512-150K-cpd25/`. The name of this new folder is now referred to as the MODEL_NAME of the trained model.

---
### Sampling with a (pre-trained) model
For reproducibility and consistency, the synthetic contexts are NOT generated on-the-fly during sampling. Instead, they are pre-generated and saved in `.npy` files, which contain Python `dicts` with identity_names/dummy_names as keys and the associated context vector as value. This is the same structure used for the training embeddings. In this archive, some pre-generated `two-stage` contexts are already included. In order to generate samples with `synthetic_uniform` contexts, quickly execute the `create_sample_identity_contexts.py` script, which will pre-compute 15.000 synthetic uniform contexts that you can use for sampling. Then, specify the path to the trained model and the contexts file that shall be used for sampling in the `sample_config.yaml`. There you can also configure the number of identities to use from the provided contexts file and the number of images per identity context. Those samples will be saved under `samples/MODEL_NAME/CONTEXT_NAME` as identity blocks, e.g. a 4x4 grid block of 128x128 images (total block size is then 512x512). These blocks can then be splitted using e.g. then `split_identity_blocks.py` script. But before doing that, they have to be aligned. The sampling script can be started via:

    
    python create_sample_identity_contexts.py
    python sample.py
     
    
**Aligning the samples**: Aligning the images using MTCNN detection and ArcFace alignment is simply done by executing the `align.py` script after having specified every data that shall be aligned in the `align_config.yaml`. Currently, when alignment for one image per identity fails, the entire identity block is instead just resized to 112x112 instead of proper alignment. This option can be disabled by setting `just_resize_if_fail: False` in the config. Then, the entire block will be discarded instead. For the generation of 10.000 identities with 50 samples each, 10.050 identities were initially sampled from 15.000 pre-generated contexts to account for future alignment failures and thereby make sure that at least 10.000 identities with 50 aligned images are available for the large-scale training.

    
    python align.py
    
    
**Splitting the blocks**: Just execute the `split_identity_blocks.py` script after ensuring that the paths are correct. The script is very straightforward and easy to modify if any issues should occur.
    
    python split_identity_blocks.py
    

---
## Training a Face Recognition (FR) model
The dependencies for training the FR model are different. We used the training setup of [USynthFace](https://github.com/fdbtrs/Unsupervised-Face-Recognition-using-Unlabeled-Synthetic-Data) and are thus referring to that for the dependencies. With the code provided under `face_recognition_training/`, the training of a CosFace FR model with the configuration file under `face_recognition_training/config/config.py` that can be changed should be started via:

    
    ./training_large_scale_with_augment.sh
    

---
## Citing IDiff-Face

If you use IDiff-Face in a scientific publication, we would appreciate you using the following citations:

```
@article{Boutros2023IDiffFace,
    doi       = {TODO},
    url       = {TODO},
    author    = {Fadi Boutros and Jonas Henry Grebe and Naser Damer and Arjan Kuijper},
    title     = {IDiff-Face: Synthetic-based Face Recognition through Fizzy Identity-conditioned Diffusion Models},
booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {TODO}
}

@article{grebe2023identity,
  title={Identity Synthesis via Latent Diffusion Models for Accurate Face Recognition},
  author={Grebe, Jonas Henry},
  year={2023}
}
```

## Most important references

```
@inproceedings{DBLP:conf/cvpr/RombachBLEO22,
  author       = {Robin Rombach and
                  Andreas Blattmann and
                  Dominik Lorenz and
                  Patrick Esser and
                  Bj{\"{o}}rn Ommer},
  title        = {High-Resolution Image Synthesis with Latent Diffusion Models},
  booktitle    = {{IEEE/CVF} Conference on Computer Vision and Pattern Recognition,
                  {CVPR} 2022, New Orleans, LA, USA, June 18-24, 2022},
  pages        = {10674--10685},
  publisher    = {{IEEE}},
  year         = {2022},
  url          = {https://doi.org/10.1109/CVPR52688.2022.01042},
  doi          = {10.1109/CVPR52688.2022.01042},
  timestamp    = {Wed, 05 Oct 2022 16:31:19 +0200},
  biburl       = {https://dblp.org/rec/conf/cvpr/RombachBLEO22.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

@inproceedings{DBLP:conf/cvpr/KarrasLA19,
  author       = {Tero Karras and
                  Samuli Laine and
                  Timo Aila},
  title        = {A Style-Based Generator Architecture for Generative Adversarial Networks},
  booktitle    = {{IEEE} Conference on Computer Vision and Pattern Recognition, {CVPR}
                  2019, Long Beach, CA, USA, June 16-20, 2019},
  pages        = {4401--4410},
  publisher    = {Computer Vision Foundation / {IEEE}},
  year         = {2019},
  url          = {http://openaccess.thecvf.com/content\_CVPR\_2019/html/Karras\_A\_Style-Based\_Generator\_Architecture\_for\_Generative\_Adversarial\_Networks\_CVPR\_2019\_paper.html},
  doi          = {10.1109/CVPR.2019.00453},
  timestamp    = {Mon, 30 Aug 2021 17:01:14 +0200},
  biburl       = {https://dblp.org/rec/conf/cvpr/KarrasLA19.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

```

## More information on remaining folders and scripts:
### Directories:
- `configs/` contains the configuration .yaml files
- `data/` contains the training images and training embeddings
- `iffusion/` contains the DDPM code
- `models/` contains the PyTorch modules and model structures
- `samples/` will contain the generated (aligned) samples, their extracted features and the contexts used for sampling
- `evaluation/` will contain the computed comparison scores
- `trained_models/` contains pre-trained models
- `utils/` contains utility modules, models and scripts
- `face_recognition_training/` contains code that was used to train face recognition models

### Main scripts:
- `main.py` contains the training script
- `sample.py` contains the sampling script
- `align.py` contains the alignment script for generated samples
- `encode.py` contains the feature extraction script for (aligned) generated samples
- `evaluate.py` contains the evaluation script for the main experiments
- `evaluate_lfw.py` contains the evaluation script for the LFW-based experiment
- `create_sample_identity_contexts.py` contains code for identity-context generation
- `split_identity_blocks.py`samples are saved as concatenated blocks per identity (can easily be modified),
              and this script can be used to split them to create identity-class folders for FR training
