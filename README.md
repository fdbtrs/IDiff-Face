# README for the Official IDiff-Face Codebase

This is a reduced anonymized version of the implementation used for the main submission. It includes the main scripts used for training and evaluating the IDiff-Face models.

Due to the size limit of the supplementary material, no pre-trained model checkpoints or generated synthetic datasets are included in this archive. However, they will be made publicly available together with the paper.

All experiments of this submission were conducted within a Docker container, whose `Dockerfile` is included in this archive. However, the scripts are itself not depending on Docker and thus the commands provided in this README to run the scripts are kept basic and thus might have to be slightly altered to match the specific environment of the user. You also might have to alter the root paths under `configs/paths/gpuc_cluster.yaml`.

##### How to reproduce the main results from the paper?
1. **Preparation of Training**: Download the FFHQ dataset (128x128) and put the `70.000` unlabelled images under `data/ffhq_128/`. The training embeddings used as contexts during training are NOT provided under `data/embeddings_elasticface_128.npy` due to the size limitation. They have to be extracted using the `extract_face_embeddings_from_dir.py` script. For that, the pre-trained ElasticFace-Arc model weights have to be downloaded from the offical repository: https://github.com/fdbtrs/ElasticFace and placed under `utils/Elastic_R100_295672backbone.pth`. The pre-trained autoencoder for the latent diffusion training is obtained from the pre-trained `fhq256` LDM from Rombach et al.: https://github.com/CompVis/latent-diffusion/blob/main/models/ldm/ffhq256/config.yaml (https://ommer-lab.com/files/latent-diffusion/ffhq.zip). Specifically, the VQModelInterface submodule is extracted and split into its encoder and decoder models, since the encoder is only used during training and the decoder is only needed for sampling. The resulting .pt files are then expected to be saved under `models/autoencoder/first_stage_encoder_state_dict.pt` and `models/autoencoder/first_stage_decoder_state_dict.pt`, respectively.
2. **Training the IDiff-Face model**: In order to train the model with 25% CPD make sure that the option `model: unet_cond_ca_cpd25` is set in the `configs/train_config.yaml`. The CPD probability can easily be changed by creating a new model specification in the `configs/model/` subconfiguration folder. In addition to that, it has to be ensured that the `dataset: ffhq_folder` option is set and that the paths in the corresponding subconfiguration `configs/dataset/ffhq_folder.yaml` are pointing to the training images and pre-extracted embeddings. The model training can be initiated by executing:
    ```
    python main.py
    ``` 

3. **Naming the trained model**: After the model is trained, the model output directory content under `outputs/DATE/TIME/` can be copied to another folder e.g. `trained_models/unet-cond-ca-bs512-150K-cpd25/`. The name of this new folder is now referred to as the MODEL_NAME of the trained model.
4. **Sampling with the trained model**: For reproducability and consistency, the synthetic contexts are NOT generated on-the-fly during sampling. Instead, they are pre-generated and saved in `.npy` files, which contain Python `dicts` with identity_names/dummy_names as keys and the associated context vector as value. This is the same structure used for the training embeddings. In this archive, some pre-generated `two-stage` contexts are already included. In order to generate samples with `synthetic_uniform` contexts, quickly execute the `create_sample_identity_contexts.py` script, which will pre-compute 15.000 synthetic uniform contexts that you can use for sampling. Then, specify the path to the trained model and the contexts file that shall be used for sampling in the `sample_config.yaml`. There you can also configure the number of identities to use from the provided contexts file and the number of images per identity context. Those samples will be saved under `samples/MODEL_NAME/CONTEXT_NAME` as identity blocks, e.g. a 4x4 grid block of 128x128 images (total block size is then 512x512). These blocks can then be splitted using e.g. then `split_identity_blocks.py` script. But before doing that, they have to be aligned. The sampling script can be started via:
    ```
    python create_sample_identity_contexts.py
    python sample.py
    ``` 
6. **Aligning the samples**: Aligning the images using MTCNN detection and ArcFace alignment is simply done by executing the `align.py` script after having specified every data that shall be aligned in the `align_config.yaml`. Currently, when alignment for one image per identity fails, the entire identity block is instead just resized to 112x112 instead of proper alignment. This option can be disabled by setting `just_resize_if_fail: False` in the config. Then, the entire block will be discarded instead. For the generation of 10.000 identities with 50 samples each, 10.050 identities were initially sampled from 15.000 pre-generated contexts to account for future alignment failures and thereby make sure that at least 10.000 identities with 50 aligned images are available for the large-scale training.
    ```
    python align.py
    ``` 
7. **Splitting the blocks**: Just execute the `split_identity_blocks.py` script after ensuring that the paths are correct. The script is very straightforward and easy to modify if any issues should occur.
    ```
    python split_identity_blocks.py
    ``` 
8. **Training FR model**: The dependencies for training the FR model are different. We used the training setup of USynthFace (https://github.com/fdbtrs/Unsupervised-Face-Recognition-using-Unlabeled-Synthetic-Data) and are thus referring to that for the dependencies. With the code provided under `face_recognition_training/`, the training of a CosFace FR model with the configuration file under `face_recognition_training/config/config.py` that can be changed should be started via:
    ```
    ./training_large_scale_with_augment.sh
    ``` 
    
##### More information on remaining folders and scripts:
###### Directories:
- `configs/` contains the configuration .yaml files
- `data/` contains the training images and training embeddings
- `iffusion/` contains the DDPM code
- `models/` contains the PyTorch modules and model structures
- `samples/` will contain the generated (aligned) samples, their extracted features and the contexts used for sampling
- `evaluation/` will contain the computed comparison scores
- `trained_models/` contains pre-trained models
- `utils/` contains utility modules, models and scripts
- `face_recognition_training/` contains code that was used to train face recognition models

###### Main scripts:
- `main.py` contains the training script
- `sample.py` contains the sampling script
- `align.py` contains the alignment script for generated samples
- `encode.py` contains the feature extraction script for (aligned) generated samples
- `evaluate.py` contains the evaluation script for the main experiments
- `evaluate_lfw.py` contains the evaluation script for the LFW-based experiment
- `create_sample_identity_contexts.py` contains code for identity-context generation
- `split_identity_blocks.py`samples are saved as concatenated blocks per identity (can easily be modified),
              and this script can be used to split them to create identity-class folders for FR training