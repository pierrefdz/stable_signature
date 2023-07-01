# ✍️ The Stable Signature: Rooting Watermarks in Latent Diffusion Models

Implementation and pretrained models.
For details, see [**the paper**](https://arxiv.org/abs/2303.15435).  

[[`Webpage`](https://pierrefdz.github.io/publications/stablesignature/)]
[[`arXiv`](https://arxiv.org/abs/2303.15435)]


## Setup


### Requirements

First, clone the repository locally and move inside the folder:
```cmd
git clone https://github.com/pierrefdz/stable_signature
cd stable_signature
```
To install the main dependencies, we recommand using conda, and install the remaining dependencies with pip:
[PyTorch](https://pytorch.org/) can be installed with:
```cmd
conda install -c pytorch torchvision pytorch==1.12.0 cudatoolkit=11.3
pip install -r requirements.txt
```
This codebase has been developed with python version 3.8, PyTorch version 1.12.0, CUDA 11.3.


### Models and data

#### Data

The paper uses a filtered [COCO](https://cocodataset.org/) dataset to fine-tune the LDM decoder.
All you need is around 500 images for training (preferably over 256x256).

#### Watermark models

The watermark extractor model can be downloaded in the following links.
The `.pth` file has not been whitened, while the `.torchscript.pt` file has been and can be used without any further processing.

| Model | Checkpoint | Torch-Script |
| --- | --- | --- |
| Extractor | [dec_48b.pt](https://dl.fbaipublicfiles.com/ssl_watermarking/dec_48b.pth) | [dec_48b_whit.torchscript.pt](https://dl.fbaipublicfiles.com/ssl_watermarking/dec_48b_whit.torchscript.pt)  |

Code to train the watermark models will be made available.
Current models come from https://github.com/facebookresearch/ssl_watermarking.

#### Stable Diffusion models

Create LDM configs and checkpoints from the [Hugging Face](https://huggingface.co/stabilityai) and [Stable Diffusion](https://github.com/Stability-AI/stablediffusion/tree/main/configs/stable-diffusion) repositories.
The code should also work for Stable Diffusion v1 without any change. 
For other models (like old LDMs or VQGANs), you may need to adapt the code to load the checkpoints.


## Usage

### Fine-tune LDM decoder

```
python finetune_ldm_decoder.py --num_keys 1
    --ldm_config path/to/ldm/config.yaml
    --ldm_ckpt path/to/ldm/ckpt.pth
    --msg_decoder_path path/to/msg/decoder/ckpt.torchscript.pt
    --train_dir path/to/train/dir
    --val_dir path/to/val/dir
```

This code should generate: 
- *num_keys* checkpoints of the LDM decoder with watermark fine-tuning (checkpoint_000.pth, etc.),
- `keys.txt`: text file containing the keys used for fine-tuning (one key per line),
- `imgs`: folder containing examples of auto-encoded images.



### Generate

Reload weights of the LDM decoder in the Stable Diffusion scripts by appending the following lines after loading the checkpoint 
(for instance, [L220 in the SD repo](https://github.com/Stability-AI/stablediffusion/blob/main/scripts/txt2img.py#L220))
```python
state_dict = torch.load(path/to/ldm/checkpoint_000.pth)
msg = model.first_stage_model.load_state_dict(state_dict, strict=False)
```


### Decode

The `decode.ipynb` notebook contains a full example of the decoding and associated statistical test.

## Acknowledgements

This code is based on the following repositories:

- https://github.com/Stability-AI/stablediffusion
- https://github.com/SteffenCzolbe/PerceptualSimilarity

To train the watermark encoder/extractor, you can refer to the following repository https://github.com/ando-khachatryan/HiDDeN. 
Changes were made from this codebase and will be made available soon.

## License

The majority of Stable Signature is licensed under CC-BY-NC, however portions of the project are available under separate license terms: `src/ldm` and `src/taming` are licensed under the MIT license.

## Citation

If you find this repository useful, please consider giving a star :star: and please cite as:


```
@article{fernandez2023stable,
  title={The Stable Signature: Rooting Watermarks in Latent Diffusion Models},
  author={Fernandez, Pierre and Couairon, Guillaume and J{\'e}gou, Herv{\'e} and Douze, Matthijs and Furon, Teddy},
  journal={arXiv preprint arXiv:2303.15435},
  year={2023}
}
```