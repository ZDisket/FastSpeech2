import os
import json
from collections import OrderedDict

import torch
import numpy as np

import hifigan
from model import FastSpeech2, ScheduledOptim, AdEMAMix, Sturmschlag
from istftnetfe import ISTFTNetFE
from model.submodels import PreEncoder


def load_pretrained_weights(model, pretrained_path):
    print(f"Loading pretrained weights from {pretrained_path}")
    ckpt = torch.load(pretrained_path)
    pretrained_dict = ckpt["model"]

    model_dict = model.state_dict()
    mismatched_shapes = []

    for name, param in pretrained_dict.items():
        if name in model_dict:
            if model_dict[name].shape != param.shape:
                mismatched_shapes.append((name, model_dict[name].shape, param.shape))
            else:
                model_dict[name] = param

    # Print mismatched shapes
    if mismatched_shapes:
        print("Mismatched shapes found:")
        for name, model_shape, pretrained_shape in mismatched_shapes:
            print(f"{name}: model shape {model_shape}, pretrained shape {pretrained_shape}")

        print("This is not an error, if you know what you're doing.")

    # Load the updated state dict with matching shapes
    model.load_state_dict(model_dict, strict=False)


def get_pre_encoder(model_path: str, device: str or torch.device):
    """
    Loads a Pre-Encoder model from a checkpoint file.

    Assumes the checkpoint was saved with the training script's structure,
    containing 'model_state_dict' and 'args' (or a compatible dict).

    Args:
        model_path (str): Path to the .pth checkpoint file.
        device (str or torch.device): The device to load the model onto ('cpu', 'cuda', etc.).

    Returns:
        tuple: A tuple containing:
            - model (nn.Module): The loaded ResNetAutoencoder1D model instance,
                                 moved to the specified device and set to eval mode.
            - model_args (argparse.Namespace or dict): The configuration arguments
                                                       used to initialize the model,
                                                       loaded from the checkpoint.
    Raises:
        FileNotFoundError: If the model_path does not exist.
        KeyError: If essential keys ('args', 'model_state_dict') are missing
                  from the checkpoint.
        RuntimeError: If load_state_dict fails (e.g., architecture mismatch).
        ImportError: If the ResNetAutoencoder1D class cannot be imported/found.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint file not found: {model_path}")

    print(f"Loading checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False) # Load to CPU first

    # --- 2. Instantiate Model ---
    try:
        model = PreEncoder(mel_channels=88, channels=[512, 512, 384], kernel_sizes=[11, 5, 3],
                             dropout=0.1)
    except NameError:
         raise ImportError("ResNetAutoencoder1D class definition not found. Ensure model.py is accessible or the class is defined.")
    except Exception as e:
         raise RuntimeError(f"Failed to instantiate model with loaded config: {e}")


    # --- 3. Load Weights ---
    if 'model_state_dict' in checkpoint:
        pretrained_weights = checkpoint['model_state_dict']
        print("Found weights under 'model_state_dict' key.")

        # Optional: Handle 'module.' prefix (if saved using DataParallel/DDP)
        clean_weights = OrderedDict()
        has_module_prefix = False
        for k, v in pretrained_weights.items():
            if k.startswith('module.'):
                has_module_prefix = True
                clean_weights[k[7:]] = v # remove `module.`
            else:
                clean_weights[k] = v
        if has_module_prefix:
            print("Removed 'module.' prefix from weight keys.")
        pretrained_weights = clean_weights # Use the cleaned dictionary

        # Load the weights using strict=True (assumes exact match)
        try:
            model.load_state_dict(pretrained_weights, strict=True)
            print("Successfully loaded model weights.")
        except RuntimeError as e:
            print(f"Error loading state_dict (likely architecture mismatch): {e}")
            raise e # Re-raise the error

    else:
        raise KeyError(f"Checkpoint missing 'model_state_dict' key containing weights.")

    # --- 4. Final Steps ---
    model.to(device) # Move model to the target device
    model.eval()     # Set model to evaluation mode
    print(f"Model loaded onto {device} and set to evaluation mode.")

    return model

def get_model(args, configs, device, train=False, model="fs", opt="adam"):
    (preprocess_config, model_config, train_config) = configs

    if model == "fs":
        model = FastSpeech2(preprocess_config, model_config).to(device)
    elif model == "st":
        model = Sturmschlag(preprocess_config, model_config).to(device)

    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        if opt == "adam":
            scheduled_optim = torch.optim.Adam(model.parameters(),
                                                lr=train_config["optimizer"]["init_lr"],
                                                eps=train_config["optimizer"]["eps"],
                                                weight_decay=train_config["optimizer"]["weight_decay"],
                                                )
        elif opt == "adamw":
            scheduled_optim = torch.optim.AdamW(model.parameters(),
                                               lr=train_config["optimizer"]["init_lr"],
                                               eps=train_config["optimizer"]["eps"],
                                               weight_decay=train_config["optimizer"]["weight_decay"],
                                               )
            print("AdamW")

        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model



def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar")
        elif speaker == "universal":
            ckpt = torch.load("hifigan/generator_universal.pth.tar")
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)
    elif name == "iSTFTNet":
        vocoder = ISTFTNetFE(None, None)
        vocoder.load_ts("istftnet/universal","cuda")


    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)
        elif name == "iSTFTNet":
            wavs = vocoder(mels.float()).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
