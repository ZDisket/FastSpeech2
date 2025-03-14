import argparse
import os

import torch

# having this True is bad (both for CUDA and ROCm) when we use diff lens each batch
torch.backends.cudnn.benchmark = False
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import itertools
from utils.model import get_model, get_vocoder, get_param_num, load_pretrained_weights
from utils.tools import to_device, log, synth_one_sample_st, test_one_fs2, log_attention_maps
from model import SturmLoss
from model.loss import LSGANLoss
from dataset import Dataset
from torch.cuda.amp import GradScaler, autocast
from bertfe import BERTFrontEnd
from preprocessor.emotion import EmotionProcessorV2

from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weights_init_he(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

from torch.optim.lr_scheduler import LRScheduler

def supports_native_bf16(device_index=0):
    # Get the compute capability of the GPU.
    major, minor = torch.cuda.get_device_capability(device_index)
    # NVIDIA Ampere GPUs (and newer) have compute capability 8.0 or higher.
    return major >= 8


class WarmupExponentialLR(LRScheduler):
    def __init__(self, optimizer, gamma, warmup_steps, last_epoch=-1):
        self.gamma = gamma
        self.warmup_steps = warmup_steps
        super(WarmupExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            warmup_factor = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            return [base_lr * (self.gamma ** (self.last_epoch - self.warmup_steps)) for base_lr in self.base_lrs]



def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    step = args.restore_step + 1
    epoch = 1
    last_epoch = -1

    if step > 1:
        steps_per_epoch = len(dataset) // batch_size
        current_epoch = step // steps_per_epoch
        epoch = current_epoch
        last_epoch = epoch - 1

    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True, model="st")
    scheduler = WarmupExponentialLR(optimizer, gamma=train_config["optimizer"]["gamma"],
                                                       last_epoch=last_epoch,
                                    warmup_steps=5 if not len(args.pretrained) else 1)

    if len(args.pretrained):
        load_pretrained_weights(model, args.pretrained)

    encoder_param = get_param_num(model.encoder)
    decoder_param = get_param_num(model.decoder)
    total_param = get_param_num(model)

    print("Number of Encoder Parameters: {:.2f}M".format(encoder_param / 1e6))
    print("Number of Decoder Parameters: {:.2f}M".format(decoder_param / 1e6))
    print("Total Number of Sturmschlag Parameters: {:.2f}M".format(total_param / 1e6))

    model = nn.DataParallel(model)
    Loss = SturmLoss(8.0).to(device)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    if len(preprocess_config["preprocessing"]["zephyr_model"]):
        raise RuntimeError("Not supported")


    if len(preprocess_config["preprocessing"]["bert_model"]):
        bert_model = BERTFrontEnd(torch.cuda.is_available(), preprocess_config["preprocessing"]["bert_model"])

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0, miniters=1)
    outer_bar.n = args.restore_step
    outer_bar.update()

    if supports_native_bf16(device):
        print("GPU natively supports BF16. Using BF16...")
        mp_dtype = torch.bfloat16
    else:
        print("GPU does NOT natively support BF16. Using FP16 instead...")
        mp_dtype = torch.float16



    # torch.autograd.set_detect_anomaly(True, True)

    scaler = GradScaler()

    while True:
        inner_bar = tqdm(total=len(loader), desc=f"Epoch {epoch}", position=1)
        for batchs in loader:
            for batch in batchs:

                with autocast(enabled=True,dtype=mp_dtype):
                    # run to_device with reduced features
                    batch = to_device(batch, device, reduced=True)

                    # Forward pass and loss computation with autocast
                    output = model(*(batch[2:]))

                    losses = Loss(batch, output)


                    total_loss = losses[0] / grad_acc_step

                # Backward pass with scaled loss
                scaler.scale(total_loss).backward()

                if step % grad_acc_step == 0:
                    # Clipping gradients
                    scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Step with GradScaler
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    losses = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = (
                        "Total Loss: {:.4f}, Mel Loss: {:.4f}, Gate Loss: {:.4f}"
                        ).format(
                        *losses
                    )

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)

                    log(train_logger, step, losses=losses)

                if step % synth_step == 0:
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample_st(
                        batch,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )

                    attn_weights = model.module.decoder.dec.decoder_layers[-1].last_weights # (B, H, L1, L2)
                    attn_soft = attn_weights.mean(dim=1).detach().cpu()

                    log_attention_maps(train_logger, attn_soft.transpose(1,2),
                                       batch[6].detach().cpu().numpy(), batch[4].detach().cpu().numpy(),
                                       step, tag_prefix="Training")

                if step % val_step == 0:
                    model.eval()
                    message = evaluate(model, step, configs, val_logger, vocoder, epoch)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    speakers = train_config['test_speakers']
                    sentences = train_config['test_sentences']

                    # Generate all combinations of speakers and sentences
                    pairs = list(itertools.product(speakers, sentences))

                    for idx, (spkid, sent) in enumerate(pairs, start=1):
                        blocks, hid = bert_model.infer(sent)
                        t_aud = test_one_fs2(model.module, vocoder, sent, blocks.cpu().numpy(), hid.cpu().numpy(), int(spkid))
                        if t_aud is None:
                            continue
                        log(
                            val_logger,
                            step=step,
                            audio=t_aud,
                            sampling_rate=preprocess_config["preprocessing"]["audio"]["sampling_rate"],
                            tag=f"Test/sentence_{idx}"
                        )

                    model.train()

                if step % save_step == 0:
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        },
                        os.path.join(
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )


                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        scheduler.step()


        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, required=False, help="output dir override", default=""
    )
    parser.add_argument(
        "-pt", "--pretrained", type=str, required=False, help="Path to pretrained model to finetune from", default=""
    )

    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)

    if len(args.output_dir):
        train_config["path"]["ckpt_path"] = f"{args.output_dir}/ckpt"
        train_config["path"]["log_path"] = f"{args.output_dir}"
        train_config["path"]["result_path"] = f"{args.output_dir}/results"

    if args.restore_step > 1:
        args.pretrained = ""

    configs = (preprocess_config, model_config, train_config)

    main(args, configs)
