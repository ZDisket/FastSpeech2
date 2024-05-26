import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num, load_pretrained_weights
from utils.tools import to_device, log, synth_one_sample, test_one_fs2, log_attention_maps
from model import FastSpeech3Loss, PatchDiscriminator, SeqDiscriminator
from dataset import Dataset
from torch.cuda.amp import GradScaler, autocast

from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    model, optimizer = get_model(args, configs, device, train=True)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=train_config["optimizer"]["gamma"],
                                                       last_epoch=last_epoch)

    if len(args.pretrained):
        load_pretrained_weights(model, args.pretrained)

    discriminator = SeqDiscriminator().to(device)
    discriminator.train()
    criterion_d = nn.BCEWithLogitsLoss()
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001)

    model = nn.DataParallel(model)
    discriminator = nn.DataParallel(discriminator)
    num_param = get_param_num(model)
    Loss = FastSpeech3Loss(preprocess_config, model_config, train_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

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

    # torch.autograd.set_detect_anomaly(True, True)

    scaler = GradScaler()
    scaler_d = GradScaler()

    discriminator_train_start_steps = 2500

    while True:
        inner_bar = tqdm(total=len(loader), desc=f"Epoch {epoch}", position=1)
        for batchs in loader:
            for batch in batchs:

                with autocast(enabled=True):
                    batch = to_device(batch, device)

                    # Forward pass and loss computation with autocast
                    output = model(*(batch[2:]))

                    # =========================== DISCRIMINATOR ==================================

                    durations_real = output[5].detach() # we don't want to optimize the AlignmentEncoder
                    durations_fake = output[4]
                    seq_lens = batch[2 + 2]

                    # the attn_hard_dur is in the linear space, bring it to the log one
                    durations_real = torch.log(durations_real.float() + 1)

                    if step > discriminator_train_start_steps:
                        # train on real
                        outputs_real = discriminator(durations_real, seq_lens)
                        real_labels = torch.ones(outputs_real.size()).to(device)

                        loss_real = criterion_d(outputs_real, real_labels)

                        # train on fake
                        outputs_fake = discriminator(durations_fake.detach(), seq_lens)
                        fake_labels = torch.zeros(outputs_fake.size()).to(device)

                        loss_fake = criterion_d(outputs_fake, fake_labels)

                        loss_d = (loss_real + loss_fake) / 2

                        scaler_d.scale(loss_d).backward()
                        scaler_d.unscale_(optimizer_d)
                        nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip_thresh)

                        scaler_d.step(optimizer_d)
                        scaler_d.update()
                        optimizer_d.zero_grad()
                    else:
                        loss_d = torch.FloatTensor([0.0]).to(device)


                    # =========================== END DISCRIMINATOR ==================================

                    losses = Loss(batch, output, epoch, model.module)

                    if step > discriminator_train_start_steps:
                        outputs_fake = discriminator(durations_fake, seq_lens)
                        gan_loss = criterion_d(outputs_fake, real_labels)
                    else:
                        gan_loss = torch.FloatTensor([0.0]).to(device)

                    losses.append(loss_d)
                    losses.append(gan_loss)
                    losses[0] += gan_loss

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
                    message2 = ("Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f},"
                                " Attention Loss: {:.4f}, Duration Temporal Loss: {:.4f}, Total Temporal Loss: {:.4f},"
                                "Duration KL Divergence Loss: {:.4f}, Pitch-Energy KL Loss: {:.4f}, Dur Discriminator Loss: {:.4f}"
                                ", Duration GAN Loss: {:.4f}"
                                ).format(
                                    *losses
                                )

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)

                    log(train_logger, step, losses=losses)

                if step % synth_step == 0:
                    fig, wav_reconstruction, wav_prediction, tag, attn_soft = synth_one_sample(
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
                    # Assuming `attention_tensor` is your tensor of attention maps shaped (batch, w, h)
                    # and `train_logger` is your initialized SummaryWriter
                    log_attention_maps(train_logger, attn_soft.transpose(1, 2).detach(),
                                       output[9].detach().cpu().numpy(), output[8].detach().cpu().numpy(),
                                       step, tag_prefix="Training")

                if step % val_step == 0:
                    model.eval()
                    message = evaluate(model, step, configs, val_logger, vocoder)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    test_sentences = [
                        "The quick brown fox jumps over the lazy dog, while the sun sets over the peaceful valley",
                        "When I visited Rome, the capital of Italy, I saw the Colosseum, the Vatican, and St. Peter's Basilica",
                        "Even though John loves chocolate, strawberries, and ice cream, he decided to try the vanilla cake instead",
                        "Peter Piper picked a peck of pickled peppers, how many pickled peppers did Peter Piper pick?",
                        "Now I see. Black human beings dislike the sound of rubbing glass probably the sound wave of the whistle"]

                    for i, sent in enumerate(test_sentences):
                        t_aud = test_one_fs2(model.module, vocoder, sent)
                        if t_aud is None:
                            continue
                        log(
                            val_logger,
                            step=step,
                            audio=t_aud,
                            sampling_rate=preprocess_config["preprocessing"]["audio"]["sampling_rate"],
                            tag=f"Test/sentence_{i}"
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
