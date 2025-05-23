# train_autoencoder.py

import os
import argparse
import numpy as np
import torch

# having this True is bad (both for CUDA and ROCm) when we use diff lens each batch
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Pad
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import random
from model.submodels import PreEncoder
from utils.model import get_param_num
from model.discriminator import MelSpectrogramPatchDiscriminator, MultiBinDiscriminator, MelSpectrogramPatchDiscriminator2D
from model.loss import LSGANLoss, MaskedMelLoss, MaskedMAE
from model.melvecquant import MVQGenerator



def signed_log1p(mel: np.ndarray) -> np.ndarray:
    # mel shape: (T, F), all positive or >=0
    # if mels are always >= 0, you can just do log1p
    return np.sign(mel) * np.log1p(np.abs(mel))

# =============================================================================
# Dataset & DataLoader for Real Spectrograms (with optional random cropping)
# =============================================================================
class RealMelSpectrogramDataset(Dataset):
    def __init__(self, data_dir: str, crop_len = None):
        """
        Args
        ----
        data_dir  : root folder containing .npy files (recursively searched)
        crop_len  : number of time-frames to return.  If None, return full length
        """
        self.real_dir   = data_dir
        self.crop_len   = crop_len
        print(f"Crop len: {self.crop_len}")
        if not os.path.isdir(self.real_dir):
            raise FileNotFoundError(f"Directory not found: {self.real_dir}")

        # collect all .npy paths
        self.filenames = [
            os.path.join(root, fn)
            for root, _, files in os.walk(self.real_dir)
            for fn in files if fn.endswith(".npy")
        ]
        if not self.filenames:
            print(f"Warning: No .npy files found in {self.real_dir} (recursively).")
        else:
            print(f"Found {len(self.filenames)} .npy files.")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx):
        path = self.filenames[idx]
        try:
            mel = np.load(path)            # (T, C)
        except Exception as e:
            print(f"[Load error] {path}: {e}")
            mel = np.zeros((1, 160), dtype=np.float32)

        if mel.ndim != 2:
            print(f"[Shape error] {path}: shape={mel.shape}")
            mel = np.zeros((1, 160), dtype=np.float32)

        full_len = mel.shape[0]
        target   = self.crop_len

        # -------- enforce length --------
        if target is not None:
            if full_len > target:  # random crop
                start = np.random.randint(0, full_len - target + 1)
                mel   = mel[start : start + target]
            elif full_len < target:  # zero-pad at end
                pad = np.zeros((target - full_len, mel.shape[1]), dtype=mel.dtype)
                mel = np.concatenate([mel, pad], axis=0)

            # final hard clamp/trim (paranoia guard)
            mel = mel[: target]
            assert mel.shape[0] == target, f"Unexpected len {mel.shape[0]} vs {target}"

        mel_len = min(full_len, target) if target is not None else full_len

        mel = mel.astype(np.float32)
       # mel = signed_log1p(mel)

        return mel, int(mel_len), os.path.basename(path)



def pad_collate_fn(batch):
    # Filter out potential error items (if __getitem__ returns None on error)
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None, None  # Handle empty batch case

    # Unpack the batch of (spectrogram, mel_len, filename)
    real_spectrograms, mel_lens, filenames = zip(*batch)

    if len({m.shape[0] for m in real_spectrograms}) == 1:
        real_padded_stacked = torch.stack([torch.as_tensor(m, dtype=torch.float32)
                                       for m in real_spectrograms])
        mel_lens_tensor     = torch.tensor(mel_lens, dtype=torch.int32)
        return real_padded_stacked, mel_lens_tensor, filenames

    # Find max mel_len *after* filtering potential errors
    if not mel_lens:
        return None, None, None
    max_mel_len = max(mel_lens)

    # Pad each spectrogram along the time dimension (dim 0)
    real_padded = []
    valid_mel_lens = []
    valid_filenames = []
    for mel, length, fname in zip(real_spectrograms, mel_lens, filenames):
        # Double check dimensions before padding
        if mel.ndim == 2:
            pad_amount = max_mel_len - length
            # Pad format is (padding_left, padding_right, padding_top, padding_bottom)
            # We want to pad the time dimension (dim 0), so pad_top=0, pad_bottom=pad_amount
            # We don't pad the mel channels dim (dim 1), so pad_left=0, pad_right=0
            # For torchvision Pad, it's (left, top, right, bottom) - THIS IS FOR 2D IMAGES.
            # For tensors directly, use F.pad. Input: (Batch, Channel, H, W) or similar.
            # Our input is (Time, MelChannels). Need (Batch, Time, MelChannels) later.
            # Let's use torch.nn.functional.pad
            # pad takes (pad_left, pad_right, pad_top, pad_bottom, ...)
            # For (Time, Mel), we need to pad dim 0 (Time) at the end.
            # F.pad requires input tensor.
            mel_tensor = torch.tensor(mel, dtype=torch.float32)
            # Padding format for F.pad is (pad_dim_N_start, pad_dim_N_end, pad_dim_N-1_start, ...)
            # For tensor (Time, Mel), need padding (0, 0, 0, pad_amount) -> pads Mel dim, pads Time dim
            padded_mel = F.pad(mel_tensor, (0, 0, 0, pad_amount), mode='constant', value=0)
            real_padded.append(padded_mel)
            valid_mel_lens.append(length)
            valid_filenames.append(fname)
        else:
            print(f"Skipping item {fname} in collate_fn due to unexpected dimensions: {mel.shape}")

    if not real_padded:
        return None, None, None  # Handle case where all items in batch had errors

    real_padded_stacked = torch.stack(real_padded)
    mel_lens_tensor = torch.tensor(valid_mel_lens, dtype=torch.int32)

    return real_padded_stacked, mel_lens_tensor, valid_filenames


def get_dataloader(data_dir, batch_size=16, shuffle=True, num_workers=0):
    dataset = RealMelSpectrogramDataset(data_dir)
    if len(dataset) == 0:
        print("Error: Dataset is empty. Cannot create DataLoader.")
        return None
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, collate_fn=pad_collate_fn,
                            pin_memory=True if num_workers > 0 else False)  # Added pin_memory
    return dataloader


# =============================================================================
# Utility: Plot Spectrogram
# =============================================================================
def plot_mel_spectrogram(spectrogram, vmin, vmax, save_path, title='Mel Spectrogram', ylabel='Frequency',
                         xlabel='Time'):
    """
    Plot a mel spectrogram with a fixed color scale and save it to a file.
    """
    if isinstance(spectrogram, torch.Tensor):
        # Ensure tensor is on CPU and detached before converting to numpy
        spectrogram = spectrogram.float().cpu().detach().numpy()

    # Ensure it's 2D
    if spectrogram.ndim != 2:
        print(f"Error plotting: Spectrogram has unexpected shape {spectrogram.shape}. Expected 2D.")
        return

    # Transpose for visualization: (mel_channels, time) -> (time, mel_channels) seems standard in audio
    # The original code transposed (Batch?, Time, Mel) -> (Batch?, Mel, Time) for imshow
    # Let's stick to the original code's transpose logic: (Time, Mel) -> (Mel, Time)
    spectrogram_display = np.transpose(spectrogram, (1, 0))

    fig, ax = plt.subplots(figsize=(10, 4))  # Use fig, ax for better control
    im = ax.imshow(spectrogram_display, aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap='magma')  # Added cmap
    fig.colorbar(im, ax=ax, format='%+2.0f dB')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    plt.tight_layout()

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)  # Close the figure to free memory


def masked_mae(pred: torch.Tensor,
               target: torch.Tensor,
               mask: torch.Tensor,
               eps: float = 1e-8) -> torch.Tensor:
    """
    Mean absolute error only over valid (non‑padded) positions.

    Args:
        pred:   (B, C, L) or (B, 1, L) — model output or feature map
        target: same shape as pred
        mask:   (B, L) or (B, 1, L) bool tensor where True = padded
        eps:    small constant to avoid div/0
    Returns:
        scalar MAE over all valid elements
    """
    # ensure mask has shape (B,1,L)
    if mask.dim() == 2:
        mask = mask.unsqueeze(1)
    # broadcast mask to channel dim if necessary
    mask = mask.expand_as(pred)

    diff = torch.abs(pred - target)
    diff = diff.masked_fill(mask, 0.0)

    valid_cnt = mask.sum()
    return diff.sum() / (valid_cnt + eps)

from torch.optim.lr_scheduler import LambdaLR
# =============================================================================
# Training Loop for the Autoencoder
# =============================================================================
def train(model, train_dataloader, eval_dataset, criterion_recon, optimizer, scaler, device, args):
    """
    Trains the autoencoder model.

    Args:
        model: The autoencoder model instance.
        train_dataloader: DataLoader for the training set.
        eval_dataset: Dataset for evaluation (used for plotting random samples).
        criterion_recon: Reconstruction loss function.
        optimizer: Optimizer for the model.
        scaler: Gradient scaler for mixed precision.
        device: The device to train on (e.g., 'cuda' or 'cpu').
        args: Command line arguments containing num_epochs, output_dir, etc.
    """
    plot_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Plots will be saved to: {plot_dir}")

    model, discriminator, discriminator_mb = model # added discriminator_mb
    optimizer, optimizer_d = optimizer
    gan_loss = LSGANLoss()
    crit_recon = MaskedMelLoss("mse")
    crit_recon_g = MaskedMelLoss("mse", group_size=20)
    discriminator_train_start_epoch = 0
    fm_lambda = 0.25
    Gloss_lambda = 15.0
    recon_lambda = 15.0
    use_fm_loss = False
    lr_lambda = lambda step: min((step + 1) / args.warmup_steps, 1.0)
    scheduler = LambdaLR(optimizer, lr_lambda)
    original_crop_size = eval_dataset.dataset.crop_len
    print(f"Orig crop = {original_crop_size}")

    if train_dataloader is None:
        print("Training dataloader is None. Skipping training.")
        return

    n_steps = 0
    for epoch in range(args.num_epochs):
        model.train()
        discriminator.train()
        discriminator_mb.train()
        epoch_loss = 0.0
        num_batches = len(train_dataloader)
        loop = tqdm(train_dataloader, leave=True, desc=f"Epoch [{epoch + 1}/{args.num_epochs}]")

        for batch_idx, batch_data in enumerate(loop):
            n_steps += 1

            # Check if batch data is valid (from collate_fn)
            if batch_data[0] is None:
                print(f"Skipping empty batch {batch_idx + 1}/{num_batches}")
                continue

            real_spectrograms, mel_lens, filenames = batch_data
            real_spectrograms = real_spectrograms.to(device)  # Shape: (batch, max_mel_len, mel_channels)
            mel_lens = mel_lens.to(device)

            # Check if batch size is > 0 after potential errors in collate/getitem
            if real_spectrograms.size(0) == 0:
                print(f"Skipping batch {batch_idx + 1}/{num_batches} due to zero valid items.")
                continue

            optimizer.zero_grad()
            optimizer_d.zero_grad()

            # Enable autocast context manager for mixed precision
            with autocast(enabled=True):  # Use enabled=True for clarity, default is True if torch.cuda.is_available()

                if epoch > discriminator_train_start_epoch:
                    # 1) D on real
                    real_logits, real_mask, real_feats = discriminator(real_spectrograms, mel_lens, return_features=True)

                    # 2) MBD on real
                    real_logits_l2, real_mask_l2, real_feats_l2 = discriminator_mb(real_spectrograms, mel_lens, return_features=True)

                # Forward pass: reconstruct the input spectrograms
                recon_spectrograms = model(real_spectrograms, mel_lens)  # Expects (batch, seq_len, features)

                if epoch > discriminator_train_start_epoch:
                    # 2) D on fake (detach generator)
                    fake_logits, fake_mask = discriminator(recon_spectrograms.detach(), mel_lens)

                    # 3) MBD on fake (detach generator)
                    fake_logits_l2, fake_mask_l2 = discriminator_mb(recon_spectrograms.detach(), mel_lens)

                    loss_MBD = torch.tensor(0.0, device=device)

                    for i, r_logits in enumerate(real_logits_l2):
                        d_loss_mask_r = real_mask_l2[0]
                        d_loss_mask_f = fake_mask_l2[0]

                        f_logits = fake_logits_l2[i]
                        loss_MBD += gan_loss.discriminator_loss(r_logits, f_logits, d_loss_mask_r, d_loss_mask_f)

                    loss_MBD /= len(real_logits_l2)
                    loss_D1 = gan_loss.discriminator_loss(real_logits, fake_logits, real_mask, fake_mask)


                    loss_D = loss_D1 + loss_MBD
                    scaler.scale(loss_D).backward()
                    scaler.step(optimizer_d)
                    scaler.update()
                else:
                    loss_D = torch.tensor(0.0, device=device)

                # Ensure output shape matches input shape
                if recon_spectrograms.shape != real_spectrograms.shape:
                    print(f"Shape mismatch! Input: {real_spectrograms.shape}, Output: {recon_spectrograms.shape}")
                    # Decide how to handle: maybe skip batch, maybe raise error
                    continue  # Skip this batch

                # Compute reconstruction loss only on valid (non-padded) elements.
                loss_recon_all = crit_recon(recon_spectrograms, real_spectrograms, mel_lens)
                loss_recon_g = crit_recon_g(recon_spectrograms, real_spectrograms, mel_lens)

                loss_recon = loss_recon_all + loss_recon_g * 0.25

                if epoch > discriminator_train_start_epoch:
                    gen_logits, gen_mask, gen_feats = discriminator(recon_spectrograms, mel_lens, return_features=True)

                    gen_logits_l2, gen_mask_l2, gen_feats_l2 = discriminator_mb(recon_spectrograms, mel_lens, return_features=True)

                    loss_gan_mbd = torch.tensor(0.0, device=device)
                    loss_fm_mbd = torch.tensor(0.0, device=device)
                    loss_fm_d1 = torch.tensor(0.0, device=device)

                    for i, g_logits in enumerate(gen_logits_l2):
                        g_loss_mask = gen_mask_l2[0]
                        loss_gan_mbd += gan_loss.generator_loss(g_logits, g_loss_mask)

                        r_feats = real_feats_l2[i]
                        g_feats = gen_feats_l2[i]

                        if use_fm_loss:
                            for (rf, mask), (ff, _) in zip(r_feats, g_feats):
                                rf = rf.detach()  # ensure no backprop into D for real feats
                                loss_fm_mbd += masked_mae(ff, rf, mask)

                            loss_fm_mbd /= len(r_feats)


                    loss_gan_mbd /= len(gen_logits_l2)
                    loss_gan_d1 = gan_loss.generator_loss(gen_logits, gen_mask)
                    loss_gan = 0.5 * (loss_gan_d1 + loss_gan_mbd)

                    if use_fm_loss:
                        for (rf, mask), (ff, _) in zip(real_feats, gen_feats):
                            rf = rf.detach() # ensure no backprop into D for real feats
                            loss_fm_d1 += masked_mae(ff, rf, mask)

                        loss_fm_d1 /= len(real_feats)
                        loss_fm = 0.5 * (loss_fm_mbd + loss_fm_d1)
                    else:
                        loss_fm = torch.tensor(0.0, device=device)
                else:
                    loss_gan = torch.tensor(0.0, device=device)
                    loss_fm = loss_gan
                    loss_gan_mbd = loss_gan

                loss = loss_recon * recon_lambda + loss_gan * Gloss_lambda + loss_fm * fm_lambda

            # Backward pass and optimization step using GradScaler
            scaler.scale(loss).backward()
            # Unscales gradients and calls optimizer.step()
            scaler.step(optimizer)
            scheduler.step()
            # Updates the scale for next iteration
            scaler.update()

            current_loss = loss.item()
            epoch_loss += loss_recon.item()
            loop.set_postfix(D=loss_D.item(),
                             G_recon=loss_recon.item(),
                             G_gan=loss_gan.item(),
                             G_gan_mbd=loss_gan_mbd.item(),
                             G_fm=loss_fm.item(),
                             total_loss=loss.item(),)

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch [{epoch + 1}/{args.num_epochs}], Average Recon Loss: {avg_loss:.6f}")

        # --- Plotting training examples ---
        if real_spectrograms is not None and real_spectrograms.size(0) > 0 and recon_spectrograms is not None:
            # For consistent visualization, determine global vmin and vmax across the batch
            with torch.no_grad():
                vmin = min(real_spectrograms.min().item(), recon_spectrograms.min().item())
                vmax = max(real_spectrograms.max().item(), recon_spectrograms.max().item())

            num_plots = min(args.num_plot_examples, real_spectrograms.size(0))
            for i in range(num_plots):
                # Extract the actual length for plotting
                actual_len = mel_lens[i].item()
                # Slice the spectrograms to their original length before plotting
                original_spec = real_spectrograms[i, :actual_len, :]
                recon_spec = recon_spectrograms[i, :actual_len, :]
                original_filename = filenames[i] if i < len(filenames) else f"Unknown_{i}"

                plot_mel_spectrogram(
                    original_spec, vmin=vmin, vmax=vmax,
                    save_path=os.path.join(plot_dir,
                                           f'epoch_{epoch + 1:03d}_train_orig_{i + 1}_{os.path.splitext(original_filename)[0]}.png'),
                    title=f'Epoch {epoch + 1} - Original {i + 1} ({original_filename})'
                )
                plot_mel_spectrogram(
                    recon_spec, vmin=vmin, vmax=vmax,
                    save_path=os.path.join(plot_dir,
                                           f'epoch_{epoch + 1:03d}_train_recon_{i + 1}_{os.path.splitext(original_filename)[0]}.png'),
                    title=f'Epoch {epoch + 1} - Reconstructed {i + 1} ({original_filename})'
                )
        else:
            print(f"Skipping training plots for epoch {epoch + 1} due to missing data.")

        # --- Evaluation plotting ---
        if (epoch + 1) % args.eval_interval == 0 and eval_dataset is not None and len(eval_dataset) > 0:
            print(f"Running evaluation plots for epoch {epoch + 1}...")
            model.eval()
            eval_dataset.dataset.crop_len = None
            with torch.no_grad():
                num_eval_samples = min(args.num_plot_examples, len(eval_dataset))
                eval_indices = random.sample(range(len(eval_dataset)), num_eval_samples)

                for i, idx in enumerate(eval_indices):
                    # Get raw data from the underlying dataset if eval_dataset is a Subset
                    if isinstance(eval_dataset, torch.utils.data.Subset):
                        actual_idx = eval_dataset.indices[idx]
                        sample, mel_len, filename = eval_dataset.dataset[actual_idx]
                    else:  # If it's the original dataset type
                        sample, mel_len, filename = eval_dataset[idx]

                    # Convert the sample (numpy array) to a tensor and add batch dimension.
                    sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)

                    # Perform inference
                    with autocast(enabled=True):
                        indices = model.encode(sample_tensor)
                        reconstructed = model.decode(indices)

                    # Determine plot range for this specific sample
                    vmin_eval = sample_tensor.min().item()
                    vmax_eval = sample_tensor.max().item()
                    if reconstructed is not None and reconstructed.numel() > 0:
                        vmin_eval = min(vmin_eval, reconstructed.min().item())
                        vmax_eval = max(vmax_eval, reconstructed.max().item())

                    # Plot the original and reconstructed spectrograms.
                    plot_mel_spectrogram(
                        sample_tensor[0],  # Remove batch dim
                        vmin=vmin_eval, vmax=vmax_eval,
                        save_path=os.path.join(plot_dir,
                                               f'epoch_{epoch + 1:03d}_eval_orig_{i + 1}_{os.path.splitext(filename)[0]}.png'),
                        title=f'Epoch {epoch + 1} Eval - Original ({filename})'
                    )
                    if reconstructed is not None and reconstructed.numel() > 0:
                        plot_mel_spectrogram(
                            reconstructed[0],  # Remove batch dim
                            vmin=vmin_eval, vmax=vmax_eval,
                            save_path=os.path.join(plot_dir,
                                                   f'epoch_{epoch + 1:03d}_eval_recon_{i + 1}_{os.path.splitext(filename)[0]}.png'),
                            title=f'Epoch {epoch + 1} Eval - Reconstructed ({filename})'
                        )
                    else:
                        print(f"Skipping reconstructed plot for eval sample {filename} due to error or empty output.")

            model.train()  # Set back to train mode after evaluation
            eval_dataset.dataset.crop_len = original_crop_size

        # Optional: Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch + 1:03d}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_loss,
                'args': args
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")


# =============================================================================
# Main execution block
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train a Mel Spectrogram Autoencoder")

    # Data args
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory containing the "real" subdirectory with .npy spectrograms.')
    parser.add_argument('--output_dir', type=str, default='training_output',
                        help='Directory to save plots and checkpoints.')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size.')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader.')
    parser.add_argument('--validation_split', type=float, default=0.1,
                        help='Fraction of data to use for validation/evaluation plotting (0.0 to 1.0).')

    # Model args
    parser.add_argument('--mel_channels', type=int, default=88,
                        help='Number of mel frequency channels in the input spectrograms.')
    parser.add_argument('--latent_dim', type=int, default=1024,
                        help='Dimension of the latent space in the autoencoder.')

    # Training args
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for the Adam optimizer.')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 hyperparameter for the Adam optimizer.')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 hyperparameter for the Adam optimizer.')
    parser.add_argument('--eval_interval', type=int, default=2,
                        help='Frequency (in epochs) to perform evaluation plotting.')
    parser.add_argument('--save_interval', type=int, default=1, help='Frequency (in epochs) to save model checkpoints.')
    parser.add_argument('--num_plot_examples', type=int, default=3,
                        help='Number of examples to plot during training and evaluation.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA even if available.')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to a pretrained checkpoint (.pth) to warm start from.')  # <-- ADDED ARGUMENT
    parser.add_argument('--warmup_steps', type=int, default=3000, help='Linear LR warm-up steps.')
    parser.add_argument('--crop_len', type=int, default=256, help='Crop Seq Len')

    args = parser.parse_args()

    # --- Setup ---
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.no_cuda:
        torch.cuda.manual_seed_all(args.seed)
        # Potentially enable deterministic algorithms for reproducibility
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False # Benchmark can introduce non-determinism

    # Set device
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Loading dataset...")
    try:
        # Load full dataset and split: (1-val_split)% training, val_split% evaluation.
        full_dataset = RealMelSpectrogramDataset(args.data_dir, args.crop_len)
        if len(full_dataset) == 0:
            print("Error: Dataset is empty after initialization. Exiting.")
            return  # Exit if no data found

        eval_size = int(args.validation_split * len(full_dataset))
        train_size = len(full_dataset) - eval_size

        if train_size <= 0 or eval_size < 0:  # Eval size can be 0 if split is 0.0
            print(
                f"Error: Invalid train/eval split sizes. Train: {train_size}, Eval: {eval_size}. Check validation_split.")
            return

        print(f"Dataset size: {len(full_dataset)}. Splitting into {train_size} train and {eval_size} eval samples.")
        # Use a generator for reproducibility if seed is set
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, eval_dataset_subset = random_split(full_dataset, [train_size, eval_size], generator=generator)

        # Create DataLoader for training
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers, collate_fn=pad_collate_fn,
                                      pin_memory=use_cuda)  # Use pin_memory if using CUDA
        #eval_dataset_subset.dataset.crop_len = None

        # We pass the eval_dataset_subset (a Subset object) directly to train for plotting
        # No DataLoader needed for eval if just plotting individual samples

    except FileNotFoundError as e:
        print(f"Error initializing dataset: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Model, Loss, Optimizer ---
    print("Initializing model...")
    # Instantiate the ResNet autoencoder and move to device.
    autoencoder = PreEncoder(mel_channels=args.mel_channels, channels=[192, 768, 1024, 1024], kernel_sizes=[3, 5, 7, 11],
                             dropout=0.1, fsq_levels=[8, 5, 5, 5]).to(device)
    #old fsq levels= [8, 5, 5, 5]


    #autoencoder = MVQGenerator(mel_channels=args.mel_channels, channels=[128, 128, 256, 512], kernel_sizes=[3, 5, 7, 9],
     #                        dropout=0.1, fsq_levels=[8, 5, 5, 5]).to(device)
    ae_params = get_param_num(autoencoder)

    discriminator = MelSpectrogramPatchDiscriminator2D(args.mel_channels, hidden_channels = [384, 384, 512, 512, 512], kernel_sizes = [7, 7, 5, 5, 3, 3], stride= [(1, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]).to(device)
    discriminator_mb = MultiBinDiscriminator(args.mel_channels,  hidden_channels = [128, 256, 256, 256, 256], kernel_sizes=[7, 5, 5, 3, 3, 3], n_bins=8, n_no_strides=2).to(device)

    disc_params = get_param_num(discriminator)
    disc2_params = get_param_num(discriminator_mb)

    print("Number of Pre-Encoder Parameters: {:.2f}M".format(ae_params / 1e6))
    print("Number of discriminator parameters: {:.2f}M".format(disc_params / 1e6))
    print("Number of multi bin discriminator parameters: {:.2f}M".format(disc2_params / 1e6))

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print(f"=> Loading checkpoint '{args.pretrained}'")
            try:
                checkpoint = torch.load(args.pretrained, map_location=device, weights_only=False)

                # --- Flexible State Dict Loading ---
                if 'model_state_dict' in checkpoint:
                    pretrained_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:  # Common alternative key
                    pretrained_dict = checkpoint['state_dict']
                else:
                    pretrained_dict = checkpoint  # Assume the whole file is the state_dict

                model_dict = autoencoder.state_dict()
                loaded_keys = []
                skipped_mismatch = []
                skipped_missing_in_model = []  # Should be empty if filtered correctly below
                model_missing_keys = []

                # 0. Handle potential 'module.' prefix if saved with DataParallel/DDP
                # Create a new dict without 'module.' prefix
                clean_pretrained_dict = {}
                for k, v in pretrained_dict.items():
                    if k.startswith('module.'):
                        clean_pretrained_dict[k[7:]] = v  # remove 'module.' prefix
                    else:
                        clean_pretrained_dict[k] = v
                pretrained_dict = clean_pretrained_dict

                # 1. Filter out keys from pretrained_dict that are not in the current model
                filtered_pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

                # 2. Filter out keys where shapes don't match
                final_dict_to_load = {}
                for k, v in filtered_pretrained_dict.items():
                    if v.shape == model_dict[k].shape:
                        final_dict_to_load[k] = v
                        loaded_keys.append(k)
                    else:
                        skipped_mismatch.append(k)
                        print(f"  Warning: Skipping layer {k} due to shape mismatch. "
                              f"Checkpoint shape: {v.shape}, Model shape: {model_dict[k].shape}")

                # 3. Load the filtered state dict
                # Use strict=False for robustness, although strict=True should work if filtering is correct
                missing_keys, unexpected_keys = autoencoder.load_state_dict(final_dict_to_load, strict=False)

                # 4. Report results
                print(f"Successfully loaded {len(loaded_keys)} layers from checkpoint.")
                if skipped_mismatch:
                    print(f"Skipped {len(skipped_mismatch)} layers due to shape mismatch.")
                if missing_keys:
                    print(
                        f"Warning: {len(missing_keys)} layers in the current model were missing from the loaded checkpoint: {missing_keys}")
                if unexpected_keys:
                    # This should ideally be empty due to our filtering, but good to check
                    print(
                        f"Warning: {len(unexpected_keys)} keys from the checkpoint were not found in the model: {unexpected_keys}")

                # Optionally load optimizer, scaler, and epoch (if resuming training)
                # Add checks for key existence before accessing
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch']
                    args.start_epoch = start_epoch  # Store start epoch in args for train loop
                    print(f"  Resuming training from epoch {start_epoch + 1}")
                # Careful when loading optimizer: only load if model architecture hasn't changed drastically
                # or if you are sure the optimizer state is compatible. Usually safer *not* to load optimizer
                # when just warm-starting with potentially different architecture.
                # if 'optimizer_state_dict' in checkpoint:
                #     try:
                #         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                #         print("  Loaded optimizer state.")
                #     except ValueError as e:
                #         print(f"  Warning: Could not load optimizer state, likely due to parameter mismatch: {e}")
                # if 'scaler_state_dict' in checkpoint and use_cuda:
                #     scaler.load_state_dict(checkpoint['scaler_state_dict'])
                #     print("  Loaded GradScaler state.")

            except Exception as e:
                print(f"Error loading checkpoint '{args.pretrained}': {e}")
                print("Training will start from scratch.")
                start_epoch = 0  # Reset start epoch on error
        else:
            print(f"Warning: Pretrained checkpoint not found at '{args.pretrained}'. Training from scratch.")
    else:
        print("No pretrained checkpoint specified. Training from scratch.")
    # --- End of Load Pretrained Weights Section ---

    # Define the reconstruction loss (MSE) computed only on non-padded regions.
    # Using reduction='sum' and manually normalizing later in the loop.
    criterion_recon = nn.MSELoss(reduction='sum')

    # Use an optimizer for the autoencoder parameters
    optimizer = optim.Adam(autoencoder.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_d = optim.Adam(list(discriminator.parameters()) + list(discriminator_mb.parameters()),
                             lr=args.lr * 1.15, betas=(0.5, 0.999))
    # Initialize GradScaler for mixed precision training
    # enabled=use_cuda means scaler is only active when using GPU
    scaler = GradScaler(enabled=use_cuda)

    # --- Start Training ---
    print("Starting training...")
    train(
        model=(autoencoder, discriminator, discriminator_mb),
        train_dataloader=train_dataloader,
        eval_dataset=eval_dataset_subset,  # Pass the Subset for eval plotting
        criterion_recon=criterion_recon,
        optimizer=(optimizer, optimizer_d),
        scaler=scaler,
        device=device,
        args=args
    )

    print("Training finished.")


if __name__ == '__main__':
    main()