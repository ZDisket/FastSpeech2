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
from model.discriminator import MelSpectrogramPatchDiscriminator
from model.loss import LSGANLoss, CharbonnierMel

# =============================================================================
# Dataset & DataLoader for Real Spectrograms
# =============================================================================
class RealMelSpectrogramDataset(Dataset):
    def __init__(self, data_dir):
        self.real_dir = data_dir
        if not os.path.isdir(self.real_dir):
            raise FileNotFoundError(f"Directory not found: {self.real_dir}")

        self.filenames = []
        print(f"Searching for .npy files recursively in: {self.real_dir}")
        for root, _, files in os.walk(self.real_dir):
            for filename in files:
                if filename.endswith('.npy'):
                    full_path = os.path.join(root, filename)
                    self.filenames.append(full_path)

        if not self.filenames:
            print(f"Warning: No .npy files found in {self.real_dir} or its subdirectories.")
        else:
            print(f"Found {len(self.filenames)} .npy files.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        real_path = self.filenames[idx]
        try:
            real_spectrogram = np.load(real_path)
        except Exception as e:
            print(f"Error loading file {real_path}: {e}")
            # Return a dummy item or raise an error
            # Returning dummy data to avoid crashing the whole batch
            # Adjust dummy shape based on expected input
            return np.zeros((1, 88), dtype=np.float32), 1, os.path.basename(real_path) + "_load_error"

        # Ensure spectrogram has 2 dimensions (time, mel_channels)
        if real_spectrogram.ndim != 2:
            print(
                f"Warning: Spectrogram {real_path} has unexpected shape {real_spectrogram.shape}. Skipping or using dummy.")
            # You might want to reshape if appropriate, or return dummy data
            return np.zeros((1, 88), dtype=np.float32), 1, os.path.basename(real_path) + "_shape_error"

        mel_len = real_spectrogram.shape[0]
        # Return basename for easier identification in logs/plots
        filename_base = os.path.basename(real_path)
        return real_spectrogram, mel_len, filename_base


def pad_collate_fn(batch):
    # Filter out potential error items (if __getitem__ returns None on error)
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None, None  # Handle empty batch case

    # Unpack the batch of (spectrogram, mel_len, filename)
    real_spectrograms, mel_lens, filenames = zip(*batch)

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
        spectrogram = spectrogram.cpu().detach().numpy()

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

    model, discriminator = model
    optimizer, optimizer_d = optimizer
    gan_loss = LSGANLoss()
    crit_recon = CharbonnierMel()

    if train_dataloader is None:
        print("Training dataloader is None. Skipping training.")
        return

    n_steps = 0
    for epoch in range(args.num_epochs):
        model.train()
        discriminator.train()
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
                # 1) D on real
                real_logits, real_mask = discriminator(real_spectrograms, mel_lens)

                # Forward pass: reconstruct the input spectrograms
                recon_spectrograms = model(real_spectrograms, mel_lens)  # Expects (batch, seq_len, features)

                # 2) D on fake (detach generator)
                fake_logits, fake_mask = discriminator(recon_spectrograms.detach(), mel_lens)
                loss_D = gan_loss.discriminator_loss(real_logits, fake_logits, real_mask, fake_mask)
                scaler.scale(loss_D).backward()
                scaler.step(optimizer_d)
                scaler.update()

                # Ensure output shape matches input shape
                if recon_spectrograms.shape != real_spectrograms.shape:
                    print(f"Shape mismatch! Input: {real_spectrograms.shape}, Output: {recon_spectrograms.shape}")
                    # Decide how to handle: maybe skip batch, maybe raise error
                    continue  # Skip this batch

                # Create a mask to ignore padded regions.
                max_mel_len = real_spectrograms.size(1)  # mel_len is now dimension 1 (time)
                # Create mask (batch, max_mel_len)
                mask_time = torch.arange(max_mel_len, device=device).expand(len(mel_lens),
                                                                            max_mel_len) < mel_lens.unsqueeze(1)
                # Expand mask to match spectrogram shape (batch, max_mel_len, mel_channels)
                mask = mask_time.unsqueeze(2).expand(-1, -1, real_spectrograms.size(2)).float()

                # Compute reconstruction loss only on valid (non-padded) elements.
                loss_recon = crit_recon(recon_spectrograms, real_spectrograms, mel_lens)

                gen_logits, gen_mask = discriminator(recon_spectrograms, mel_lens)
                loss_gan = gan_loss.generator_loss(gen_logits, gen_mask)
                loss = loss_recon + loss_gan

            # Backward pass and optimization step using GradScaler
            scaler.scale(loss).backward()
            # Unscales gradients and calls optimizer.step()
            scaler.step(optimizer)
            # Updates the scale for next iteration
            scaler.update()

            current_loss = loss.item()
            epoch_loss += current_loss
            loop.set_postfix(D=loss_D.item(),
                             G_recon=loss_recon.item(),
                             G_gan=loss_gan.item(),
                             total_loss=loss.item(),)

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch [{epoch + 1}/{args.num_epochs}], Average Loss: {avg_loss:.6f}")

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
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate for the Adam optimizer.')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 hyperparameter for the Adam optimizer.')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 hyperparameter for the Adam optimizer.')
    parser.add_argument('--eval_interval', type=int, default=2,
                        help='Frequency (in epochs) to perform evaluation plotting.')
    parser.add_argument('--save_interval', type=int, default=5, help='Frequency (in epochs) to save model checkpoints.')
    parser.add_argument('--num_plot_examples', type=int, default=3,
                        help='Number of examples to plot during training and evaluation.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA even if available.')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to a pretrained checkpoint (.pth) to warm start from.')  # <-- ADDED ARGUMENT

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
        full_dataset = RealMelSpectrogramDataset(args.data_dir)
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
    autoencoder = PreEncoder(mel_channels=args.mel_channels, channels=[512, 512, 384], kernel_sizes=[11, 5, 3],
                             dropout=0.1, fsq_levels=[8, 6, 5]).to(device)

    ae_params = get_param_num(autoencoder)

    discriminator = MelSpectrogramPatchDiscriminator(args.mel_channels,  hidden_channels = [1024, 1024, 1024, 2048]).to(device)
    disc_params = get_param_num(discriminator)


    print("Number of Pre-Encoder Parameters: {:.2f}M".format(ae_params / 1e6))
    print("Number of discriminator parameters: {:.2f}M".format(disc_params / 1e6))

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
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # Initialize GradScaler for mixed precision training
    # enabled=use_cuda means scaler is only active when using GPU
    scaler = GradScaler(enabled=use_cuda)

    # --- Start Training ---
    print("Starting training...")
    train(
        model=(autoencoder, discriminator),
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