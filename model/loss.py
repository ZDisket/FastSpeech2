import torch
import torch.nn as nn
from numba import jit
import numpy as np
from torch.nn import functional as F
import torchbnn as bnn
from .modules import SafeLogSoftmax
from utils.tools import compute_phoneme_level_features_optimized


class LSGANLoss(nn.Module):
    def __init__(self, real_label=1.0, fake_label=0.0, decay=0.99, use_lecam=True):
        super(LSGANLoss, self).__init__()
        self.real_label = real_label
        self.fake_label = fake_label
        self.decay = decay
        self.use_lecam = use_lecam
        self.criterion = nn.MSELoss()

        # Initialize EMA variables for discriminator predictions
        self.register_buffer("ema_real", torch.tensor(0.0))
        self.register_buffer("ema_fake", torch.tensor(0.0))
        self.ema_initialized = False  # Flag to check initialization

    def update_ema(self, current_real, current_fake):
        """Update EMA values for real and fake scores."""
        # Ensure EMA tensors are on the same device as the inputs
        device = current_real.device
        self.ema_real = self.ema_real.to(device)
        self.ema_fake = self.ema_fake.to(device)

        if not self.ema_initialized:
            # Initialize EMA values with the mean of current inputs (detached from graph)
            self.ema_real.copy_(current_real.mean().detach())
            self.ema_fake.copy_(current_fake.mean().detach())
            self.ema_initialized = True
        else:
            # Update EMA values using exponential moving average formula
            self.ema_real.mul_(self.decay).add_(1 - self.decay, current_real.mean().detach())
            self.ema_fake.mul_(self.decay).add_(1 - self.decay, current_fake.mean().detach())

    def lecam_loss(self, real_output, fake_output):
        """Compute the LeCam regularization loss."""
        # Ensure EMA tensors are on the same device as the outputs
        device = real_output.device
        self.ema_real = self.ema_real.to(device)
        self.ema_fake = self.ema_fake.to(device)

        # Detach EMA values to ensure no gradients flow through them
        ema_real = self.ema_real.detach()
        ema_fake = self.ema_fake.detach()

        # Relu to ensure non-negative loss components
        lecam_real = torch.mean((real_output - ema_fake).clamp(min=0) ** 2)
        lecam_fake = torch.mean((ema_real - fake_output).clamp(min=0) ** 2)
        return lecam_real + lecam_fake

    def discriminator_loss(self, real_output, fake_output):
        """Discriminator loss with optional LeCam regularization."""
        # Standard LSGAN loss
        real_loss = self.criterion(real_output, torch.full_like(real_output, self.real_label))
        fake_loss = self.criterion(fake_output, torch.full_like(fake_output, self.fake_label))
        lsgan_loss = 0.5 * (real_loss + fake_loss)

        # Compute and optionally add LeCam regularization
        if self.use_lecam:
            # Update EMA values
            self.update_ema(real_output, fake_output)
            # Compute LeCam loss
            lecam_reg = self.lecam_loss(real_output, fake_output)
            return lsgan_loss + lecam_reg
        else:
            return lsgan_loss

    def generator_loss(self, fake_output):
        """Generator loss remains the same."""
        return self.criterion(fake_output, torch.full_like(fake_output, self.real_label))


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1) - generalized to handle tensors of arbitrary shapes."""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        # Flatten the tensor dimensions except for the batch dimension
        # to handle arbitrary tensor shapes.
        if x.dim() > 1:
            x = x.view(x.size(0), -1)
        if y.dim() > 1:
            y = y.view(y.size(0), -1)

        # Calculate the Charbonnier loss
        diff = x - y
        loss = torch.sqrt(diff.pow(2) + self.eps ** 2).mean()  # Mean across all dimensions except batch
        return loss


class Charbonnier1D(nn.Module):
    """Charbonnier Loss for 1D sequences (batch_size, seq_len)."""

    def __init__(self, eps=1e-6):
        super(Charbonnier1D, self).__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute the Charbonnier loss between predictions and ground truth for 1D sequences.

        Parameters:
            x (torch.Tensor): Predictions of shape (batch_size, seq_len).
            y (torch.Tensor): Ground truth of shape (batch_size, seq_len).
            mask (torch.Tensor): Boolean mask of shape (batch_size, seq_len), where True means excluded (invalid).

        Returns:
            torch.Tensor: Computed Charbonnier loss for valid elements.
        """
        assert x.shape == y.shape, "Shape mismatch between predictions and ground truth"
        assert mask.shape == x.shape, "Shape mismatch between mask and predictions/ground_truth"

        # Masked difference calculation
        diff = x - y
        diff = diff[~mask]  # Include only valid elements using the inverted mask

        # Charbonnier loss calculation
        loss = torch.sqrt(diff.pow(2) + self.eps ** 2).mean()  # Mean across valid dimensions
        return loss


class MSE1D(nn.Module):
    """Mean Squared Error Loss for 1D sequences with masking (batch_size, seq_len)."""

    def __init__(self):
        super(MSE1D, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute the Mean Squared Error loss between predictions and ground truth for 1D sequences,
        considering a mask to exclude certain entries from the loss computation.

        Parameters:
            x (torch.Tensor): Predictions of shape (batch_size, seq_len).
            y (torch.Tensor): Ground truth of shape (batch_size, seq_len).
            mask (torch.Tensor): Boolean mask of shape (batch_size, seq_len), where True means excluded (invalid).

        Returns:
            torch.Tensor: Computed MSE loss for valid elements.
        """

        assert x.shape == y.shape, f"Shape mismatch between predictions and ground truth, {x.size()}, vs {y.size()}"
        assert mask.shape == x.shape, "Shape mismatch between mask and predictions/ground_truth"

        # Apply the mask by selecting elements where mask is False
        valid_x = x[~mask]
        valid_y = y[~mask]

        # Calculate MSE for the valid elements
        mse_loss = F.mse_loss(valid_x, valid_y, reduction='mean')  # Calculate mean only over the unmasked elements
        return mse_loss


class TemporalConsistencyLoss(nn.Module):
    def __init__(self, weight: float = 1.0, use_mse: bool = False):
        """
        Initializes the temporal consistency loss module.

        Parameters:
            weight (float): Weight of the temporal consistency loss.
            use_mse (bool): Flag to use MSE loss instead of L1 loss.
        """
        super(TemporalConsistencyLoss, self).__init__()
        self.weight = weight
        self.use_mse = use_mse

    def forward(self, predictions: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute the temporal consistency loss between predictions and ground truth.

        Parameters:
            predictions (torch.Tensor): Predictions of shape (batch_size, seq_length).
            ground_truth (torch.Tensor): Ground truth of shape (batch_size, seq_length).
            mask (torch.Tensor): Mask of shape (batch_size, seq_length), where True means excluded (invalid).

        Returns:
            torch.Tensor: Computed temporal consistency loss.
        """
        # Ensure predictions and ground truth have the same shape
        assert predictions.shape == ground_truth.shape, "Shape mismatch between predictions and ground truth"
        assert mask.shape == predictions.shape, f"Shape mismatch between mask and predictions/ground_truth, mask: {mask.size()}, preds = {predictions.size()}"

        # Compute consecutive differences for predictions and ground truth
        diff_pred = predictions[:, 1:] - predictions[:, :-1]
        diff_gt = ground_truth[:, 1:] - ground_truth[:, :-1]

        # Create the consecutive differences mask
        mask_diff = ~(mask[:, 1:] | mask[:, :-1])

        # Apply the mask to the differences
        diff_pred_masked = diff_pred[mask_diff]
        diff_gt_masked = diff_gt[mask_diff]

        # Calculate the temporal consistency loss
        if self.use_mse:
            # Use mean squared error loss
            temporal_loss = torch.mean((diff_pred_masked - diff_gt_masked) ** 2)
        else:
            # Use L1 loss (default)
            temporal_loss = torch.mean(torch.abs(diff_pred_masked - diff_gt_masked))

        return temporal_loss * self.weight


class BinLoss(torch.nn.modules.loss._Loss):
    def __init__(self):
        super().__init__()

    def forward(self, hard_attention, soft_attention):
        soft_attention = torch.nan_to_num(soft_attention)
        log_input = torch.clamp(soft_attention[hard_attention == 1], min=1e-12)
        log_sum = torch.log(log_input).sum()
        return -log_sum / hard_attention.sum()


class ForwardSumLoss(torch.nn.modules.loss._Loss):
    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.log_softmax = SafeLogSoftmax(dim=3)
        self.ctc_loss = torch.nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(input=attn_logprob, pad=(1, 0), value=self.blank_logprob)

        total_loss = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid] + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[: query_lens[bid], :, : key_lens[bid] + 1]

            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            loss = self.ctc_loss(
                curr_logprob,
                target_seq,
                input_lengths=query_lens[bid: bid + 1],
                target_lengths=key_lens[bid: bid + 1],
            )
            total_loss += loss

        total_loss /= attn_logprob.shape[0]
        return total_loss


def pad_tensor_to_max_width(tensor, lens_w):
    # Determine the current width (w) and the desired width
    current_w = tensor.shape[2]
    max_w = lens_w.max().item()

    # Calculate the padding needed on the right of the tensor
    if current_w < max_w:
        # Pad only the width (3rd dimension), padding format is (left, right, top, bottom)
        padding = (0, max_w - current_w, 0, 0)
        tensor = F.pad(tensor, padding, "constant", 0)  # Adding zero padding

    return tensor


class DurationMatchingLoss(nn.Module):
    def __init__(self):
        super(DurationMatchingLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, log_durations: torch.Tensor, mel_mask: torch.Tensor, duration_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss that encourages the sum of log durations to match the log lengths of mel-spectrograms,
        taking into account masks for invalid durations and mel-spectrogram lengths.

        Args:
            log_durations (torch.Tensor): A tensor of shape (batch, max_duration_len) containing log durations.
            mel_mask (torch.Tensor): A boolean tensor of shape (batch, max_mel_len) where True indicates a valid mel element.
            duration_mask (torch.Tensor): A boolean tensor of shape (batch, max_duration_len) where True indicates a valid duration.

        Returns:
            torch.Tensor: The computed loss.
        """
        # Ensure the masks are boolean tensors
        mel_mask = mel_mask.bool()
        duration_mask = duration_mask.bool()

        # Infer the mel lengths from the mel mask
        mel_lengths = mel_mask.sum(dim=1).float()

        # Apply the mask to the log durations, setting invalid positions to a very large negative number
        log_durations_masked = log_durations.masked_fill(duration_mask == 0, -float('inf'))

        # Sum the log durations for each sequence using logsumexp to maintain numerical stability
        log_duration_sums = log_durations_masked.logsumexp(dim=1)

        # Convert mel lengths to log space
        log_mel_lengths = torch.log(mel_lengths + 1e-6)

        # Compute the mean squared error loss between log duration sums and log mel lengths
        loss = self.mse_loss(log_duration_sums, log_mel_lengths)

        return loss


class FastSpeech3Loss(nn.Module):
    """ FastSpeech2+1 Loss """

    def __init__(self, preprocess_config, model_config, train_config):
        super(FastSpeech3Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

        self.forward_sum = ForwardSumLoss()
        self.bin_loss = BinLoss()
        self.mse2_loss = MSE1D()
        self.charb_loss = Charbonnier1D()
        self.temp_loss = TemporalConsistencyLoss(1.0, True)  # I tested 0.35, 0.5, 0.75, but 1.0 is best
        self.pe_temp_loss = TemporalConsistencyLoss(1.0, True)  # I tested 0.35, 0.5, 0.75, but 1.0 is best
        self.kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)

        self.duration_kl_loss_weight = 1.0
        self.pitch_energy_kl_loss_weight = 1.0


        # With all our new losses (attention, masked duration, temporal), the mel loss (individual) goes from being 20% of the loss
        # to just 6% and audio quality suffers greatly. We re-weight, although too much is detrimental.
        self.mel_loss_weight = 1.0
        self.mel_postnet_loss_weight = 1.0

        self.bin_loss_start_epoch = train_config["optimizer"]["bin_loss_start_epoch"]
        self.bin_loss_warmup_epochs = train_config["optimizer"]["bin_loss_warmup_epochs"]

    def forward(self, inputs, predictions, epoch, model=None):
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            _,
            _,
            _,
        ) = inputs[6:]
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            input_lengths,
            output_lengths,
            attn_logprob,
            attn_hard,
            attn_soft,
            attn_hard_dur,
            _,
        ) = predictions

        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(attn_hard_dur.float() + 1e-6)

        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_level":
            pitch_targets = compute_phoneme_level_features_optimized(attn_hard_dur, pitch_targets)

        if self.energy_feature_level == "phoneme_level":
            energy_targets = compute_phoneme_level_features_optimized(attn_hard_dur, energy_targets)

        pitch_t, pitch_p = pitch_targets.clone(), pitch_predictions.clone()
        energy_t, energy_p = energy_targets.clone(), energy_predictions.clone()

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        # log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        # log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets) * self.mel_loss_weight
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets) * self.mel_postnet_loss_weight

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)

        # these masks are True=valid, our loss functions take True=invalid
        duration_loss = self.mse2_loss(
            log_duration_predictions,
            log_duration_targets,
            ~src_masks,
        )

        # temporal consistency

        duration_temporal = self.temp_loss(log_duration_predictions,
                                           log_duration_targets,
                                           ~src_masks)

        pitch_energy_mask = ~mel_masks if self.pitch_feature_level == "frame_level" else ~src_masks

        pitch_temporal = self.pe_temp_loss(pitch_p,
                                           pitch_t,
                                           pitch_energy_mask)

        energy_temporal = self.pe_temp_loss(energy_p,
                                            energy_t,
                                            pitch_energy_mask)

        total_temporal = duration_temporal + pitch_temporal + energy_temporal


        # sometimes (almost always for some reason), output_lengths.max() == attn_logprob.size(2) + 1
        output_lengths = torch.clamp_max(output_lengths, attn_logprob.size(2))
        al_forward_sum = self.forward_sum(attn_logprob=attn_logprob, in_lens=input_lengths, out_lens=output_lengths)

        total_attn_loss = al_forward_sum

        if epoch > self.bin_loss_start_epoch:
            bin_loss_scale = min((epoch - self.bin_loss_start_epoch) / self.bin_loss_warmup_epochs, 1.0)
            al_match_loss = self.bin_loss(hard_attention=attn_hard, soft_attention=attn_soft) * bin_loss_scale
            total_attn_loss += al_match_loss

        kl_duration = self.kl_loss(model.variance_adaptor.duration_predictor)
        kl_energy = self.kl_loss(model.variance_adaptor.energy_predictor)
        kl_pitch = self.kl_loss(model.variance_adaptor.pitch_predictor)

        kl_pitch_energy = kl_energy + kl_pitch


        total_kl = kl_duration * self.duration_kl_loss_weight + kl_pitch_energy * self.pitch_energy_kl_loss_weight

        total_loss = (
                mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss +
                total_attn_loss + total_temporal + total_kl
        )

        ret = (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
            total_attn_loss,
            duration_temporal,
            total_temporal,
            kl_duration,
            kl_pitch_energy,
        )
        return list(ret)


class MaskedMAE(nn.Module):
    """
    A masked regression loss that can switch between MAE, MSE, and Charbonnier losses.

    Args:
        loss_type (str): One of 'mae', 'mse', or 'charbonnier'. Determines which loss to use.
        epsilon (float): Small constant used in the Charbonnier loss. Default is 1e-6.
    """

    def __init__(self, loss_type="mae", epsilon=1e-6):
        super(MaskedMAE, self).__init__()
        if loss_type not in ["mae", "mse", "charbonnier"]:
            raise ValueError("loss_type must be one of 'mae', 'mse', or 'charbonnier'")
        self.loss_type = loss_type
        self.epsilon = epsilon

    def forward(self, pred, target, mask):
        """
        Args:
            pred: Tensor of shape (B, T, mel_dim) containing predictions.
            target: Tensor of shape (B, T, mel_dim) containing ground truth values.
            mask: Float Tensor of shape (B, T, 1) with 1.0 for valid positions and 0.0 for padded positions.

        Returns:
            Scalar loss normalized by the number of valid (non-padded) elements.
        """
        error = pred - target

        if self.loss_type == "mae":
            loss = torch.abs(error)
        elif self.loss_type == "mse":
            loss = error ** 2
        elif self.loss_type == "charbonnier":
            loss = torch.sqrt(error ** 2 + self.epsilon ** 2)
        else:
            raise ValueError("Unsupported loss_type: {}".format(self.loss_type))

        loss = loss * mask
        total_loss = loss.sum() / mask.sum()
        return total_loss

class MaskedBCE(nn.Module):
    """
    Computes Binary Cross Entropy loss with logits on a per-element basis,
    masking out the padded positions. A positive weight is applied to handle class imbalance.
    """
    def __init__(self, pos_weight=8.0):
        super(MaskedBCE, self).__init__()
        # pos_weight must be a tensor and registered so that it moves with the model.
        self.register_buffer('pos_weight', torch.tensor(pos_weight))
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=self.pos_weight)

    def forward(self, pred, target, mask):
        """
        Args:
            pred: Tensor of shape (B, T), logits for the gate prediction.
            target: Tensor of shape (B, T), where 1 indicates the stop token.
            mask: Float Tensor of shape (B, T) with 1.0 for valid positions, 0.0 for padded.
        Returns:
            Scalar BCE loss normalized by the number of valid elements.
        """
        loss = self.bce_loss(pred, target)  # (B, T)
        loss = loss * mask
        total_loss = loss.sum() / mask.sum()
        return total_loss


class SturmLoss(nn.Module):
    """
    Loss function for the TTS model (Sturmschlag) that computes:
      - MAE loss (L1 loss) for the mel spectrogram, using a mask to ignore padded positions.
      - Gate loss using BCEWithLogitsLoss (with a positive weight of 8.0) for the stop token prediction,
        again masked over the valid frames.

    The expected input from the model's forward pass is a tuple:
        (mel_pred, gate_pred, text_mask, mel_mask)
    And the batch tuple is assumed to be:
        (speakers, texts, src_lens, mels, mel_lens, em_hidden)
    where:
        - mel_pred: Predicted mel spectrogram, shape (B, T, mel_dim)
        - gate_pred: Predicted gate logits, shape (B, T, 1)
        - mel_mask: Boolean mask (B, T), where True indicates padded positions.
        - mels: Ground truth mel spectrogram, shape (B, T, mel_dim)
        - mel_lens: Actual lengths of each mel sequence (B,)
    """

    def __init__(self, pos_weight=8.0, mel_regression="mae"):
        super(SturmLoss, self).__init__()
        self.masked_mae = MaskedMAE(mel_regression)
        self.masked_bce = MaskedBCE(pos_weight=pos_weight)

    def forward(self, batch, model_out):
        """
        Args:
            model_out (tuple): Contains:
                mel_pred: Predicted mel spectrogram, shape (B, T, mel_dim)
                gate_pred: Predicted gate logits, shape (B, T, 1)
                text_mask: Mask for text (unused in loss computation)
                mel_mask: Bool Tensor of shape (B, T), where True indicates padded positions.
            batch (tuple): Contains:
                speakers, texts, src_lens, mels, mel_lens, em_hidden
                - mels: Ground truth mel spectrogram, shape (B, T, mel_dim)
                - mel_lens: Tensor of shape (B,) containing the actual mel lengths.

        Returns:
            total_loss: Sum of mel loss and gate loss.
            mel_loss: Scalar MAE loss for the mel spectrogram.
            gate_loss: Scalar BCE loss for the gate prediction.
        """
        mel_pred, gate_pred, text_mask, mel_mask = model_out
        mels_target = batch[5]  # (B, T, mel_dim)
        mel_lens = batch[4]  # (B,)

        # Create a valid mask for mel loss: invert mel_mask (assumed to be True for padded)
        valid_mel_mask = (~mel_mask).unsqueeze(-1).float()  # (B, T, 1)
        mel_loss = self.masked_mae(mel_pred, mels_target, valid_mel_mask)

        # Build the gate target vectorized:
        B, T, _ = gate_pred.size()
        # Create a time index tensor: shape (B, T)
        time_idx = torch.arange(T, device=gate_pred.device).unsqueeze(0).expand(B, T)
        # For each sample, mark the last valid time step (mel_lens - 1) as the stop token (1), others 0.
        gate_target = (time_idx == (mel_lens - 1).unsqueeze(1)).float()  # (B, T)

        # Squeeze gate_pred from (B, T, 1) to (B, T)
        gate_pred = gate_pred.squeeze(-1)
        # Create a valid mask for gate loss: same as mel, but shape (B, T)
        valid_gate_mask = (~mel_mask).float()
        gate_loss = self.masked_bce(gate_pred, gate_target, valid_gate_mask)

        total_loss = mel_loss + gate_loss

        ret = (
            total_loss,
            mel_loss,
            gate_loss,
        )

        return list(ret)
