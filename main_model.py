import numpy as np
import torch
import torch.nn as nn

from diff_models import diff_CSDI


def conv1d_with_kaiming_init(in_channels, out_channels, kernel_size):
    """Creates a 1D convolutional layer with Kaiming normal initialization."""
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class CSDIBase(nn.Module):
    def __init__(self, target_dim, config, device):
        super(CSDIBase, self).__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_feature_dim = config["model"]["featureemb"]
        self.target_strategy = config["model"]["target_strategy"]

        self.emb_total_dim = self.emb_feature_dim + 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 2
        self.diff_model = diff_CSDI(config_diff, input_dim)

        # Parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        self.beta = self._setup_beta_schedule(config_diff)
        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def _setup_beta_schedule(self, config_diff):
        """Set up the beta schedule for diffusion."""
        if config_diff["schedule"] == "quad":
            return np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            return np.linspace(config_diff["beta_start"], config_diff["beta_end"], self.num_steps)

    def get_randmask(self, observed_mask):
        """Generate a random mask for conditioning."""
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        """Generate a historical mask for conditioning."""
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:  # draw another sample for histmask (i-1 corresponds to another sample)
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1]
        return cond_mask

    def get_side_info(self, cond_mask):
        """Generate side information for the model."""
        B, K, L = cond_mask.shape
        feature_embed = self.embed_layer(torch.arange(self.target_dim).to(self.device))  # (K, emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = feature_embed.permute(0, 3, 2, 1)  # (B, *, K, L)
        side_mask = cond_mask.unsqueeze(1)  # (B, 1, K, L)

        return torch.cat([side_info, side_mask], dim=1)

    def calc_loss_valid(self, observed_data, cond_mask, observed_mask, side_info, is_train):
        """Calculate loss for validation."""
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t)
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1):
        """Calculate the loss for the model."""
        B, K, L = observed_data.shape
        t = (torch.ones(B) * set_t).long().to(self.device) if not is_train else torch.randint(0, self.num_steps, [B]).to(self.device)

        current_alpha = self.alpha_torch[t]  # (B, 1, 1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diff_model(total_input, side_info, t)  # (B, K, L)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        """Set input data to the diffusion model."""
        cond_obs = (cond_mask * observed_data).unsqueeze(1)
        noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
        total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B, 2, K, L)
        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        """Impute missing data."""
        B, K, L = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                cond_obs = (cond_mask * observed_data).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B, 2, K, L)

                predicted = self.diff_model(diff_input, side_info, torch.tensor([t]).to(self.device))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        """Forward pass of the model."""
        observed_data, observed_mask, gt_mask, for_pattern_mask, _, status = self.process_data(batch)

        cond_mask = gt_mask if is_train == 0 else (
            self.get_randmask(observed_mask) if self.target_strategy == "random" else self.get_hist_mask(observed_mask, for_pattern_mask=for_pattern_mask)
        )

        side_info = self.get_side_info(cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        generated_data = self.impute(observed_data, cond_mask, side_info, 3).to(self.device)
        generated_data_median = torch.median(generated_data.permute(0, 1, 3, 2), dim=1).values

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train), \
               generated_data_median.permute(0, 2, 1), (observed_mask - cond_mask), status, observed_data, observed_mask

    def evaluate(self, batch, n_samples):
        """Evaluate the model."""
        observed_data, observed_mask, gt_mask, _, cut_length, _ = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask
            side_info = self.get_side_info(cond_mask)
            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

            for i in range(len(cut_length)):  # To avoid double evaluation
                target_mask[i, ..., 0:cut_length[i].item()] = 0

        return samples, observed_data, target_mask, observed_mask


class TSB_eICU(CSDIBase):
    def __init__(self, config, device, target_dim=45):
        super(TSB_eICU, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        """Process data for the eICU dataset."""
        observed_data = batch["observed_data"].to(self.device).float().unsqueeze(-1)
        observed_mask = batch["observed_mask"].to(self.device).float().unsqueeze(-1)
        gt_mask = batch["gt_mask"].to(self.device).float().unsqueeze(-1)
        status = batch["status"].to(self.device).long()

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return observed_data, observed_mask, gt_mask, for_pattern_mask, cut_length, status


class Bottleneck(nn.Module):
    def __init__(self, in_channel, med_channel, out_channel, downsample=False):
        super(Bottleneck, self).__init__()
        self.stride = 2 if downsample else 1
        self.dropout = nn.Dropout(0.5)

        self.layer = nn.Sequential(
            nn.Conv1d(in_channel, med_channel, 1, self.stride),
            nn.BatchNorm1d(med_channel),
            nn.ReLU(),
            nn.Conv1d(med_channel, med_channel, 3, padding=1),
            nn.BatchNorm1d(med_channel),
            nn.ReLU(),
            nn.Conv1d(med_channel, out_channel, 1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()
        )

        self.res_layer = nn.Conv1d(in_channel, out_channel, 1, self.stride) if in_channel != out_channel else None

    def forward(self, x):
        residual = self.res_layer(x) if self.res_layer else x
        return self.dropout(self.layer(x)) + residual

