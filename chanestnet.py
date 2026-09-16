import argparse
import time
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

parser = argparse.ArgumentParser(description="ChanEstNet training, channel estimation, and LRT evaluation for multi-tag ambient backscatter detection.")
parser.add_argument("--M", type=int, default=4, help="Number of reader antennas.")
parser.add_argument("--N_tags", type=int, default=2, help="Number of passive tags.")
parser.add_argument("--K", type=int, default=20, help="RF source symbols per tag symbol period.")
parser.add_argument("--T", type=int, default=160, help="Total tag symbol periods per frame.")
parser.add_argument("--P", type=int, default=32, help="Number of pilot symbol periods per frame.")
parser.add_argument("--mod_type", type=str, default="qpsk", choices=["cscg", "qpsk", "16qam"], help="Ambient modulation type.")
parser.add_argument("--num_train_frames", type=int, default=25600, help="Number of training frames.")
parser.add_argument("--num_val_frames", type=int, default=2000, help="Number of validation frames.")
parser.add_argument("--batch_size", type=int, default=128, help="Training batch size.")
parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs.")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
parser.add_argument("--lr_decay_every", type=int, default=30, help="StepLR interval in epochs. <=0 disables decay.")
parser.add_argument("--lr_decay_gamma", type=float, default=0.5, help="StepLR decay factor.")
parser.add_argument("--train_snr_db", type=str, default="0,4,8,12,16,20", help="Comma-separated training SNRs (dB).")
parser.add_argument("--val_snr_db", type=float, default=10.0, help="Validation SNR (dB) for checkpoint selection.")
parser.add_argument("--att_db", type=float, default=-20.0, help="Attenuation (dB).")
parser.add_argument("--num_online_frames", type=int, default=10000, help="Frames per SNR for evaluation.")
parser.add_argument("--eval_snr_db", type=str, default="0,4,8,12,16,20", help="Comma-separated SNRs (dB) for post-training BER evaluation.")
parser.add_argument("--num_workers", type=int, default=0, help="Number of data-loading workers.")
parser.add_argument("--hidden_dim", type=int, default=32, help="Number of filters in the hidden Conv1D layers.")
parser.add_argument("--kernel_size", type=int, default=3, help="Conv1D kernel size.")
parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to store checkpoints and logs.")

args = parser.parse_args()

M = args.M
N_tags = args.N_tags
K = args.K
T = args.T
P = args.P
mod_type = args.mod_type
num_train_frames = args.num_train_frames
num_val_frames = args.num_val_frames
batch_size = args.batch_size
num_epochs = args.num_epochs
learning_rate = args.learning_rate
lr_decay_every = args.lr_decay_every
lr_decay_gamma = args.lr_decay_gamma
att_dB = args.att_db
att_factor = np.sqrt(10 ** (att_dB / 10))
hidden_dim = args.hidden_dim
kernel_size = args.kernel_size


def float_tag(value):
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.6g}".replace("-", "m").replace(".", "p")


offline_snr_dbs = [float(s.strip()) for s in args.train_snr_db.split(",") if s.strip()]
offline_noise_vars = [10 ** (-snr / 10.0) for snr in offline_snr_dbs]
if len(offline_snr_dbs) == 1:
    offline_SNR_dB = offline_snr_dbs[0]; _snr_tag = float_tag(offline_SNR_dB)
else:
    offline_SNR_dB = None
    _snr_tag = "mix" + "_".join(float_tag(s) for s in offline_snr_dbs)
val_SNR_dB = args.val_snr_db
val_noise_var = 10 ** (-val_SNR_dB / 10.0)
num_online_frames = args.num_online_frames
snr_dbs = [float(s.strip()) for s in args.eval_snr_db.split(",") if s.strip()]
num_workers = args.num_workers
target_type = "channels"

data_len = T - P
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

checkpoint_dir = Path(args.output_dir)
checkpoint_dir.mkdir(parents=True, exist_ok=True)

checkpoint_suffix = (
    f"chanestnet_k{kernel_size}_hid{hidden_dim}_M{M}_N{N_tags}_K{K}_T{T}_P{P}_mod-{mod_type}_"
    f"trainSNR{_snr_tag}_valSNR{float_tag(val_SNR_dB)}_"
    f"att{float_tag(att_dB)}dB.pt"
)
checkpoint_path = checkpoint_dir / checkpoint_suffix

log_path = None


def log_message(msg):
    print(msg)
    if log_path is not None:
        with log_path.open("a") as f:
            f.write(msg + "\n")


def sample_source_symbols(mod_type, K):
    if mod_type.lower() == "cscg":
        s = (np.random.randn(K) + 1j * np.random.randn(K)) / np.sqrt(2)
    elif mod_type.lower() == "qpsk":
        const = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
        s = np.random.choice(const, size=K)
    elif mod_type.lower() == "16qam":
        vals = np.array([-3, -1, 1, 3]) / np.sqrt(10)
        c = np.array([r + 1j * i for r in vals for i in vals])
        s = np.random.choice(c, size=K)
    else:
        raise ValueError(mod_type)
    return s


def sample_source_symbols_batch(mod_type, K, R):
    if mod_type.lower() == "cscg":
        return ((np.random.randn(R, K) + 1j * np.random.randn(R, K)) / np.sqrt(2)).astype(np.complex64)
    elif mod_type.lower() == "qpsk":
        const = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
        return np.random.choice(const, size=(R, K)).astype(np.complex64)
    elif mod_type.lower() == "16qam":
        vals = np.array([-3, -1, 1, 3]) / np.sqrt(10)
        c = np.array([r + 1j * i for r in vals for i in vals])
        return np.random.choice(c, size=(R, K)).astype(np.complex64)
    else:
        raise ValueError(mod_type)


def generate_random_channels(M, N_tags):
    h = (np.random.randn(M) + 1j * np.random.randn(M)) / np.sqrt(2)
    g_list = [(np.random.randn(M) + 1j * np.random.randn(M)) / np.sqrt(2)
              for _ in range(N_tags)]
    return h, g_list


def build_subslot_and_corr(w, M, K, noise_var):
    s = sample_source_symbols(mod_type, K)
    noise = ((np.random.randn(M, K) + 1j * np.random.randn(M, K)) / np.sqrt(2)
             * np.sqrt(noise_var)).astype(np.complex64)
    wave = (w[:, None] * s[None, :] + noise).astype(np.complex64)
    denom = float(np.sum(np.abs(s) ** 2))
    num = wave @ np.conjugate(s).astype(np.complex64)
    return wave, (num / denom).astype(np.complex64)


def generate_onehot_pilot_corr_cnn(M, N_tags, K, P, noise_var, att_factor, target_type="channels"):
    h, g_list = generate_random_channels(M, N_tags)
    v_list = [att_factor * g for g in g_list]
    num_configs = N_tags + 1
    if P < num_configs:
        raise ValueError(f"P={P} < N+1={num_configs}")

    W_eff = np.stack([h] + [h + v for v in v_list], axis=0).astype(np.complex64)  # (num_configs, M)

    num_repetitions = P // num_configs
    remainder = P % num_configs
    config_assignment = np.concatenate([
        np.tile(np.arange(num_configs), num_repetitions),
        np.arange(remainder),
    ]).astype(np.int64)  # (P,)

    S = sample_source_symbols_batch(mod_type, K, P)                                   # (P, K)
    noise = ((np.random.randn(P, M, K) + 1j * np.random.randn(P, M, K)) / np.sqrt(2)
             * np.sqrt(noise_var)).astype(np.complex64)                                # (P, M, K)
    W_per_slot = W_eff[config_assignment]                                             # (P, M)
    wave = W_per_slot[:, :, None] * S[:, None, :] + noise                            # (P, M, K)

    num = np.einsum('pmk,pk->pm', wave, np.conjugate(S))                              # (P, M)
    denom = np.sum(np.abs(S) ** 2, axis=1)                                            # (P,)
    corr_per_slot = (num / denom[:, None]).astype(np.complex64)                       # (P, M)

    counts = np.bincount(config_assignment, minlength=num_configs).astype(np.float32)  # (num_configs,)
    sub_corr = np.zeros((num_configs, M), dtype=np.complex64)
    np.add.at(sub_corr, config_assignment, corr_per_slot)
    sub_corr = (sub_corr / counts[:, None]).astype(np.complex64)                       # (num_configs, M)

    sub_corr_ri = np.stack([sub_corr.real, sub_corr.imag], axis=0)                      # (2, num_configs, M)

    label_rows = [np.concatenate([h.real, h.imag], axis=0)]
    for v in v_list:
        label_rows.append(np.concatenate([v.real, v.imag], axis=0))
    label = np.concatenate(label_rows, axis=0)

    return sub_corr_ri.astype(np.float32), label.astype(np.float32)


class CorrCNNChannelDatasetWithNoise(Dataset):
    def __init__(self, num_frames, M, N_tags, K, P, noise_vars, att_factor,
                 target_type="channels", precompute=False):
        super().__init__()
        self.num_frames = num_frames
        self.M = M; self.N_tags = N_tags; self.K = K; self.P = P
        if isinstance(noise_vars, (int, float)):
            self.noise_vars = [float(noise_vars)]
        else:
            self.noise_vars = list(noise_vars)
        self.att_factor = att_factor
        self.target_type = target_type
        self.precompute = precompute
        self.cached_items = None
        if self.precompute:
            self.cached_items = []
            for i in range(self.num_frames):
                nv = self.noise_vars[i % len(self.noise_vars)]
                s, l = generate_onehot_pilot_corr_cnn(
                    self.M, self.N_tags, self.K, self.P,
                    nv, self.att_factor, self.target_type)
                self.cached_items.append((
                    torch.tensor(s, dtype=torch.float32),
                    torch.tensor(l, dtype=torch.float32),
                    torch.tensor(nv, dtype=torch.float32),
                ))

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        if self.cached_items is not None:
            return self.cached_items[idx]
        nv = self.noise_vars[idx % len(self.noise_vars)]
        s, l = generate_onehot_pilot_corr_cnn(
            self.M, self.N_tags, self.K, self.P,
            nv, self.att_factor, self.target_type)
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(l, dtype=torch.float32),
            torch.tensor(nv, dtype=torch.float32),
        )


class ChanEstNet(nn.Module):
    def __init__(self, M, N_tags, hidden=32, kernel_size=3):
        super().__init__()
        in_ch = 2 * (N_tags + 1)
        self.in_ch = in_ch; self.M = M; self.N_plus_1 = N_tags + 1
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(hidden, in_ch, kernel_size=kernel_size, padding=kernel_size // 2),
        )

    def forward(self, x):
        B = x.size(0)
        x = x.permute(0, 2, 1, 3).contiguous().reshape(B, self.in_ch, self.M)
        return self.net(x).reshape(B, -1)


def build_model(M, N_tags, hidden_dim, kernel_size):
    return ChanEstNet(M, N_tags, hidden=hidden_dim, kernel_size=kernel_size)


def snr_balanced_mse_loss(pred, label, noise_var):
    per_sample_se = ((pred - label) ** 2).sum(dim=1)                # (B,)
    feat_dim = pred.size(1)
    per_sample_mse = per_sample_se / feat_dim                       # (B,)
    weighted = per_sample_mse / noise_var                           # (B,)
    return weighted.mean()


def train_model(model, optimizer, scheduler, train_loader, val_loader,
                num_epochs, checkpoint_path):
    best_val = float("inf")
    best_state = None
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0; total_train_mse = 0.0; count_train = 0
        for in_batch, label_batch, nv_batch in train_loader:
            in_batch = in_batch.to(device); label_batch = label_batch.to(device)
            nv_batch = nv_batch.to(device)
            optimizer.zero_grad()
            pred = model(in_batch)
            loss = snr_balanced_mse_loss(pred, label_batch, nv_batch)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                mse = F.mse_loss(pred, label_batch, reduction="mean")
            total_train_loss += loss.item() * in_batch.size(0)
            total_train_mse += mse.item() * in_batch.size(0)
            count_train += in_batch.size(0)
        avg_train_loss = total_train_loss / count_train
        avg_train_mse = total_train_mse / count_train

        model.eval()
        total_val_mse = 0.0; count_val = 0
        with torch.no_grad():
            for in_batch, label_batch, nv_batch in val_loader:
                in_batch = in_batch.to(device); label_batch = label_batch.to(device)
                pred = model(in_batch)
                l_mse = F.mse_loss(pred, label_batch, reduction="mean")
                total_val_mse += l_mse.item() * in_batch.size(0)
                count_val += in_batch.size(0)
        avg_val_mse = total_val_mse / count_val

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]
        log_message(
            f"Epoch {epoch+1}/{num_epochs} => "
            f"SNRbal Loss={avg_train_loss:.6f}, Raw Train MSE={avg_train_mse:.6f}, "
            f"Val MSE={avg_val_mse:.6f}, lr={current_lr:.3e}, elapsed {elapsed:.1f}s"
        )

        if avg_val_mse < best_val:
            best_val = avg_val_mse
            best_state = copy.deepcopy(model.state_dict())
            torch.save({
                "state_dict": best_state, "val_mse": avg_val_mse, "epoch": epoch,
                "model_type": "chanestnet", "kernel_size": kernel_size,
                "hidden_dim": hidden_dim, "M": M, "N_tags": N_tags,
                "K": K, "T": T, "P": P, "mod_type": mod_type,
                "att_db": att_dB, "offline_snr_dbs": offline_snr_dbs,
                "val_snr_db": val_SNR_dB,
            }, checkpoint_path)
            log_message(f"  -> checkpoint updated: {checkpoint_path.name}")

        if scheduler is not None:
            scheduler.step()

    total_time = time.time() - start_time
    log_message(f"\nTraining complete. Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    if best_state is not None:
        model.load_state_dict(best_state)
        log_message(f"Loaded best checkpoint with val_mse={best_val:.6f}")
    return model


def parse_hv(pred_out, N_tags, M):
    x = pred_out.reshape(N_tags + 1, 2 * M)
    h_cplx = x[0, :M] + 1j * x[0, M:]
    v_list = [x[i, :M] + 1j * x[i, M:] for i in range(1, N_tags + 1)]
    return h_cplx, v_list


def ls_estimate_from_corr(sub_corr, N_tags):
    return sub_corr[0], [sub_corr[i + 1] - sub_corr[0] for i in range(N_tags)]


def build_w_all(h_est, v_est_list):
    N = len(v_est_list); num_hyp = 2 ** N; M = h_est.shape[0]
    w_all = np.zeros((num_hyp, M), dtype=np.complex64)
    for j in range(num_hyp):
        bits_str = np.binary_repr(j, width=N)
        wj = h_est.copy()
        for i, bv in enumerate(bits_str):
            if bv == "1": wj += v_est_list[i]
        w_all[j] = wj
    return w_all


def sample_data_symbols(M, N_tags, data_len, h, v_list, noise_var):
    data_bits = np.random.randint(0, 2, size=(data_len, N_tags))                      # (T, N)
    if N_tags > 0:
        V = np.stack(v_list, axis=0).astype(np.complex64)                              # (N, M)
        W_all = (h[None, :] + data_bits.astype(np.complex64) @ V).astype(np.complex64) # (T, M)
    else:
        W_all = np.broadcast_to(h[None, :], (data_len, M)).astype(np.complex64).copy()
    S_data = sample_source_symbols_batch(mod_type, K, data_len)                        # (T, K)
    noise = ((np.random.randn(data_len, M, K) + 1j * np.random.randn(data_len, M, K))
             / np.sqrt(2) * np.sqrt(noise_var)).astype(np.complex64)                    # (T, M, K)
    X_data = (W_all[:, :, None] * S_data[:, None, :] + noise).astype(np.complex64)     # (T, M, K)
    return X_data, S_data, data_bits


def lrt_detection(X_data, S_data, w_hat, noise_var, N_tags):
    preds = np.einsum("hm,dk->dhmk", w_hat, S_data)
    diff = X_data[:, None, :, :] - preds
    ll = -np.sum(np.abs(diff) ** 2, axis=(2, 3)) / noise_var
    best_j = np.argmax(ll, axis=1)
    shifts = np.arange(N_tags - 1, -1, -1, dtype=np.int64)
    return ((best_j[:, None] >> shifts) & 1).astype(int)


def generate_pilot_corr_online(h, v_list, M, N_tags, K, P, noise_var):
    num_configs = N_tags + 1
    W_eff = np.stack([h] + [h + v for v in v_list], axis=0).astype(np.complex64)      # (num_configs, M)
    num_repetitions = P // num_configs
    remainder = P % num_configs
    config_assignment = np.concatenate([
        np.tile(np.arange(num_configs), num_repetitions),
        np.arange(remainder),
    ]).astype(np.int64)                                                                 # (P,)
    S = sample_source_symbols_batch(mod_type, K, P)                                     # (P, K)
    noise = ((np.random.randn(P, M, K) + 1j * np.random.randn(P, M, K)) / np.sqrt(2)
             * np.sqrt(noise_var)).astype(np.complex64)                                  # (P, M, K)
    W_per_slot = W_eff[config_assignment]                                               # (P, M)
    wave = W_per_slot[:, :, None] * S[:, None, :] + noise                              # (P, M, K)
    num = np.einsum('pmk,pk->pm', wave, np.conjugate(S))                                # (P, M)
    denom = np.sum(np.abs(S) ** 2, axis=1)                                              # (P,)
    corr_per_slot = (num / denom[:, None]).astype(np.complex64)                          # (P, M)
    counts = np.bincount(config_assignment, minlength=num_configs).astype(np.float32)
    sub_corr = np.zeros((num_configs, M), dtype=np.complex64)
    np.add.at(sub_corr, config_assignment, corr_per_slot)
    sub_corr = (sub_corr / counts[:, None]).astype(np.complex64)
    sub_corr_ri = np.stack([sub_corr.real, sub_corr.imag], axis=0).astype(np.float32)
    return sub_corr, sub_corr_ri


def compute_lmmse_matrix(N_tags, alpha_sq, noise_var, K, R_per_config):
    dim = N_tags + 1
    A = np.eye(dim); A[:, 0] = 1.0
    Sigma_theta = np.diag(np.array([1.0] + [alpha_sq] * N_tags))
    sigma_n_sq = np.array([noise_var / (K * R_per_config[i]) for i in range(dim)])
    C = A @ Sigma_theta @ A.T + np.diag(sigma_n_sq)
    return Sigma_theta @ A.T @ np.linalg.inv(C)


def lmmse_estimate(sub_corr, W):
    M_loc = sub_corr.shape[1]; N = sub_corr.shape[0] - 1
    theta = np.zeros_like(sub_corr)
    for m in range(M_loc):
        theta[:, m] = (W @ sub_corr[:, m].real) + 1j * (W @ sub_corr[:, m].imag)
    return theta[0], [theta[i + 1] for i in range(N)]


def channel_mse_components(h_est, v_est, h_true, v_true):
    h_err = np.concatenate([h_est.real - h_true.real, h_est.imag - h_true.imag])
    h_mse = np.mean(h_err ** 2)
    v_errs = [np.concatenate([ve.real - vt.real, ve.imag - vt.imag])
              for ve, vt in zip(v_est, v_true)]
    v_mse = np.mean(np.concatenate(v_errs) ** 2) if v_errs else 0.0
    total = np.mean(np.concatenate([h_err] + v_errs) ** 2)
    return h_mse, v_mse, total


def main():
    global log_path
    log_path = checkpoint_path.with_suffix(".txt")
    log_path.write_text("")

    log_message("==== ChanEstNet Training Configuration ====")
    log_message(f"M={M}, N_tags={N_tags}, K={K}, T={T}, P={P}, mod_type={mod_type}")
    log_message(f"att_dB={att_dB}, att_factor={att_factor:.6f}")
    log_message(f"offline_snr_dbs={offline_snr_dbs}, val_SNR_dB={val_SNR_dB}")
    log_message(f"hidden_dim={hidden_dim}, kernel_size={kernel_size}")
    log_message(f"num_train_frames={num_train_frames}, batch_size={batch_size}, "
                f"num_epochs={num_epochs}, lr={learning_rate}")
    log_message(f"Loss: SNR-balanced MSE (per-sample MSE / sigma_n^2)")
    log_message(f"Device: {device}\n")

    train_data = CorrCNNChannelDatasetWithNoise(
        num_train_frames, M, N_tags, K, P, offline_noise_vars, att_factor,
        target_type=target_type)
    val_data = CorrCNNChannelDatasetWithNoise(
        num_val_frames, M, N_tags, K, P, val_noise_var, att_factor,
        target_type=target_type, precompute=True)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            drop_last=False, num_workers=0, pin_memory=True)

    model = build_model(M, N_tags, hidden_dim, kernel_size).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    log_message(f"Model: ChanEstNet (k={kernel_size}, h={hidden_dim}) | "
                f"Parameters: {num_params:,}")
    log_message(f"Architecture:\n{model}\n")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = None
    if lr_decay_every > 0 and lr_decay_gamma < 1.0:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_decay_every, gamma=lr_decay_gamma)

    log_message("Starting training...")
    model = train_model(model, optimizer, scheduler, train_loader, val_loader,
                        num_epochs, checkpoint_path)
    log_message("")

    alpha_sq = 10 ** (att_dB / 10)
    num_configs = N_tags + 1
    R_base = P // num_configs; remainder_p = P % num_configs
    R_per_config = np.array([R_base + (1 if i < remainder_p else 0)
                             for i in range(num_configs)])

    net_label = "ChanEstNet"

    log_message("Starting online evaluation...")
    eval_start_time = time.time()

    ber_results = {"Oracle": {}, "LS": {}, "LMMSE(opt)": {}, net_label: {}}
    mse_results = {"LS": {}, "LMMSE(opt)": {}, net_label: {}}

    for snr_db in snr_dbs:
        snr_lin = 10 ** (snr_db / 10)
        noise_var_online = 1.0 / snr_lin
        W_matched = compute_lmmse_matrix(N_tags, alpha_sq, noise_var_online, K, R_per_config)
        tot_err = {"Oracle": 0, "LS": 0, "LMMSE(opt)": 0, net_label: 0}
        tot_mse = {"LS": 0.0, "LMMSE(opt)": 0.0, net_label: 0.0}
        tot_bits = 0; n_frames = 0

        model.eval()
        with torch.no_grad():
            for _ in range(num_online_frames):
                h, g_list = generate_random_channels(M, N_tags)
                v_list = [att_factor * g for g in g_list]

                sub_corr, sub_corr_ri = generate_pilot_corr_online(
                    h, v_list, M, N_tags, K, P, noise_var_online)

                h_ls, v_ls_list = ls_estimate_from_corr(sub_corr, N_tags)
                h_lmmse, v_lmmse_list = lmmse_estimate(sub_corr, W_matched)

                sub_in_t = torch.tensor(sub_corr_ri, dtype=torch.float32,
                                        device=device).unsqueeze(0)
                pred = model(sub_in_t)[0].cpu().numpy()
                h_net, v_net_list = parse_hv(pred, N_tags, M)

                for name, (he, ve) in [("LS", (h_ls, v_ls_list)),
                                       ("LMMSE(opt)", (h_lmmse, v_lmmse_list)),
                                       (net_label, (h_net, v_net_list))]:
                    _, _, tm = channel_mse_components(he, ve, h, v_list)
                    tot_mse[name] += tm

                w_oracle = build_w_all(h, v_list)
                w_ls = build_w_all(h_ls, v_ls_list)
                w_lmmse = build_w_all(h_lmmse, v_lmmse_list)
                w_net = build_w_all(h_net, v_net_list)

                X_data, S_data, data_bits = sample_data_symbols(
                    M, N_tags, data_len, h, v_list, noise_var_online)
                tot_bits += data_bits.size

                for name, w_hat in [("Oracle", w_oracle), ("LS", w_ls),
                                    ("LMMSE(opt)", w_lmmse), (net_label, w_net)]:
                    pred_bits = lrt_detection(X_data, S_data, w_hat, noise_var_online, N_tags)
                    tot_err[name] += int(np.sum(np.abs(pred_bits - data_bits)))
                n_frames += 1

        for name in ber_results:
            ber_results[name][snr_db] = tot_err[name] / tot_bits
        for name in mse_results:
            mse_results[name][snr_db] = tot_mse[name] / n_frames
        log_message(f"  SNR={snr_db:>5.1f} dB done")

    eval_time = time.time() - eval_start_time
    log_message(f"\nEvaluation complete. Time: {eval_time:.1f}s ({eval_time/60:.1f} min)")

    keys = ["Oracle", "LS", "LMMSE(opt)", net_label]
    log_message("\n==== Detection BER (coherent LRT) ====")
    hdr = f"{'SNR(dB)':>8}" + "".join(f" | {k:>12}" for k in keys)
    log_message(hdr); log_message("-" * len(hdr))
    for snr_db in snr_dbs:
        log_message(f"{snr_db:>8.1f}" + "".join(
            f" | {ber_results[k][snr_db]:>12.4f}" for k in keys))

    keys = ["LS", "LMMSE(opt)", net_label]
    log_message("\n==== Channel Estimation MSE (total) ====")
    hdr = f"{'SNR(dB)':>8}" + "".join(f" | {k:>12}" for k in keys)
    log_message(hdr); log_message("-" * len(hdr))
    for snr_db in snr_dbs:
        log_message(f"{snr_db:>8.1f}" + "".join(
            f" | {mse_results[k][snr_db]:>12.6f}" for k in keys))


if __name__ == "__main__":
    main()
