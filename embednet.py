import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy
from pathlib import Path

parser = argparse.ArgumentParser(description="EmbedNet training and BER evaluation for multi-tag ambient backscatter detection.")
parser.add_argument("--M", type=int, default=4, help="Number of reader antennas.")
parser.add_argument("--N_tags", type=int, default=2, help="Number of passive tags.")
parser.add_argument("--K", type=int, default=20, help="RF source symbols per tag symbol period.")
parser.add_argument("--T", type=int, default=160, help="Total tag symbol periods per frame.")
parser.add_argument("--P", type=int, default=32, help="Number of pilot symbol periods per frame.")
parser.add_argument("--mod_type", type=str, default="qpsk", choices=["cscg", "qpsk", "16qam"], help="Ambient modulation type.")
parser.add_argument("--num_meta_iterations", type=int, default=2000, help="Number of meta-training iterations.")
parser.add_argument("--meta_batch_size", type=int, default=128, help="Meta batch size.")
parser.add_argument("--meta_lr", type=float, default=1e-3, help="Meta learning rate.")
parser.add_argument("--meta_val_interval", type=int, default=50, help="Iterations between validations.")
parser.add_argument("--num_val_episodes", type=int, default=512, help="Number of validation episodes per run.")
parser.add_argument("--lr_decay_every", type=int, default=500, help="StepLR interval in meta-iterations. <=0 disables decay.")
parser.add_argument("--lr_decay_gamma", type=float, default=0.5, help="StepLR decay factor.")
parser.add_argument("--train_snr_db", type=float, default=20.0, help="Fixed training SNR (dB).")
parser.add_argument("--val_snr_db", type=float, default=20.0, help="Validation SNR (dB) for checkpoint selection.")
parser.add_argument("--att_db", type=float, default=-20.0, help="Attenuation (dB).")
parser.add_argument("--num_online_frames", type=int, default=10000, help="Frames per SNR for evaluation.")
parser.add_argument("--eval_snr_db", type=str, default="0,4,8,12,16,20", help="Comma-separated SNRs (dB) for post-training BER evaluation.")
parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to store checkpoints and logs.")
args = parser.parse_args()

M = args.M
N_tags = args.N_tags
K = args.K
T = args.T
P = args.P
mod_type = args.mod_type
num_meta_iterations = args.num_meta_iterations
meta_batch_size = args.meta_batch_size
meta_lr = args.meta_lr
meta_val_interval = args.meta_val_interval
num_val_episodes = args.num_val_episodes
lr_decay_every = args.lr_decay_every
lr_decay_gamma = args.lr_decay_gamma
train_SNR_dB = args.train_snr_db
train_noise_var = 1.0 / (10 ** (train_SNR_dB / 10))
att_dB = args.att_db
att_factor = np.sqrt(10 ** (att_dB / 10))
num_online_frames = args.num_online_frames
eval_snr_dbs = [float(s.strip()) for s in args.eval_snr_db.split(",") if s.strip()]
val_SNR_dB = args.val_snr_db
val_noise_var = 10 ** (-val_SNR_dB / 10.0)
checkpoint_dir = Path(args.output_dir)
checkpoint_dir.mkdir(parents=True, exist_ok=True)


def float_tag(value):
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.6g}".replace("-", "m").replace(".", "p")


checkpoint_suffix = (
    f"embednet_2block_M{M}_N{N_tags}_K{K}_T{T}_P{P}_mod-{mod_type}_"
    f"trainSNR{float_tag(train_SNR_dB)}_valSNR{float_tag(val_SNR_dB)}_"
    f"att{float_tag(att_dB)}dB.pt"
)
checkpoint_path = checkpoint_dir / checkpoint_suffix

log_path = checkpoint_path.with_suffix(".txt")
log_path.write_text("")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def log_message(message):
    print(message)
    with log_path.open("a") as f:
        f.write(message + "\n")


log_message("==== Training Configuration ====")
log_message(f"M={M}, N_tags={N_tags}, K={K}, T={T}, P={P}, mod_type={mod_type}")
log_message(f"train_snr_db={train_SNR_dB}, val_snr_db={val_SNR_dB}, att_dB={att_dB}")
log_message(f"num_meta_iterations={num_meta_iterations}, meta_batch_size={meta_batch_size}, meta_lr={meta_lr}")
log_message(f"num_val_episodes={num_val_episodes}, meta_val_interval={meta_val_interval}")
log_message(f"lr_decay_every={lr_decay_every}, lr_decay_gamma={lr_decay_gamma}")
log_message(f"num_online_frames={num_online_frames}, eval_snr_db={eval_snr_dbs}")
log_message(f"Encoder: 2-block Conv2d (2->32->64) + LayerNorm + ReLU, AdaptivePool(4x4), Linear(1024->64)")
log_message(f"Decision rule: prototype mean + squared Euclidean distance")
log_message(f"Device: {device}")


def bits_to_class(bit_array):
    val = 0
    for b in bit_array:
        val = (val << 1) | b
    return val


def sample_source_symbols(mod_type, K):
    if mod_type.lower() == "cscg":
        real_part = np.random.randn(K) / np.sqrt(2)
        imag_part = np.random.randn(K) / np.sqrt(2)
        s = real_part + 1j * imag_part
    elif mod_type.lower() == "qpsk":
        constellation = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
        s = np.random.choice(constellation, size=K)
    elif mod_type.lower() == "16qam":
        re = np.array([-3, -1, 1, 3]) / np.sqrt(10)
        im = np.array([-3, -1, 1, 3]) / np.sqrt(10)
        points = []
        for r in re:
            for i in im:
                points.append(r + 1j * i)
        points = np.array(points)
        s = np.random.choice(points, size=K)
    else:
        raise ValueError("Unsupported mod_type: choose from {cscg, qpsk, 16qam}")
    return s


def sample_source_symbols_batch(mod_type, K, R):
    mt = mod_type.lower()
    if mt == "cscg":
        return ((np.random.randn(R, K) + 1j * np.random.randn(R, K)) / np.sqrt(2)).astype(np.complex64)
    if mt == "qpsk":
        c = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
        return np.random.choice(c, size=(R, K)).astype(np.complex64)
    if mt == "16qam":
        vals = np.array([-3, -1, 1, 3]) / np.sqrt(10)
        pts = np.array([r + 1j * i for r in vals for i in vals])
        return np.random.choice(pts, size=(R, K)).astype(np.complex64)
    raise ValueError("Unsupported mod_type: choose from {cscg, qpsk, 16qam}")


def generate_frame(M, N_tags, K, noise_var, att_factor=1.0, pilots=False, pilot_bits=None):
    h = (np.random.randn(M) + 1j * np.random.randn(M)) / np.sqrt(2)
    g_arr = ((np.random.randn(N_tags, M) + 1j * np.random.randn(N_tags, M)) / np.sqrt(2))  # (N, M)

    bits_all = np.random.randint(0, 2, size=(T, N_tags)).astype(np.int64)
    if pilots and pilot_bits is not None:
        bits_all[:P] = pilot_bits[:P]

    shifts = np.arange(N_tags - 1, -1, -1, dtype=np.int64)
    frame_labels = np.sum(bits_all * (1 << shifts), axis=1).astype(np.int64)

    W_all = (h[None, :] + att_factor * (bits_all.astype(np.complex128) @ g_arr)).astype(np.complex128)  # (T, M)

    S_all = sample_source_symbols_batch(mod_type, K, T)                                    # (T, K)
    noise = ((np.random.randn(T, M, K) + 1j * np.random.randn(T, M, K))
             / np.sqrt(2) * np.sqrt(noise_var))                                             # (T, M, K)
    X_all = W_all[:, :, None] * S_all[:, None, :] + noise                                   # (T, M, K)

    R_all = np.einsum('tmk,tnk->tmn', X_all, X_all.conj()) / K                              # (T, M, M)

    frame_samples = np.stack(
        [R_all.real.astype(np.float32), R_all.imag.astype(np.float32)], axis=1)              # (T, 2, M, M)

    return frame_samples, frame_labels


def generate_frames_batched(B, M, N_tags, K, noise_var, att_factor=1.0, pilots=False, pilot_bits=None):
    h = (np.random.randn(B, M) + 1j * np.random.randn(B, M)) / np.sqrt(2)                   # (B, M)
    g_arr = ((np.random.randn(B, N_tags, M) + 1j * np.random.randn(B, N_tags, M))
             / np.sqrt(2))                                                                  # (B, N, M)

    bits_all = np.random.randint(0, 2, size=(B, T, N_tags)).astype(np.int64)
    if pilots and pilot_bits is not None:
        bits_all[:, :P] = pilot_bits[None, :P]

    shifts = np.arange(N_tags - 1, -1, -1, dtype=np.int64)
    frame_labels = np.sum(bits_all * (1 << shifts), axis=2).astype(np.int64)                # (B, T)

    W_all = h[:, None, :] + att_factor * np.einsum(
        'btn,bnm->btm',
        bits_all.astype(np.complex128),
        g_arr.astype(np.complex128),
    )                                                                                        # (B, T, M)

    S_all = sample_source_symbols_batch(mod_type, K, B * T).reshape(B, T, K)                 # (B, T, K)
    nv = np.asarray(noise_var, dtype=np.float64)
    if nv.ndim == 0:
        nv_sqrt = np.sqrt(float(nv))
    else:
        nv_sqrt = np.sqrt(nv).reshape(-1, 1, 1, 1)                                           # (B, 1, 1, 1)
    noise = ((np.random.randn(B, T, M, K) + 1j * np.random.randn(B, T, M, K))
             / np.sqrt(2)) * nv_sqrt                                                         # (B, T, M, K)
    X_all = W_all[:, :, :, None] * S_all[:, :, None, :] + noise                              # (B, T, M, K)

    R_all = np.einsum('btmk,btnk->btmn', X_all, X_all.conj()) / K                            # (B, T, M, M)

    frame_samples = np.stack(
        [R_all.real.astype(np.float32), R_all.imag.astype(np.float32)], axis=2)              # (B, T, 2, M, M)

    return frame_samples, frame_labels


def get_fixed_pilot_bits(N_tags, P):
    n_classes = 2 ** N_tags
    combos = []
    for c in range(n_classes):
        bits_str = np.binary_repr(c, width=N_tags)
        bits = [int(b) for b in bits_str]
        combos.append(bits)
    combos = np.array(combos, dtype=np.int64)

    repeats = P // n_classes
    remainder = P % n_classes
    pilot_list = []
    for row in combos:
        for _ in range(repeats):
            pilot_list.append(row)
    idx = 0
    while len(pilot_list) < P:
        pilot_list.append(combos[idx])
        idx += 1

    pilot_array = np.array(pilot_list[:P], dtype=np.int64)
    return pilot_array


pilot_bits = get_fixed_pilot_bits(N_tags, P)


def classes_to_bits(class_tensor, n_bits):
    class_np = class_tensor.detach().cpu().numpy().astype(np.int64)
    shifts = np.arange(n_bits - 1, -1, -1, dtype=np.int64)
    return ((class_np[:, None] >> shifts) & 1).astype(np.int64)


def compute_bit_errors(pred_classes, true_classes, n_bits):
    pred_bits = classes_to_bits(pred_classes, n_bits)
    true_bits = classes_to_bits(true_classes, n_bits)
    bit_diff = np.abs(pred_bits - true_bits)
    return int(bit_diff.sum()), int(bit_diff.size)


class EmbedNet(nn.Module):
    def __init__(self, input_shape, out_dim=64, pool_size=(4, 4)):
        super().__init__()
        H, W = input_shape  # (M, M)

        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm([32, H, W])
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.ln2 = nn.LayerNorm([64, H, W])
        self.adaptive_pool = nn.AdaptiveAvgPool2d(pool_size)

        dummy = torch.zeros(1, 2, *input_shape)
        x = F.relu(self.ln1(self.conv1(dummy)))
        x = F.relu(self.ln2(self.conv2(x)))
        x = self.adaptive_pool(x)
        feat_dim = x.numel()

        self.fc = nn.Linear(feat_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.ln1(self.conv1(x)))
        x = F.relu(self.ln2(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def _proto_logits(emb_q, proto):
    qq = (emb_q ** 2).sum(-1, keepdim=True)         # (B, Q, 1)
    pp = (proto ** 2).sum(-1).unsqueeze(1)          # (B, 1, C)
    qp = torch.einsum('bqd,bcd->bqc', emb_q, proto)
    return -(qq + pp - 2 * qp)


def _build_prototypes(emb_s, support_y, n_classes):
    one_hot = F.one_hot(support_y, num_classes=n_classes).to(emb_s.dtype)   # (P, C)
    counts = one_hot.sum(dim=0).clamp(min=1)                                 # (C,)
    return (one_hot.T @ emb_s) / counts.unsqueeze(1)                         # (C, D)


def embednet_batched_distances(emb_net, fs, n_classes, P):
    B, T_, C_in, H, W = fs.shape
    emb = emb_net(fs.view(B * T_, C_in, H, W)).view(B, T_, -1)
    emb_s = emb[:, :P]
    emb_q = emb[:, P:]
    return emb_s, emb_q


def embednet_batched_loss(emb_net, fs, fl, n_classes, P):
    emb_s, emb_q = embednet_batched_distances(emb_net, fs, n_classes, P)
    y_s = fl[:, :P]
    y_q = fl[:, P:]

    one_hot = F.one_hot(y_s, num_classes=n_classes).to(emb_s.dtype)          # (B, P, C)
    counts = one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)                    # (B, C, 1)
    prototypes = torch.einsum('bpc,bpd->bcd', one_hot, emb_s) / counts        # (B, C, D)

    logits = _proto_logits(emb_q, prototypes)                        # (B, Q, C)
    B, Q, C = logits.shape
    return F.cross_entropy(logits.reshape(B * Q, C), y_q.reshape(B * Q)), logits


def embednet_batched_predict(emb_net, fs, fl, n_classes, P):
    emb_s, emb_q = embednet_batched_distances(emb_net, fs, n_classes, P)
    y_s = fl[:, :P]
    y_q = fl[:, P:]

    one_hot = F.one_hot(y_s, num_classes=n_classes).to(emb_s.dtype)
    counts = one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
    prototypes = torch.einsum('bpc,bpd->bcd', one_hot, emb_s) / counts

    logits = _proto_logits(emb_q, prototypes)
    pred = torch.argmax(logits, dim=2)
    return pred, y_q


n_classes = 2 ** N_tags
emb_net = EmbedNet(input_shape=(M, M), out_dim=64).to(device)
optimizer = optim.Adam(emb_net.parameters(), lr=meta_lr)
scheduler = None
if lr_decay_every > 0 and lr_decay_gamma < 1.0:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_every, gamma=lr_decay_gamma)


log_message("Starting EmbedNet training with checkpointing...")
log_message(f"Generating {num_val_episodes} fixed validation episodes at SNR={val_SNR_dB} dB...")
val_fs_np, val_fl_np = generate_frames_batched(
    num_val_episodes, M, N_tags, K, val_noise_var, att_factor=att_factor,
    pilots=True, pilot_bits=pilot_bits,
)
val_fs = torch.from_numpy(val_fs_np).to(device)
val_fl = torch.from_numpy(val_fl_np).to(device)
val_chunk = meta_batch_size

best_val_loss = float("inf")
best_val_ber = float("inf")
best_state = copy.deepcopy(emb_net.state_dict())
start_time = time.time()

for meta_iter in range(num_meta_iterations):
    emb_net.train()
    optimizer.zero_grad(set_to_none=True)

    frame_samples_np, frame_labels_np = generate_frames_batched(
        meta_batch_size, M, N_tags, K, train_noise_var,
        att_factor=att_factor, pilots=True, pilot_bits=pilot_bits,
    )
    fs_t = torch.from_numpy(frame_samples_np).to(device, non_blocking=True)
    fl_t = torch.from_numpy(frame_labels_np).to(device, non_blocking=True)

    loss, _ = embednet_batched_loss(emb_net, fs_t, fl_t, n_classes, P)
    loss_value = loss.item()
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    if meta_iter % meta_val_interval == 0:
        emb_net.eval()
        with torch.no_grad():
            val_total_loss_sum = 0.0
            val_total_bits = 0
            val_total_bit_errors = 0
            for start in range(0, num_val_episodes, val_chunk):
                end = min(start + val_chunk, num_val_episodes)
                fs_chunk = val_fs[start:end]
                fl_chunk = val_fl[start:end]
                chunk_loss, _ = embednet_batched_loss(emb_net, fs_chunk, fl_chunk, n_classes, P)
                val_total_loss_sum += chunk_loss.item() * (end - start)
                pred, y_q = embednet_batched_predict(emb_net, fs_chunk, fl_chunk, n_classes, P)
                pred_bits = classes_to_bits(pred.reshape(-1), N_tags)
                true_bits = classes_to_bits(y_q.reshape(-1), N_tags)
                val_total_bit_errors += int(np.abs(pred_bits - true_bits).sum())
                val_total_bits += int(pred_bits.size)
            val_loss_mean = val_total_loss_sum / num_val_episodes
            val_ber = val_total_bit_errors / val_total_bits
        emb_net.train()

        if (val_ber < best_val_ber) or (abs(val_ber - best_val_ber) <= 1e-12 and val_loss_mean < best_val_loss):
            best_val_loss = val_loss_mean
            best_val_ber = val_ber
            best_state = copy.deepcopy(emb_net.state_dict())
            torch.save(
                {
                    "state_dict": best_state,
                    "train_loss": loss_value,
                    "val_loss": best_val_loss,
                    "val_ber": best_val_ber,
                    "iteration": meta_iter,
                    "val_snr_db": val_SNR_dB,
                    "train_snr_db": train_SNR_dB,
                    "lr_decay_every": lr_decay_every,
                    "lr_decay_gamma": lr_decay_gamma,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "M": M,
                    "N_tags": N_tags,
                    "K": K,
                    "T": T,
                    "P": P,
                    "mod_type": mod_type,
                    "att_db": att_dB,
                    "normalization": "layernorm",
                },
                checkpoint_path,
            )
            log_message(f"  -> checkpoint updated: {checkpoint_path.name}")

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]
        log_message(
            f"Iter {meta_iter}, train loss = {loss_value:.4f}, val loss = {val_loss_mean:.4f}, "
            f"val BER = {val_ber:.6f}, lr = {current_lr:.3e}, elapsed {elapsed:.1f}s"
        )

log_message("Training complete.")

if best_state is not None:
    emb_net.load_state_dict(best_state)
    log_message(f"Loaded best checkpoint with validation BER {best_val_ber:.6f} and validation loss {best_val_loss:.4f}")


log_message("Starting online evaluation over SNRs...")
ber_results = {}
emb_net.eval()
eval_chunk = meta_batch_size
with torch.no_grad():
    for snr_db in eval_snr_dbs:
        SNR_lin = 10 ** (snr_db / 10)
        noise_var_online = 1.0 / SNR_lin

        total_test_bits = 0
        total_bit_errors = 0

        remaining = num_online_frames
        while remaining > 0:
            B_eval = min(eval_chunk, remaining)
            fs_np, fl_np = generate_frames_batched(
                B_eval, M, N_tags, K, noise_var_online,
                att_factor=att_factor, pilots=True, pilot_bits=pilot_bits,
            )
            fs_t = torch.from_numpy(fs_np).to(device)
            fl_t = torch.from_numpy(fl_np).to(device)

            pred, y_q = embednet_batched_predict(emb_net, fs_t, fl_t, n_classes, P)
            pred_bits = classes_to_bits(pred.reshape(-1), N_tags)
            true_bits = classes_to_bits(y_q.reshape(-1), N_tags)
            total_bit_errors += int(np.abs(pred_bits - true_bits).sum())
            total_test_bits += int(pred_bits.size)
            remaining -= B_eval

        BER = total_bit_errors / total_test_bits
        ber_results[snr_db] = BER
        log_message(f"SNR = {snr_db} dB => BER = {BER:.4f}")

log_message("\nFinal BER results:")
for snr_db in sorted(ber_results.keys()):
    log_message(f"SNR={snr_db} dB: BER={ber_results[snr_db]:.4f}")
