# Deep Learning-Enabled Multi-Tag Detection in Ambient Backscatter Communications

Code for the paper "Deep Learning-Enabled Multi-Tag Detection in Ambient
Backscatter Communications" (IEEE Transactions on Wireless Communications,
2026) by Talha Akyıldız and Hessam Mahdavifar.

The code implements the two proposed multi-tag detectors. EmbedNet maps sample covariance features to joint tag states through pilot-derived prototypes. ChanEstNet estimates effective channel coefficients from one-hot pilot correlations and supplies them to a multi-hypothesis likelihood ratio test.

## Citation

```bibtex
@article{akyildiz2026deep,
  title   = {Deep Learning-Enabled Multi-Tag Detection in Ambient Backscatter Communications},
  author  = {Aky{\i}ld{\i}z, Talha and Mahdavifar, Hessam},
  journal = {IEEE Transactions on Wireless Communications},
  year    = {2026},
  doi     = {10.1109/TWC.2026.3734021}
}
```

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

Tested with Python 3.13, PyTorch 2.9, and NumPy 2.1. CPU, CUDA, and Apple
Silicon execution are supported by PyTorch where available.

## Layout

```
embednet.py        EmbedNet training and BER evaluation
chanestnet.py      ChanEstNet training, channel estimation, and LRT evaluation
requirements.txt   Python dependencies
```

## Paper configuration

M = 4 antennas, K = 20 samples per tag symbol, T = 160 symbol periods per frame,
P = 32 pilots, attenuation -20 dB, QPSK ambient source (`--mod_type cscg` for
Gaussian), BER evaluation over 0–20 dB with 10000 frames per SNR.

## Running the experiments

ChanEstNet (mixed-SNR training over {0,4,8,12,16,20} dB, 100 epochs, Adam 1e-3):

```
python chanestnet.py --mod_type qpsk --M 4 --N_tags 2 --K 20 --T 160 --P 32 \
    --att_db -20 --num_epochs 100 --train_snr_db 0,4,8,12,16,20 --val_snr_db 10 \
    --eval_snr_db 0,4,8,12,16,20 --output_dir runs/chanestnet_N2
```

EmbedNet (fixed 20 dB training SNR, 2000 meta-iterations, 128 episodes each, Adam 1e-3):

```
python embednet.py --mod_type qpsk --M 4 --N_tags 2 --K 20 --T 160 --P 32 \
    --att_db -20 --train_snr_db 20 --val_snr_db 20 --eval_snr_db "0,4,8,12,16,20" \
    --output_dir runs/embednet_N2
```

Both scripts default to two tags. Change `--N_tags` for other tag counts and
use `--mod_type cscg` for the Gaussian ambient source.

## Output

Each run trains the network, keeps the best validation checkpoint, and then
evaluates BER over the SNRs in `--eval_snr_db`. The checkpoint (`.pt`) and a
log file with the configuration and final BER values are written to
`--output_dir`.

Because channel, source, and noise realizations are generated randomly,
individual runs can produce slightly different numerical results.

## License

MIT — see [LICENSE](LICENSE).
