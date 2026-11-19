"""Verbatim matched-IC statistics from downstream_slotlam/causal_extended/evaluate.py.
Original source SHA256: 7f94ad286c91a9c4b2b05bfcbc719f3181dbf74b563f473d9a60cc3ac8c0805d
"""
import numpy as np

KINDS = ["identity", "camera", "photo", "real"]


def distances(frame, z):
    order = {k: frame.index[frame.kind == k].to_numpy() for k in KINDS}
    order = {k: frame.loc[order[k]].sort_values('pair_id').index.to_numpy() for k in KINDS}
    zi, zc, zp, zr = [z[order[k]] for k in KINDS]
    standard_deviation = zr.std(0)
    sd = standard_deviation + 1e-6
    d = {k: np.linalg.norm((zz - zi) / sd, axis=1) for k, zz in [('camera', zc), ('photo', zp), ('real', zr)]}
    raw_d = {k: np.linalg.norm(zz - zi, axis=1) / np.sqrt(z.shape[1]) for k, zz in [('camera', zc), ('photo', zp), ('real', zr)]}
    table = frame.loc[order['real'], ['pair_id', 'subset', 'episode_id']].reset_index(drop=True)
    for k in d:
        table['d_' + k] = d[k]
        table['raw_rms_' + k] = raw_d[k]
    diagnostics = dict(dim=int(z.shape[1]),
        real_latent_mean_coordinate_std=float(standard_deviation.mean()),
        fraction_real_coordinates_std_below_1e_6=float(np.mean(standard_deviation < 1e-6)),
        fraction_real_response_zero=float(np.mean(raw_d['real'] < 1e-10)),
        fraction_camera_response_zero=float(np.mean(raw_d['camera'] < 1e-10)),
        median_standardized_rms={k: float(np.median(v) / np.sqrt(z.shape[1])) for k, v in d.items()},
        median_raw_rms={k: float(np.median(v)) for k, v in raw_d.items()},
        unique_real_vectors=int(len(np.unique(zr, axis=0))),
        warning='Absolute latent distance/RMS is diagnostic; IC ratios and action decoding must be interpreted jointly.')
    return table, diagnostics


def bootstrap(table, draws=1000, seed=73):
    # Resample complete source episodes within each data source; all perturbations remain paired.
    rng = np.random.default_rng(seed)
    sources = []
    for _, block in table.groupby('subset', sort=True):
        sources.append([g.index.to_numpy() for _, g in block.groupby('episode_id', sort=True)])
    result = []
    d = table[['d_camera', 'd_photo', 'd_real']].to_numpy()
    for _ in range(draws):
        indices = np.concatenate([np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))]) for groups in sources])
        x = d[indices]
        median = np.median(x, axis=0)
        if median[2] <= 0:
            raise ValueError('Degenerate real response in bootstrap')
        result.append([median[0] / median[2], median[1] / median[2], np.mean(x[:, 0] > x[:, 2]), np.mean(x[:, 1] > x[:, 2])])
    keys = ['IC_camera', 'IC_photo', 'P_camera_gt_real', 'P_photo_gt_real']
    values = np.asarray(result)
    return dict(draws=draws, seed=seed, resampling='subset-stratified episode cluster bootstrap; perturbations paired',
        scale='per-coordinate real standard deviations fixed at observed sample values; conditional bootstrap',
        n_episode_groups=int(sum(len(g) for g in sources)),
        percentile95={key: np.quantile(values[:, i], [.025, .975]).tolist() for i, key in enumerate(keys)})
