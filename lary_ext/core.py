import hashlib
import json
from pathlib import Path
import numpy as np


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def identity(row):
    # Deliberately exclude model-specific la_path; preserve path components.
    fields = [str(row.get(k, '')) for k in ('src_img', 'tgt_img', 'action')]
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')


def new_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=False)
    return p


class Bundle:
    """Disk-backed arrays. No pickle, model loading, or image decoding."""
    def __init__(self, root):
        self.root = Path(root)
        self.meta = json.loads((self.root / 'manifest.json').read_text())
        if self.meta.get('format') != 'lary-ext-bundle-v1':
            raise ValueError('Unsupported bundle format')
        for name in ('z', 'target', 'valid', 'ids', 'episodes', 'groups'):
            setattr(self, name, np.load(self.root / (name + '.npy'),
                                       mmap_mode='r', allow_pickle=False))
        n = len(self.ids)
        if n==0:raise ValueError('Empty bundle')
        if (self.z.ndim != 2 or self.target.ndim != 3 or
                self.valid.shape != self.target.shape or self.valid.dtype != bool or
                any(len(getattr(self, k)) != n for k in
                    ('z', 'target', 'valid', 'episodes', 'groups'))):
            raise ValueError('Inconsistent bundle shapes/masks')
        if len(set(self.ids.tolist())) != n:
            raise ValueError('Duplicate sample IDs')
        if any(not str(e).strip() for e in self.episodes):
            raise ValueError('Missing episode ID: row-level splitting is prohibited')

    def __len__(self):
        return len(self.ids)

    def verify(self):
        def snapshot():
            return {name:[(self.root/name).stat().st_size,(self.root/name).stat().st_mtime_ns,
                          (self.root/name).stat().st_ctime_ns]
                    for name in self.meta['array_sha256']}
        before=snapshot();mh=file_hash(self.root/'manifest.json')
        ticket=self.root/'verified_sha256.json'
        if ticket.exists():
            old=json.loads(ticket.read_text())
            if old.get('manifest_sha256')==mh and old.get('file_stats')==before:
                return self
        for name, expected in self.meta['array_sha256'].items():
            if file_hash(self.root / name) != expected:
                raise ValueError('Changed bundle array: ' + name)
        if snapshot()!=before:raise ValueError('Bundle changed during verification')
        write_json(ticket,{'manifest_sha256':mh,'file_stats':before,
                           'method':'SHA256 verified; reused only while size,mtime,ctime remain unchanged'})
        return self


def create_bundle(root, z, target, episodes, groups=None, ids=None, valid=None,
                  provenance=None):
    """Small-fixture/programmatic writer; production exporter uses memmaps."""
    root = new_dir(root)
    n = len(z)
    arrays = dict(z=np.asarray(z, np.float32), target=np.asarray(target, np.float32),
                  episodes=np.asarray(episodes, str),
                  groups=np.asarray(groups if groups is not None else ['default']*n, str),
                  ids=np.asarray(ids if ids is not None else [str(i) for i in range(n)], str),
                  valid=np.asarray(valid if valid is not None else np.isfinite(target), bool))
    for k, v in arrays.items():
        np.save(root / (k+'.npy'), v, allow_pickle=False)
    finish_bundle(root, provenance or {})
    return Bundle(root)


def finish_bundle(root, provenance):
    root = Path(root)
    write_json(root / 'manifest.json', dict(format='lary-ext-bundle-v1',
        provenance=provenance,
        array_sha256={p.name:file_hash(p) for p in sorted(root.glob('*.npy'))}))


def fit_stats(bundle, indices):
    """Per-robot, per-action-coordinate moments, fitted ONLY on budget rows."""
    if len(indices) == 0:
        raise ValueError('Empty training budget')
    out = {}
    for g in sorted(set(bundle.groups[indices].tolist())):
        ix = np.asarray(indices)[bundle.groups[indices] == g]
        d = bundle.target.shape[-1]
        count, total, squares = np.zeros(d), np.zeros(d), np.zeros(d)
        for start in range(0, len(ix), 512):
            y = np.asarray(bundle.target[ix[start:start+512]], np.float64)
            v = bundle.valid[ix[start:start+512]]
            if not np.isfinite(y[v]).all():
                raise ValueError('Nonfinite valid training targets')
            y = np.where(v, y, 0)
            count += v.sum((0, 1)); total += y.sum((0, 1)); squares += (y*y).sum((0, 1))
        if (count == 0).any():
            raise ValueError('Training budget has unsupported target coordinates: '+g)
        mean = total/count
        std = np.sqrt(np.maximum(squares/count-mean*mean, 0))
        std[std < 1e-6] = 1.0  # same floor convention as upstream
        out[g] = {'mean':mean.tolist(), 'std':std.tolist(), 'count':count.tolist()}
    return out


def stats_for(stats, groups):
    missing = set(groups.tolist()) - set(stats)
    if missing:
        raise ValueError('No training-only normalization for robot groups: '+str(missing))
    return (np.asarray([stats[g]['mean'] for g in groups])[:, None, :],
            np.asarray([stats[g]['std'] for g in groups])[:, None, :])


def masked_mse(pred, target, valid):
    pred, target, valid = np.asarray(pred), np.asarray(target), np.asarray(valid, bool)
    if pred.shape != target.shape or valid.shape != target.shape:
        raise ValueError('Prediction/target/mask mismatch')
    if not np.isfinite(pred[valid]).all() or not np.isfinite(target[valid]).all():
        raise ValueError('Nonfinite prediction/target on valid coordinates')
    return float(np.mean((pred[valid]-target[valid])**2)) if valid.any() else None
