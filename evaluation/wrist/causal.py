"""Run original Causal metric definitions with ridge, IC, or the official MLP adapter."""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from .causal_data import dump, metrics, sha
from .causal_fit import official_model


class OfficialProbe:
    def __init__(self, args, serial):
        self.args, self.serial = args, serial

    def fit(self, x, y, groups):
        x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)
        episodes = np.unique(groups)
        if len(episodes) < 2:
            raise ValueError("An independent selection episode is required")
        held = np.random.default_rng(0).choice(episodes, max(1, len(episodes)//5), replace=False)
        hold = np.flatnonzero(np.isin(groups, held))
        fit = np.flatnonzero(~np.isin(groups, held))
        self.x_mu, self.x_sd = x[fit].mean(0), x[fit].std(0) + 1e-6
        self.y_mu, self.y_sd = y[fit].mean(0), y[fit].std(0) + 1e-8
        run = Path(self.args.output)
        with tempfile.TemporaryDirectory(dir=self.args.scratch) as stage:
            stage = Path(stage)
            np.save(stage/"x.npy", (x-self.x_mu)/self.x_sd)
            np.save(stage/"y.npy", (y-self.y_mu)/self.y_sd)
            np.save(stage/"fit.npy", fit)
            np.save(stage/"hold.npy", hold)
            cmd = [self.args.train_python, "-m", "torch.distributed.run", "--standalone",
                   f"--nproc_per_node={self.args.ranks}", "-m", "evaluation.wrist.causal_fit",
                   "--work", str(stage), "--source", self.args.source, "--seed", str(self.args.seed)]
            if self.args.smoke:
                cmd.append("--smoke")
            with (run / f"fit_{self.serial}.log").open("w") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            selection = json.loads((stage/"selection.json").read_text())
            selection["hold_episodes_sha256"] = hashlib.sha256(
                json.dumps(sorted(map(str, held))).encode()).hexdigest()
            dump(run/f"selection_{self.serial}.json", selection)
            self.net = official_model(self.args.source, x.shape[1], y.shape[1])
            self.net.load_state_dict(torch.load(stage/"best.pt", map_location="cpu", weights_only=True))
            self.net.eval()
        return self

    def predict(self, x):
        scaled = (np.asarray(x,np.float32)-self.x_mu)/self.x_sd
        self.net.cuda()
        with torch.inference_mode():
            pred = np.concatenate([self.net(torch.from_numpy(part).cuda()).cpu().numpy()
                                   for part in np.array_split(scaled, max(1, (len(scaled)+1023)//1024))])
        self.net.cpu()
        torch.cuda.empty_cache()
        return pred*self.y_sd+self.y_mu


def ridge_solve(x, y, alpha):
    xt, yt = [torch.as_tensor(v, dtype=torch.float64, device="cuda") for v in (x,y)]
    dual = len(x) < x.shape[1]
    g = xt @ xt.T if dual else xt.T @ xt
    g.diagonal().add_(alpha)
    return (xt.T @ torch.linalg.solve(g, yt) if dual else
            torch.linalg.solve(g, xt.T @ yt)).cpu().numpy()


def record_probe_predictions(native, output):
    """Capture the original evaluator's actual predictions, without a second forward pass."""
    original_target = native.probe_target
    serial = 0
    def traced(kind, x, y, groups, evals, dimensions):
        nonlocal serial
        serial += 1
        builder = native.make_probe
        hooks, predictions = [], []
        def capture(name):
            probe = builder(name)
            predict = probe.predict
            hooks.append((probe,predict))
            def record(features):
                pred = predict(features)
                predictions.append(pred)
                return pred
            probe.predict = record
            return probe
        native.make_probe = capture
        try:
            result, probe = original_target(kind,x,y,groups,evals,dimensions)
        finally:
            native.make_probe = builder
            for instrumented, predict in hooks:
                instrumented.predict = predict
        paths = {}
        sets = [(name,target) for name,(features,target) in evals.items() if len(features)]
        for (name,target),pred in zip(sets,predictions,strict=True):
            np.testing.assert_allclose(native.r2_score(target,pred),result[name]["all"],atol=1e-12,rtol=0)
            relative = Path("predictions") / f"{serial:02d}_{name.replace('/','_')}.npz"
            (output/relative).parent.mkdir(exist_ok=True)
            np.savez_compressed(output/relative,prediction=pred,target=target)
            paths[name]=str(relative)
        result["_paired_predictions"] = paths
        return result,probe
    native.probe_target = traced


def run(a):
    output, cohort, features = map(Path,(a.output,a.cohort,a.features))
    native = metrics(a.lary_root)
    lock = json.loads((cohort/"cohort.json").read_text())
    splits = ["ic"] if a.readout=="ic" else ["train","val","unseen"]
    weight_proof = json.loads(Path(a.weights).read_text())
    vectors = {}
    for s in splits:
        if sha(cohort/f"{s}.csv") != lock["csv_sha256"][s]:
            raise ValueError("Changed cohort CSV")
        marker = json.loads((features/f"{s}.json").read_text())
        identity = marker["identity"]
        if (identity["smoke"] or identity["model"] != a.model or identity["weights"] != weight_proof
                or identity.get("backbone_record_sha256") != "55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b"
                or identity["cohort_sha256"] != sha(cohort/"cohort.json")
                or marker["sha256"] != sha(features/f"{s}.npy")):
            raise ValueError("Features incompatible with frozen Causal cohort or checkpoint")
        vectors[s] = np.load(features/f"{s}.npy", mmap_mode="r")
    output.mkdir(parents=True,exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    rng = np.random.default_rng(13)
    for n, width in ((50,7),(7,50)):
        x = rng.normal(size=(n,width))
        y = rng.normal(size=(n,3))
        np.testing.assert_allclose(ridge_solve(x,y,.1), native.RidgeProbe._solve(x,y,.1),
                                   rtol=1e-8, atol=1e-8)
    native.RidgeProbe._solve = staticmethod(ridge_solve)
    if a.readout=="ic":
        path = Path(a.lary_root)/"causal/intervention.py"
        if sha(path) != "2fd35b330a53690ada1bd216685083564a1cbd5dcec25b0ef23855344c707cce":
            raise ValueError("IC source changed")
        spec = importlib.util.spec_from_file_location("aligned_ic",path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.RidgeProbe._solve = staticmethod(ridge_solve)
        from .ic_diagnostics import distances, bootstrap
        if sha(cohort/"ic_legacy.csv")!=lock["csv_sha256"]["ic_legacy"]:
            raise ValueError("Changed legacy IC cohort")
        frame = pd.read_csv(cohort/"ic_legacy.csv")
        mask = frame.pair_id.isin(pd.read_csv(cohort/"ic.csv").pair_id).to_numpy()
        results = {}
        for name, subset in (("matched_605", mask), ("legacy_1500", np.ones(len(frame), bool))):
            current = frame.loc[subset].reset_index(drop=True)
            z = vectors["ic"][subset]
            stats = module.analyse(current,z)
            table, diag = distances(current,z)
            for k in ("camera","photo"):
                np.testing.assert_allclose(np.median(table["d_"+k])/np.median(table.d_real),
                                           stats["IC_"+k], rtol=1e-6)
            table.to_csv(output/(name+"_distances.csv"),index=False)
            stats.update(diagnostics=diag,bootstrap=bootstrap(table))
            results[name] = stats
        result = dict(results["matched_605"], legacy_1500=results["legacy_1500"])
    else:
        def load(df, *unused, **kwargs):
            parts = df.la_path.str.split(":")
            split = parts.iloc[0][0]
            if any(p[0]!=split for p in parts):
                raise ValueError("Mixed split")
            return np.asarray(vectors[split][[int(p[1]) for p in parts]])
        native.load_latents = load
        original = native.make_probe
        serial = 0
        def make(kind):
            nonlocal serial
            if kind=="ridge":
                return original(kind)
            serial += 1
            return OfficialProbe(a,serial)
        native.make_probe = make
        if sha(cohort/"comp.csv") != lock["csv_sha256"]["comp"]:
            raise ValueError("COMP labels changed")
        if a.smoke:
            if a.readout!="official-mlp":
                raise ValueError("The real optimizer smoke applies to the MLP")
            frame = pd.read_csv(cohort/"train.csv")
            # Select whole, deterministic episodes, with a full-size first update.
            groups = (frame.subset+"/"+frame.episode_id).to_numpy()
            eligible = np.unique(groups)
            ids = np.flatnonzero(np.isin(groups, np.random.default_rng(0).permutation(eligible)[:128]))
            p = make("mlp").fit(vectors["train"][ids],frame[native.A_COLS].to_numpy()[ids],groups[ids])
            pred = p.predict(vectors["val"][:16])
            if pred.shape!=(16,12) or not np.isfinite(pred).all():
                raise ValueError("Real Causal optimizer smoke failed")
            dump(output/"smoke.json",dict(passed=True, scope="real interface/gradient smoke only"))
            return
        record_probe_predictions(native,output)
        result = native.run(SimpleNamespace(
            model=a.model,stride=15,probe="mlp" if a.readout=="official-mlp" else "ridge",
            la_root=str(features),train_csv=str(cohort/"train.csv"),val_csv=str(cohort/"val.csv"),
            unseen_csv=str(cohort/"unseen.csv"),comp_val_csv=str(cohort/"comp.csv"),
            pairs_dir=lock["source_root"],max_dim=1000000,oracle=None,oracle_noise=0.))
    result.update(readout_profile=a.readout,seed=a.seed if a.readout=="official-mlp" else None,
                  cohort_sha256=sha(cohort/"cohort.json"),model=a.model,weights=weight_proof,
                  data_protocol="matched Causal, stride15, source-episode purge; no PCA",
                  native_metrics_sha256=lock["native_metrics_sha256"],
                  backbone_record_sha256="55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b",
                  regression_source_sha256=sha(a.source) if a.readout=="official-mlp" else None,
                  ridge_solver_parity_passed=True, independent_review="pending",
                  global_batch=2048 if a.readout=="official-mlp" else None)
    dump(output/"result.json",result)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for key in ("lary-root","source","cohort","features","model","output","weights"):
        parser.add_argument("--"+key,required=True)
    parser.add_argument("--readout",choices=["official-mlp","ridge","ic"],required=True)
    parser.add_argument("--scratch",default=None)
    parser.add_argument("--ranks",type=int,choices=[4,8],default=4)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--train-python",default=sys.executable)
    run(parser.parse_args())
