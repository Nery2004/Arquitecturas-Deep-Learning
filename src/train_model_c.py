"""Entrena la apuesta C con TRAIN y selecciona exclusivamente con VALIDATION."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import random
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import RANDOM_SEED
from .evaluation import PRIMARY_METRIC_NAME, average_precision
from .hybrid_model import HybridGRU
from .preprocessing import (AGGREGATE_FEATURES, CURRENT_CATEGORICAL_FEATURES,
                            CURRENT_NUMERIC_FEATURES, MAX_HISTORY,
                            SEQUENCE_CATEGORICAL_FEATURES, SEQUENCE_NUMERIC_FEATURES,
                            file_sha256, processed_fingerprint)
from .sequence_model import count_trainable_parameters

HYPOTHESIS = ("Combinar la información secuencial con las variables agregadas mejorará la detección "
              "porque cada representación resume un aspecto diferente del comportamiento de una tarjeta.")
SUCCESS_CRITERION = ("C debe superar el AP de B en VALIDATION y posteriormente no aumentar el costo económico; "
                     "esta etapa solo resuelve la parte predictiva.")
C_INPUT_KEYS = ("X_sequence_numeric", "X_sequence_categorical", "history_length",
                "X_current_numeric", "X_current_categorical", "X_aggregate", "y", "example_index")
BATCH_SIZE = 256
MAX_EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 8e-4


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True); torch.set_num_threads(4)


def load_hybrid_split(split: str, root: Path = Path(".")) -> dict[str, np.ndarray]:
    if split not in {"train", "validation"}:
        raise ValueError("Modelo C solo carga TRAIN o VALIDATION")
    with np.load(root / f"data/processed/model_inputs_{split}.npz") as archive:
        return {key: archive[key] for key in C_INPUT_KEYS}


class HybridDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray]) -> None: self.a = arrays
    def __len__(self) -> int: return len(self.a["y"])
    def __getitem__(self, i: int) -> tuple[torch.Tensor, ...]:
        a=self.a
        return (torch.from_numpy(a["X_sequence_numeric"][i]),
                torch.from_numpy(a["X_sequence_categorical"][i].astype(np.int64)),
                torch.tensor(a["history_length"][i],dtype=torch.long),
                torch.from_numpy(a["X_current_numeric"][i]),
                torch.from_numpy(a["X_current_categorical"][i].astype(np.int64)),
                torch.from_numpy(a["X_aggregate"][i]),torch.tensor(a["y"][i],dtype=torch.float32))


def verify_frozen_state(root: Path) -> tuple[dict[str, Any],dict[str, Any],dict[str,Any],str]:
    processed=json.loads((root/"data/processed/processed_metadata.json").read_text())
    a=json.loads((root/"artefactos/model_a/model_a_metadata.json").read_text())
    b=json.loads((root/"artefactos/model_b/model_b_metadata.json").read_text())
    f=json.loads((root/"artefactos/model_b/falsification_metadata.json").read_text())
    artifact=root/"artefactos/preprocessing"
    paths=[root/"data/processed/split_config.json",root/"data/processed/example_index.csv",root/"data/processed/aggregate_features_raw.csv",
           *(artifact/f"{name}.json" for name in ("aggregate_scaler","current_scaler","sequence_scaler","vocabularies")),
           *(root/f"data/processed/model_inputs_{s}.npz" for s in ("train","validation","test"))]
    assert processed_fingerprint(paths)==processed["processed_fingerprint"]==a["processed_dataset_fingerprint"]==b["processed_dataset_fingerprint"]
    assert file_sha256(root/"data/generated/transactions.csv")==processed["dataset_source_fingerprint"]
    assert file_sha256(root/"artefactos/model_a/model_a.joblib")==a["model_fingerprint_sha256"]
    assert file_sha256(root/"artefactos/model_b/model_b.pt")==b["model_fingerprint_sha256"]==f["model_b_checkpoint_hash"]
    split_hash=file_sha256(root/"data/processed/split_config.json");assert split_hash==a["split_fingerprint"]==b["split_fingerprint"]
    return processed,a,b,split_hash


def candidates() -> list[dict[str,Any]]:
    return [
        {"candidate_id":"c1","aggregate_hidden":16,"fusion_hidden":32,"dropout":.4,"learning_rate":LEARNING_RATE},
        {"candidate_id":"c2","aggregate_hidden":32,"fusion_hidden":64,"dropout":.4,"learning_rate":LEARNING_RATE},
        {"candidate_id":"c3","aggregate_hidden":32,"fusion_hidden":32,"dropout":.5,"learning_rate":LEARNING_RATE},
    ]


def make_model(config: dict[str,Any], vocab: dict[str,Any]) -> HybridGRU:
    return HybridGRU(len(vocab["merchant_category"]),len(vocab["channel"]),aggregate_size=len(AGGREGATE_FEATURES),
        aggregate_hidden=config["aggregate_hidden"],fusion_hidden=config["fusion_hidden"],dropout=config["dropout"])


def loaders(train: dict[str,np.ndarray],validation: dict[str,np.ndarray]) -> tuple[DataLoader,DataLoader]:
    generator=torch.Generator().manual_seed(RANDOM_SEED)
    return (DataLoader(HybridDataset(train),batch_size=BATCH_SIZE,shuffle=True,num_workers=0,generator=generator),
            DataLoader(HybridDataset(validation),batch_size=BATCH_SIZE,shuffle=False,num_workers=0))


def forward(model: nn.Module,batch: tuple[torch.Tensor,...],device:torch.device) -> tuple[torch.Tensor,torch.Tensor]:
    tensors=[x.to(device) for x in batch]
    return model(*tensors[:-1]),tensors[-1]


@torch.no_grad()
def evaluate(model:nn.Module,loader:DataLoader,loss_fn:nn.Module,device:torch.device,neutral_aggregates:bool=False):
    model.eval();losses=[];scores=[];labels=[]
    for batch in loader:
        if neutral_aggregates:
            batch=list(batch);batch[5]=torch.zeros_like(batch[5]);batch=tuple(batch)
        logits,y=forward(model,batch,device);losses.append(float(loss_fn(logits,y))*len(y))
        scores.append(torch.sigmoid(logits).cpu().numpy());labels.append(y.cpu().numpy())
    y=np.concatenate(labels);risk=np.concatenate(scores)
    return sum(losses)/len(y),average_precision(y,risk),risk


def train_candidate(config:dict[str,Any],train:dict[str,np.ndarray],validation:dict[str,np.ndarray],vocab:dict[str,Any],device:torch.device):
    set_seed(RANDOM_SEED);train_loader,val_loader=loaders(train,validation);model=make_model(config,vocab).to(device)
    pos_weight=float((train["y"]==0).sum()/train["y"].sum())
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight,device=device));optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"])
    history=[];best_ap=-1.;best_epoch=0;best_state=None;stale=0;started=time.perf_counter()
    for epoch in range(1,MAX_EPOCHS+1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True);logits,y=forward(model,batch,device);loss=loss_fn(logits,y);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        train_loss,train_ap,_=evaluate(model,DataLoader(HybridDataset(train),batch_size=BATCH_SIZE),loss_fn,device)
        val_loss,val_ap,_=evaluate(model,val_loader,loss_fn,device)
        history.append({"epoch":epoch,"train_loss":train_loss,"validation_loss":val_loss,"train_ap":train_ap,"validation_ap":val_ap,"learning_rate":config["learning_rate"]})
        if val_ap>best_ap+1e-8:best_ap=val_ap;best_epoch=epoch;best_state=deepcopy(model.state_dict());stale=0
        else:
            stale+=1
            if stale>=PATIENCE:break
    model.load_state_dict(best_state);train_loss,train_ap,train_scores=evaluate(model,DataLoader(HybridDataset(train),batch_size=BATCH_SIZE),loss_fn,device)
    val_loss,val_ap,val_scores=evaluate(model,val_loader,loss_fn,device)
    result={**config,"best_epoch":best_epoch,"epochs_run":len(history),"trainable_parameters":count_trainable_parameters(model),
            "train_loss":train_loss,"validation_loss":val_loss,"train_ap":train_ap,"validation_ap":val_ap,
            "train_validation_gap":train_ap-val_ap,"fit_time_seconds":time.perf_counter()-started,"pos_weight":pos_weight,
            "train_scores":train_scores,"validation_scores":val_scores}
    return model,history,result


def write_scores(path:Path,indices:np.ndarray,index:list[dict[str,str]],y:np.ndarray,scores:np.ndarray,split:str)->None:
    fields=("example_id","transaction_id","target_timestamp","y_true","risk_score","split","fraud_type","hard_negative_type")
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator="\n");writer.writeheader()
        for local,global_index in enumerate(indices):
            row=index[int(global_index)];writer.writerow({"example_id":row["example_id"],"transaction_id":row["transaction_id"],"target_timestamp":row["target_timestamp"],"y_true":int(y[local]),"risk_score":f"{scores[local]:.17g}","split":split,"fraud_type":row["fraud_type"],"hard_negative_type":row["hard_negative_type"]})


def group_summary(index:list[dict[str,str]],indices:np.ndarray,scores:np.ndarray,field:str,names:tuple[str,...]):
    groups={name:[] for name in names}
    for local,global_index in enumerate(indices):
        name=index[int(global_index)][field]
        if name in groups:groups[name].append(float(scores[local]))
    return {k:{"n":len(v),"mean":float(np.mean(v)),"median":float(np.median(v)),"q90":float(np.quantile(v,.9))} for k,v in groups.items()}


def save_curves(history:list[dict[str,float]],root:Path)->tuple[Path,Path]:
    epochs=[r["epoch"] for r in history];loss=root/"figures/model_c_training_loss.png";ap=root/"figures/model_c_training_ap.png"
    fig,ax=plt.subplots(figsize=(8,5));ax.plot(epochs,[r["train_loss"] for r in history],label="TRAIN");ax.plot(epochs,[r["validation_loss"] for r in history],label="VALIDATION");ax.set(title="Modelo C — Loss por epoch",xlabel="Epoch",ylabel="BCEWithLogitsLoss");ax.legend();fig.tight_layout();fig.savefig(loss,dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));ax.plot(epochs,[r["train_ap"] for r in history],label="TRAIN");ax.plot(epochs,[r["validation_ap"] for r in history],label="VALIDATION");ax.set(title="Modelo C — Average Precision por epoch",xlabel="Epoch",ylabel="AP");ax.legend();fig.tight_layout();fig.savefig(ap,dpi=150);plt.close(fig);return loss,ap


def run_training(root:Path=Path("."))->dict[str,Any]:
    processed,a,b,split_hash=verify_frozen_state(root);train=load_hybrid_split("train",root);validation=load_hybrid_split("validation",root)
    assert len(train["y"])==a["n_train"]==b["n_train"] and len(validation["y"])==a["n_validation"]==b["n_validation"]
    assert np.isfinite(train["X_aggregate"]).all() and np.isfinite(validation["X_aggregate"]).all()
    vocab=json.loads((root/"artefactos/preprocessing/vocabularies.json").read_text());device=torch.device("cpu");trained=[]
    for config in candidates():
        model,history,result=train_candidate(config,train,validation,vocab,device);trained.append({"model":model,"history":history,"result":result})
    best=max(x["result"]["validation_ap"] for x in trained);selected=next(x for x in trained if x["result"]["validation_ap"]>=best-.001)
    result=selected["result"];replica_config={k:result[k] for k in ("candidate_id","aggregate_hidden","fusion_hidden","dropout","learning_rate")}
    _,_,replica=train_candidate(replica_config,train,validation,vocab,device)
    model=selected["model"];loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(result["pos_weight"]));val_loader=DataLoader(HybridDataset(validation),batch_size=BATCH_SIZE)
    _,neutral_ap,_=evaluate(model,val_loader,loss_fn,device,neutral_aggregates=True)
    output=root/"artefactos/model_c";output.mkdir(parents=True,exist_ok=True);(root/"experiments").mkdir(exist_ok=True)
    checkpoint={"state_dict":model.state_dict(),"config":replica_config,"merchant_vocab_size":len(vocab["merchant_category"]),"channel_vocab_size":len(vocab["channel"]),"aggregate_size":len(AGGREGATE_FEATURES)}
    model_path=output/"model_c.pt";torch.save(checkpoint,model_path)
    with (root/"data/processed/example_index.csv").open(newline="") as handle:index=list(csv.DictReader(handle))
    write_scores(output/"train_scores.csv",train["example_index"],index,train["y"],result["train_scores"],"train");write_scores(output/"validation_scores.csv",validation["example_index"],index,validation["y"],result["validation_scores"],"validation")
    with (output/"training_history.csv").open("w",newline="") as handle:writer=csv.DictWriter(handle,fieldnames=selected["history"][0].keys(),lineterminator="\n");writer.writeheader();writer.writerows(selected["history"])
    with (root/"experiments/model_c_results.csv").open("w",newline="") as handle:
        fields=("candidate_id","aggregate_hidden","fusion_hidden","dropout","learning_rate","trainable_parameters","best_epoch","epochs_run","train_ap","validation_ap","delta_vs_b","fit_time_seconds","selected")
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator="\n");writer.writeheader()
        for item in trained:
            row={k:v for k,v in item["result"].items() if k in fields};row["delta_vs_b"]=item["result"]["validation_ap"]-b["validation_ap"];row["selected"]=item is selected;writer.writerow(row)
    loss_path,ap_path=save_curves(selected["history"],root)
    fig,ax=plt.subplots(figsize=(8,5));ax.bar(("A","B","C","C agregados=0"),(a["validation_ap"],b["validation_ap"],result["validation_ap"],neutral_ap),color=("#5b8c5a","#3b82a0","#7251a3","#d39c43"));ax.set(title="Modelos A/B/C — Validation AP",ylabel="Average Precision",ylim=(0,1));fig.tight_layout();comparison=root/"figures/model_abc_validation_comparison.png";fig.savefig(comparison,dpi=150);plt.close(fig)
    fraud=group_summary(index,validation["example_index"],result["validation_scores"],"fraud_type",("testing_cashout","channel_takeover","amount_anomaly"));hard=group_summary(index,validation["example_index"],result["validation_scores"],"hard_negative_type",("legitimate_trip","legitimate_large_purchase","legitimate_shopping_spree","legitimate_micro_purchases"))
    metadata={"model_name":"MODEL_C_CANDIDATE","hypothesis":HYPOTHESIS,"success_criterion":SUCCESS_CRITERION,"control_model":"B",
        "architecture":"GRU history + current branch + historical aggregate branch","training_from_scratch":True,
        "gru":{"hidden_size":64,"num_layers":1,"bidirectional":False},"embeddings":{"merchant":6,"channel":3},
        "max_history":MAX_HISTORY,"sequence_numeric_features":list(SEQUENCE_NUMERIC_FEATURES),
        "sequence_categorical_features":list(SEQUENCE_CATEGORICAL_FEATURES),
        "current_numeric_features":list(CURRENT_NUMERIC_FEATURES),
        "current_categorical_features":list(CURRENT_CATEGORICAL_FEATURES),
        "current_branch_hidden":64,"aggregate_branch":{"input_features":list(AGGREGATE_FEATURES),"hidden":result["aggregate_hidden"]},
        "fusion_hidden":result["fusion_hidden"],"dropout":result["dropout"],"learning_rate":result["learning_rate"],"optimizer":"AdamW","batch_size":BATCH_SIZE,"pos_weight":result["pos_weight"],"best_epoch":result["best_epoch"],"epochs_run":result["epochs_run"],"trainable_parameters":result["trainable_parameters"],"seed":RANDOM_SEED,
        "dataset_fingerprint":processed["dataset_source_fingerprint"],"processed_fingerprint":processed["processed_fingerprint"],"split_fingerprint":split_hash,
        "training_split":"TRAIN","selection_split":"VALIDATION","early_stopping_split":"VALIDATION","test_evaluated":False,
        "train_ap":result["train_ap"],"validation_ap":result["validation_ap"],"train_validation_gap":result["train_validation_gap"],
        "model_b_validation_ap":b["validation_ap"],"delta_c_vs_b":result["validation_ap"]-b["validation_ap"],"relative_delta_c_vs_b":(result["validation_ap"]-b["validation_ap"])/b["validation_ap"],
        "predictive_criterion_met":result["validation_ap"]>b["validation_ap"],"economic_criterion_status":"pending",
        "ablation_normal_ap":result["validation_ap"],"ablation_aggregates_neutral_ap":neutral_ap,"ablation_difference":result["validation_ap"]-neutral_ap,
        "validation_fraud_score_summary":fraud,"validation_hard_negative_score_summary":hard,
        "reproducibility_validation_ap_run_1":result["validation_ap"],"reproducibility_validation_ap_run_2":replica["validation_ap"],"reproducibility_absolute_difference":abs(result["validation_ap"]-replica["validation_ap"]),
        "primary_metric":PRIMARY_METRIC_NAME,"model_fingerprint_sha256":file_sha256(model_path),"software_versions":{"python":sys.version.split()[0],"numpy":np.__version__,"torch":torch.__version__,"scikit-learn":importlib.metadata.version("scikit-learn")},"created_at_utc":datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (output/"model_c_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    clean=[{k:v for k,v in item["result"].items() if k not in ("train_scores","validation_scores")} for item in trained]
    return {"metadata":metadata,"candidate_results":clean,"figures":[str(loss_path),str(ap_path),str(comparison)]}


def main()->None:print(json.dumps(run_training(),indent=2))
if __name__=="__main__":main()
