import numpy as np
import argparse
import torch
import torch.nn.functional as F
from model import phon_finetune
from pipeline import load_data
import pudb
import wandb
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm
from transformers import HubertForCTC, TrainingArguments, Trainer
import datasets
import transformers
import os
import soundfile as sf
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import glob
import re

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids,spaces_between_special_tokens=True)
    # we do not want to group tokens when computing the metrics

    label_str = processor.batch_decode(pred.label_ids,group_tokens=False,spaces_between_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    speakers = ["01MBF","02MBF","03MBM","04MSF","05ENF","06ENM","07ENF","08MBM","09ENF","11MBF","12MSF",\
            "14MSF","15ENM","16ENM","17ENF","18ENF","19ENM","20MBF","21ENF","22MBF","24MSF","25MSM","26MSM",\
            "27MSM","28ENF","29MBM","30MSM","31MBM","32ENM","33ENM","34ENM","35ENM","36ENF","37ENF","38ENM","39ENM","40ENF"]

    corr_layer_ = []
    max_corr_ = []
    max_corr_layer_ = []
    for s in speakers:
        corr_layer = mae_probe(s)
        corr_layer_.append(corr_layer)
    a = np.mean(np.vstack(corr_layer_),axis=0) 
            
    best_probe_layer = np.argmax(a)
    best_probe_value = np.max(a)

    fig,ax = plt.subplots(1,1,figsize=(12,8))

    plt.plot(a)
    ax.set_ylabel("Correlation")
    ax.set_xlabel("Layer")
    wandb.log({"Layer-wise Correlation":fig})
    
    return {"WER": wer,"Best Avg Correlation":best_probe_value,"Best Layer":best_probe_layer}

def corr_calc(pred,gt):

    pred_c = pred - np.mean(pred,axis=0)
    gt_c = gt - np.mean(gt,axis=0)
    corr = np.mean(np.sum(pred_c*gt_c,axis=0)/np.sqrt(np.sum(pred_c**2,axis=0)*np.sum(gt_c**2,axis=0)))

    return corr

def remove_3D_data(y):
    ind_to_del = [1,4,6,7,8,9,10,11,13,16,18,19,20,22,25,27,28,29]
    twod_y = np.delete(y,ind_to_del,axis=1)
    return twod_y


def mae_probe(speaker):
    
    features = {}

    def get_features(name):
        def hook(model, input, output):
            features[name] = output[0].detach().cpu().numpy()
        return hook

    for i,layer in enumerate(model.base_model.encoder.layers): 
        layer.register_forward_hook(get_features(f'{i} feats'))


    files = glob.glob(f"../probing/mae/{speaker}/split_wav/*")
    files = [f for f in files if "Words" not in f]                                                                                                                                                 
    comma = [f for f in files if "Comma" in f]                                                                                                                                                     
    bamboo = [f for f in files if "Bamboo" in f]                                                                                                                                                   
    caterpillar = [f for f in files if "Caterpillar" in f]                                                                                                                                         
    sents = [f for f in files if "Sents2" in f or "Sents3" in f]                                                                                                                                   
    passages = comma + bamboo + caterpillar + sents                                                                                                                                                
    train = [f for f in files if f not in passages]
    test = [f for f in files if f in passages]
    mu = np.load(f"../probing/mae/{speaker}/ema_mean.npy")
    std = np.load(f"../probing/mae/{speaker}/ema_std.npy")        

    ema_path = f"../probing/mae/{speaker}/split_ema/"
    coefs = []
    em = []
    hst = {}

    for i in range(24):
        hst[f"{i}"] = []
        hst[f"{i}"] = []

    for f in tqdm(train):
        data,samplerate = sf.read(f)
        ema_path = re.sub("split_wav","split_ema",f)
        art = np.load(f"{ema_path[:-4]}.npy")
        input_values = processor(data,sampling_rate=samplerate,return_tensors="pt").input_values.to(device)
        hidden_states = model(input_values).logits
        
        
        for l in range(24):
            hstates = features[f"{l} feats"].squeeze()
            
            if l ==0:
                if np.size(art,0)!=np.size(hstates,0):
                    art = art[:np.size(hstates,0)]
                
                em.append(art)

    
            hst[f"{l}"].append(hstates)
    ema = np.concatenate(em,axis=0)[:15000]
    ema = (ema-mu)/std

    for i in tqdm(range(24)):

        feats = np.concatenate(hst[f"{i}"],axis=0)[:15000]
        est = LinearRegression(fit_intercept=False).fit(feats,ema)

        coefs.append(est.coef_)
        

    corrs = {}

    for i in range(24):
        corrs[f"{i}"] = []

    for f in tqdm(test):
        data, samplerate = sf.read(f)
        ema_path = re.sub("split_wav","split_ema",f)
        art = np.load(f"{ema_path[:-4]}.npy")

        input_values = processor(data,sampling_rate=samplerate,return_tensors="pt").input_values.to(device)
        hidden_states = model(input_values).logits
        
        for l in range(24):

            hstates = features[f"{l} feats"].squeeze()
            pred = np.matmul(hstates,coefs[l].T)
            if np.size(art,0)!=np.size(hstates,0):
                art = art[:np.size(hstates,0)]

            gt = (art-mu)/std

            corrs[f"{l}"].append(corr_calc(pred,gt))

    layer_corr = []

    for i in range(24):
            layer_corr.append(np.mean(corrs[f"{i}"]))
        

    return layer_corr

if __name__ == "__main__":
    transformers.logging.set_verbosity_info()
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo_name ="hubert_ft"
    train_dataset, valid_dataset,data_collater, wer_metric, processor = load_data()
    global model
    model = HubertForCTC.from_pretrained("facebook/hubert-large-ll60k",
            vocab_size=42,
            ctc_loss_reduction="mean",
            pad_token_id=processor.tokenizer.pad_token_id).to(device)
    
    

    model.freeze_feature_encoder()
    #os.environ["WANDB_DISABLED"] = "true"    

    training_args = TrainingArguments(
        output_dir ="hubert_ft/test",
        per_device_train_batch_size=16,
        evaluation_strategy="steps",
        num_train_epochs=2,
        fp16=True,
        gradient_checkpointing=True,
        save_steps=1000,
        eval_steps=1,
        logging_steps=100,
        learning_rate=1e-4,
        warmup_steps=1500,
        save_total_limit=2,
        #report_to="wandb",
        #run_name="hubert_finetuning",
        push_to_hub=False)
    
    trainer = Trainer(
    model=model,
    data_collator=data_collater,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset["train"],
    eval_dataset=valid_dataset["train"],
    tokenizer=processor.feature_extractor)
    trainer.train()
    

    wandb.finish()
