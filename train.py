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

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids,spaces_between_special_tokens=True)
    # we do not want to group tokens when computing the metrics

    label_str = processor.batch_decode(pred.label_ids,group_tokens=False,spaces_between_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    
    best_probe,layer = probe()
    
    return {"wer": wer,"best_probe_value":best_probe,"best_probe_layer":layer}

def corr_calc(pred,gt):

    pred_c = pred - np.mean(pred,axis=0)
    gt_c = gt - np.mean(gt,axis=0)
    corr = np.mean(np.sum(pred_c*gt_c,axis=0)/np.sqrt(np.sum(pred_c**2,axis=0)*np.sum(gt_c**2,axis=0)))

    return corr

def remove_3D_data(y):
    ind_to_del = [1,4,6,7,8,9,10,11,13,16,18,19,20,22,25,27,28,29]
    twod_y = np.delete(y,ind_to_del,axis=1)
    return twod_y


def probe():
    
    features = {}

    def get_features(name):
        def hook(model, input, output):
            features[name] = output[0].detach().cpu().numpy()
        return hook

    for i,layer in enumerate(model.base_model.encoder.layers): 
        print(f"{i}")
        layer.register_forward_hook(get_features(f'{i} feats'))


    med = "./eval/medina/truewav/"
    ema_path = "./eval/medina/truema/"
    
    files = os.listdir(med)

    train = files[:100]
    test = files[-100:]
    coefs = []
    em = {}
    hst = {}

    for i in range(23):
        em[f"{i}"] = []
        hst[f"{i}"] = []

    for f in tqdm(train):
        data,samplerate = sf.read(med+f)
        art = remove_3D_data(np.load(f"{ema_path}{f[:-4]}.npy"))
        input_values = processor(data,sampling_rate=samplerate,return_tensors="pt").input_values.to(device)
        hidden_states = model(input_values).logits
        
        for l in range(23):
            hstates = features[f"{l} feats"].squeeze()

            if np.size(art,0)!=np.size(hstates,0):
                art = art[:np.size(hstates,0)]
    
            em[f"{l}"].append(art)
            hst[f"{l}"].append(hstates)
    
    mu = remove_3D_data(np.expand_dims(np.load("./eval/medina/medina_mean.npy"),axis=0))
    std = remove_3D_data(np.expand_dims(np.load("./eval/medina/medina_std.npy"),axis=0))

    for i in tqdm(range(23)):
        ema = np.concatenate(em[f"{i}"],axis=0)
        ema = (ema-mu)/std

        feats = np.concatenate(hst[f"{i}"],axis=0)
        est = LinearRegression(fit_intercept=False).fit(feats,ema)

        coefs.append(est.coef_)
        

    corrs = {}

    for i in range(23):
        corrs[f"{i}"] = []

    for f in tqdm(test):
        data, samplerate = sf.read(med+f)
        art = remove_3D_data(np.load(f"{ema_path}{f[:-4]}.npy"))

        input_values = processor(data,sampling_rate=samplerate,return_tensors="pt").input_values.to(device)
        hidden_states = model(input_values).logits
        
        for l in range(23):

            hstates = features[f"{l} feats"].squeeze()
            pred = np.matmul(hstates,coefs[l].T)
            if np.size(art,0)!=np.size(hstates,0):
                art = art[:np.size(hstates,0)]

            gt = (art-mu)/std

            corrs[f"{l}"].append(corr_calc(pred,gt))

    corr_final = []

    for i in range(23):
            corr_final.append(np.mean(corrs[f"{i}"]))
    
    fig,ax =plt.subplots()

    ax.plot(corr_final)
    ax.set_ylabel("Correlation")
    ax.set_xlabel("Layer")
    wandb.log({"Probe Performance":fig})

    probe = np.max(corr_final)
    layer = np.argmax(corr_final)
    

    return probe,layer



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
    

    training_args = TrainingArguments(
        output_dir ="hubert_ft/5hr",
        per_device_train_batch_size=8,
        evaluation_strategy="steps",
        num_train_epochs=50,
        fp16=True,
        gradient_checkpointing=True,
        save_steps=1000,
        eval_steps=500,
        logging_steps=10,
        learning_rate=1e-5,
        warmup_steps=10,
        save_total_limit=2,
        report_to="wandb",
        run_name="hubert_finetuning",
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
