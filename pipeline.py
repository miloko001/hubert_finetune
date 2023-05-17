import speechbrain as sb
from speechbrain.tokenizers.SentencePiece import SentencePiece as SP
import numpy as np
import torch
import pudb
import json
#from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.preprocessing import LabelBinarizer as LB
from ARPABET import apbet
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor,Wav2Vec2CTCTokenizer
from datasets import load_dataset,load_metric
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import string

def prepare_dataset(batch):
    signal,sr = sf.read(batch["path"])
    perceived = batch["perceived"].translate(str.maketrans('','',string.punctuation))

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(signal, sampling_rate=sr).input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(perceived).input_ids
    return batch



def load_data():

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset = load_dataset("csv",data_files="./data/train_5hr.csv",column_names = ["path","spk","stime","etime","perceived","target"],delimiter="\t")
    valid_dataset = load_dataset("csv",data_files="./data/dev.csv",column_names = ["path","spk","stime","etime","perceived","target"],delimiter="\t")

        

    tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="UNK", pad_token="PAD", word_delimiter_token="",padding="longest")    
    
    tokenizer.push_to_hub("hubert_ft")

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    global processor
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer) 
    
    train_dataset = train_dataset.map(prepare_dataset)
    valid_dataset = valid_dataset.map(prepare_dataset)
    
    train_dataset = train_dataset.remove_columns(["path","spk","stime","etime","perceived","target"])
    valid_dataset = valid_dataset.remove_columns(["path","spk","stime","etime","perceived","target"])

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    
    wer_metric = load_metric("wer")

    return train_dataset,valid_dataset, data_collator, wer_metric, processor



class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """
    def __init__(self,processor: Wav2Vec2Processor,padding: Union[bool, str] = True,
            max_length: Optional[int] = None,
            max_length_labels: Optional[int] = None,
            pad_to_multiple_of: Optional[int] = None,
            pad_to_multiple_of_labels: Optional[int] = None):
        
        self.processor=processor
        self.max_length=max_length
        self.max_length_labels=max_length_labels
        self.padding =padding
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_to_multiple_of_labels = pad_to_multiple_of_labels


    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
