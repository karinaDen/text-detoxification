import torch
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from create_dataset import preprocess_function, postprocess_text, create_dataset
import argparse
from datasets import load_metric
import pandas as pd

import warnings 
warnings.filterwarnings('ignore')

tokenizer = AutoTokenizer.from_pretrained('t5-small')


def compute_metrics(eval_preds):
    """
    Computes the metrics for the given predictions.
    Args:
    - eval_preds: The predictions made by the model.
    Returns:
    - A dictionary containing the computed metrics.
    """
    metric = load_metric("sacrebleu")



    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result



def train(args, model):
    """
    Trains the model.
    Args:
    - args: The arguments passed to the script.
    - model: The model to train.
    """
    epochs = args.epochs

    # Prefix for model input
    prefix = "paraphrase:"

    # setting random seed for transformers library
    transformers.set_seed(420)

    # defining the parameters for training
    batch_size = args.batch_size

    # read traiin and validation data
    train_df = pd.read_csv(args.train_path + "train.csv")
    val_df = pd.read_csv(args.train_path + "validation.csv")

    # create HuggingFace Dataset
    dataset = create_dataset(train_df, val_df)

    args = Seq2SeqTrainingArguments(
        f"{args.output_path}-finetuned-toxic-to-detox",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=10,
        predict_with_generate=True,
        fp16=True,
        report_to='tensorboard',
    )

    # preprocess the datasets
    tokenized_datasets = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, prefix=prefix),
        batched=True
    )

    # create the data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # instantiate the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # train the model
    trainer.train()

    # save the model
    trainer.save_model(f"{args.output_path}best-model")


def main(args):
    model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')


    train(args, model)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", help="path to train dataet", default="../data/interim/")
    parser.add_argument("--output_path", help="path to save model", default="..src/models/T5/")
    parser.add_argument("--epochs", help="number of epochs", default=10)
    parser.add_argument("--batch_size", help="batch size", default=32)
    args = parser.parse_args()


    main(args)