from datasets import DatasetDict, Dataset


def create_dataset(train_df, val_df):
    """
    Create a HuggingFace Dataset from the given DataFrame.
    Args:
        - train_df: pandas DataFrame containing columns 'reference', 'translation', 'ref_tox', and 'trn_tox'
        - val_df: pandas DataFrame containing columns 'reference', 'translation', 'ref_tox', and 'trn_tox'
    Returns:
        - hf_dataset: HuggingFace Dataset dictionary
    """

    # Create the datasets
    train_dataset = Dataset.from_dict(
        {"translation": [{"toxic": source, "detox": target} 
                         for source, target in zip(train_df["reference"], train_df["translation"])]}
    )
    valid_dataset = Dataset.from_dict(
        {"translation": [{"toxic": source, "detox": target} 
                         for source, target in zip(val_df["reference"], val_df["translation"])]}
    )

    # Return the datasets in a dictionary
    return DatasetDict(train=train_dataset, validation=valid_dataset)



def preprocess_function(examples, tokenizer, prefix, max_length=128, 
                        source="toxic", target="detox"):
    """
    Preprocess the given examples using the given tokenizer.

    Args:
        examples: The examples to be preprocessed.
        tokenizer: The tokenizer to be used.
        prefix: The prefix to be used.
        max_length: The maximum length of the input sequence.
        source: The source sentence.
        target: The target sentence.

    Returns:
        The preprocessed examples.
    """
    inputs = [prefix + ex[source] for ex in examples["translation"]]
    targets = [ex[target] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(targets, max_length=max_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs


def postprocess_text(preds, labels):
    """
    Simple postprocessing for text
    Args:
        preds: The predicted texts.
        labels: The label texts.

    Returns:
        The postprocessed predicted texts and label texts.
    """
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels
