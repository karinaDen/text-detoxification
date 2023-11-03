metrics calculation was credited to [this](https://github.com/s-nlp/detox/blob/main) repository.


For wieting similarity, you need to download the weights of the model [here](https://storage.yandexcloud.net/nlp/wieting_similarity_data.zip)
Put the weights in the folder `wieting_similarity_data` and run the following command:

```python metric.py --inputs PATH_TO_INPUTS --preds PATH_TO_PREDS```

Both inputs and predictions should be plain text files with one comment per line.