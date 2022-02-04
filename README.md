# DSCGN

## Requirements

At least one GPU is required to run the code.

Before running, you need to first install the required packages by typing following commands:

```
$ pip3 install -r requirements.txt
```

Python 3.6 or above is strongly recommended; using older python versions might lead to package incompatibility issues.

## Reproducing the Results

We provide training bash scripts [```main.sh```](main.sh) for running DSCGN on one data set.

You can change the data set to run DSCGN on different datasets.

## Command Line Arguments

The meanings of the command line arguments will be displayed upon typing
```
python src/train.py -h
```
The following arguments directly affect the performance of the DSCGN and need to be set carefully:

* ```train_batch_size```, ```accum_steps```, ```gpus```: These three arguments should be set together. You need to make sure that the effective training batch size, calculated as ```train_batch_size * accum_steps * gpus```, is around 128.
* ```eval_batch_size```: This argument only affects the speed of the algorithm; use as large evaluation batch size as your GPUs can hold.
* ```max_len```: This argument controls the maximum length of utterances (Web documents) fed into the DSCGN (longer utterances(Web documents) will be truncated). Ideally, ```max_len``` should be set to the length of the longest utterance(Web document) (```max_len``` cannot be larger than ```512``` under BERT architecture), but using larger ```max_len``` also consumes more GPU memory, resulting in smaller batch size and longer training time. Therefore, you can trade DSCGN accuracy for faster training by reducing ```max_len```.
* ```train_epochs```: They control how many epochs to train the DSCGN

Other arguments can be kept as their default values.
More details will be provided soon.
