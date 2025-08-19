## Requirements
The main requirements are:
- python 3.7
- torch 1.7.0 
- tqdm
- transformers 4.2.1
- bert4keras

## Usage
* **Get pre-trained BERT model**
Download [BERT-BASE-CASED](https://huggingface.co/bert-base-cased) and put it under `./pretrained`.

* **Train and select the model**
```
python run.py --dataset=CTIdata_1  --train=train  --rounds=4
python run.py --dataset=CTIdata_2  --train=train  --rounds=4

```

* **Evaluate on the test set**
```
python run.py --dataset=CTIdata_1  --train=test  --rounds=4
python run.py --dataset=CTIdata_2   --train=test  --rounds=4

```

### Acknowledgement
Parts of our codes come from [bert4keras](https://github.com/bojone/bert4keras).
