# Evaluate bias in MLM

Code for the paper: [Unmasking the Mask -- Evaluating Social Biases in Masked Language Models](https://arxiv.org/abs/2104.07496). If you use any part of this work, please cite the following citation:
```
@article{Kaneko:AUL:2021,
    title={Unmasking the Mask -- Evaluating Social Biases in Masked Language Models},
    author={Masahiro Kaneko and Danushka Bollegala},
    journal={arXiv},
    year={2021}
}
```
## 🛠 Setup

You can install all required packages with following command.
```
pip install -r requirements.txt
```

You can downlaod [CrowS-Pairs (CP)](https://github.com/nyu-mll/crows-pairs) and [StereoSet (SS)](https://github.com/moinnadeem/StereoSet) datasets and preprocess them with following commands.
```
mkdir -p data
wget -O data/cp.csv https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv
wget -O data/ss.json https://raw.githubusercontent.com/moinnadeem/StereoSet/master/data/dev.json
python -u preprocess.py --input crows_pairs --output data/paralled_cp.json
python -u preprocess.py --input stereoset --output data/paralled_ss.json
```


## 🧑🏻‍💻 How to evaluate
You can evaluate MLMs (BERT, RoBERTa and ALBERT) on AULA, AUL, [CP score(CPS)](https://www.aclweb.org/anthology/2020.emnlp-main.154/) and [SS score(SSS)](https://arxiv.org/abs/2004.09456)-intrasentence on CP and SS datasets with following command. You also can specify pre-trained MLM path using `--model`.
```
python evaluate.py --data [cp, ss] --output /Your/output/path --model [bert, roberta, albert] --method [aula, aul, cps, sss]
```

## 📜 License
See the LICENSE file
