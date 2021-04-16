# Evaluate bias in MLM

Code for the paper: "Unmasking the Mask -- Evaluating Social Biases in Masked Language Models". If you use any part of this work, please cite the following citation:
```
@article{Kaneko:AUL:2021,
    title={Unmasking the Mask -- Evaluating Social Biases in Masked Language Models},
    author={Masahiro Kaneko and Danushka Bollegala},
    journal={arXiv},
    year={2021}
}
```
## 🛠 Setup

You can install all required packages with following.
```
pip install -r requirements.txt
```

You can downlaod [CP](https://github.com/nyu-mll/crows-pairs) and [SS](https://github.com/moinnadeem/StereoSet) datasets and preprocess them with following.
```
wget -O data/cp.csv https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv
wget -O data/ss.json https://raw.githubusercontent.com/moinnadeem/StereoSet/master/data/dev.json
python -u preprocess.py --input crows_pairs --output data/paralled_cp.json
python -u preprocess.py --input stereoset --output data/paralled_ss.json
```


## 🧑🏻‍💻 How to evaluate
You can evaluate MLMs (BERT, RoBERTa and ALBERT) on AULA, AUL, [CPS](https://www.aclweb.org/anthology/2020.emnlp-main.154/) and [SSS](https://arxiv.org/abs/2004.09456)-intrasentence with following command.
```
python evaluate.py --data [cp, ss] --output /Your/output/path --model [bert, roberta, albert] --method [aula, aul, cps, sss]
```

## 📜 License
See the LICENSE file
