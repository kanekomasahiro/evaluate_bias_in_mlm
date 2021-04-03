# evaluate_mlm

Code for the paper: "Unmasking the Mask -- Evaluating Social Biases in Masked Language Models". If you use any part of this work, make sure you include the following citation:
```
@article{Kaneko:AUL:2021,
    title={Unmasking the Mask -- Evaluating Social Biases in Masked Language Models},
    author={Masahiro Kaneko and Danushka Bollegala},
    journal={arXiv},
    year={2021}
}
```
## Requirements
```
pip install -u requirements.txt
```

## How to evaluate
You can evaluate MLMs (BERT, RoBERTa and ALBERT) on AUL, [CPS](https://www.aclweb.org/anthology/2020.emnlp-main.154/) and [SSS](https://arxiv.org/abs/2004.09456)-intrasentence with following a command.
```
./evaluate.sh [bert, roberta, albert] [aul, cps, sss]
```

## License
See the LICENSE file