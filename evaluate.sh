model=$1
method=$2
echo $model $method

mkdir -p data
mkdir -p result

wget -O data/cp.csv https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv
wget -O data/ss.json https://raw.githubusercontent.com/moinnadeem/StereoSet/master/data/dev.json

python -u preprocess.py --input crows_pairs --output data/paralled_cp.json
python -u preprocess.py --input stereoset --output data/paralled_ss.json

python -u evaluate.py --input data/paralled_cp.json --output result/aul_cp_${model}.txt --model ${model} --method ${method}
python -u evaluate.py --input data/paralled_ss.json --output result/aul_ss_${model}.txt --model ${model} --method ${method}
