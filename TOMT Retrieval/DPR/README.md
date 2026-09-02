
# Environment Setup

```bash
pyenv virtualenv 3.7.10 tomt-kir-dpr
echo "tomt-kir-dpr" > ./.python-version
# create a sym link so code can be accessed
ln -s ../tomt ./tomt
#pip install torch==1.8.1
# pip install pytrec_eval
cd multihop_dense_retrieval 
bash setup.sh
```

# Prepare the data for DPR

```bash 

mkdir -p dataset/
unzip ../data_release/Books.zip -d dataset
unzip ../data_release/Movies.zip -d dataset
srun python create_data_dpr.py --root dataset/ --dataset Movies --file_path dataset/Movies/DPR/
srun python create_data_dpr.py --root dataset/ --dataset Books --file_path dataset/Books/DPR/

python create_data_faiss_index.py --root dataset/ --dataset Movies --file_path dataset/Movies/DPR/
python create_data_faiss_index.py --root dataset/ --dataset Books --file_path dataset/Books/DPR/

python inference_format_test.py --root dataset/ --dataset Movies --file_path dataset/Movies/DPR/
python inference_format_test.py --root dataset/ --dataset Books --file_path dataset/Books/DPR/
```




# Train DPR
```bash

CUDA_VISIBLE_DEVICES=0,1 srun -p gpu --time=24:00:00 --gres=gpu:volta:2 --mem=150G python -u multihop_dense_retrieval/mdr/retrieval/train_single.py \
    --do_train \
    --prefix dpr \
    --predict_batch_size 125 \
    --model_name roberta-base \
    --train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --accumulate_gradients 1 \
    --learning_rate 2e-5 \
    --output_dir ./models/Movies/DPR \
    --train_file ./dataset/Movies/DPR/train_dpr.json \
    --predict_file ./dataset/Movies/DPR/validation_dpr.json \
    --seed 16 \
    --eval-period -1 \
    --max_c_len 512 \
    --max_q_len 512 \
    --warmup-ratio 0.1 \
    --shared-encoder \
    --num_train_epochs 25

# grab the model name! e.g models/Movies/DPR/10-25-2022/dpr-seed16-bsz4-fp16False-lr2e-05-decay0.0-warm0.1-roberta-base
ls -ld ./models/Movies/DPR/*/*
MOVIES_MODEL_NAME=<enter model name>

CUDA_VISIBLE_DEVICES=0,1 srun -p gpu --time=24:00:00 --gres=gpu:volta:2 --mem=150G python -u multihop_dense_retrieval/mdr/retrieval/train_single.py \
    --do_train \
    --prefix dpr \
    --predict_batch_size 125 \
    --model_name roberta-base \
    --train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --accumulate_gradients 1 \
    --learning_rate 2e-5 \
    --output_dir ./models/Books/DPR \
    --train_file ./dataset/Books/DPR/train_dpr.json \
    --predict_file ./dataset/Books/DPR/validation_dpr.json \
    --seed 16 \
    --eval-period -1 \
    --max_c_len 512 \
    --max_q_len 512 \
    --warmup-ratio 0.1 \
    --shared-encoder \
    --num_train_epochs 25


# grab the model name! e.g models/Books/DPR/10-25-2022/dpr-seed16-bsz4-fp16False-lr2e-05-decay0.0-warm0.1-roberta-base
ls -ld ./models/Books/DPR/*/*
BOOKS_MODEL_NAME=<enter model name>

```

# Build FAISS index

Start executing this in the `multihop_dense_retrieval` folder:
```bash

cp scripts/encode_corpus.py ./
# create Movies index
srun -p gpu --time=01:00:00 --gres=gpu:2 --mem=64G python encode_corpus.py \
    --do_predict \
    --predict_batch_size 100 \
    --model_name roberta-base \
    --predict_file ../dataset/Movies/DPR/id2doc2.json \
    --init_checkpoint ../$MOVIES_MODEL_NAME/checkpoint_best.pt    \
    --embed_save_path ../dataset/Movies/DPR/index \
    --max_c_len 512   \
    --num_workers 20

# create Books index
srun -p gpu --time=01:00:00 --gres=gpu:2 --mem=64G python encode_corpus.py \
    --do_predict \
    --predict_batch_size 100 \
    --model_name roberta-base \
    --predict_file ../dataset/Books/DPR/id2doc2.json \
    --init_checkpoint ../$BOOKS_MODEL_NAME/checkpoint_best.pt    \
    --embed_save_path ../dataset/Books/DPR/index \
    --max_c_len 512   \
    --num_workers 20

```

    
# Inference on test set

Execute this in the `multihop_dense_retrieval` folder:


```bash
# copy/overwrite the given file with our file that produces predictions.json file
cp ../eval_retrieval.py mdr/retrieval/eval_retrieval.py

srun -p gpu --time=01:00:00 --gres=gpu:2 --mem=64G python -u mdr/retrieval/eval_retrieval.py ../dataset/Movies/DPR/qas_test.json ../dataset/Movies/DPR/index.npy ../dataset/Movies/DPR/index/id2doc.json ../$MOVIES_MODEL_NAME/checkpoint_best.pt --batch-size 300 --model-name roberta-base --shared-encoder --save-pred ../$MOVIES_MODEL_NAME/predictions.json --topk 1000

srun -p gpu --time=01:00:00 --gres=gpu:2 --mem=64G python -u mdr/retrieval/eval_retrieval.py ../dataset/Books/DPR/qas_test.json ../dataset/Books/DPR/index.npy ../dataset/Books/DPR/index/id2doc.json ../$BOOKS_MODEL_NAME/checkpoint_best.pt --batch-size 300 --model-name roberta-base --shared-encoder --save-pred ../$BOOKS_MODEL_NAME/predictions.json --topk 1000

# prf
srun -p gpu --time=01:00:00 --gres=gpu:2 --mem=64G python -u mdr/retrieval/eval_retrieval_prf.py ../dataset/Movies/DPR/qas_test.json ../dataset/Movies/DPR/index.npy ../dataset/Movies/DPR/index/id2doc.json ../$MOVIES_MODEL_NAME/checkpoint_best.pt --batch-size 300 --model-name roberta-base --shared-encoder --save-pred ../$MOVIES_MODEL_NAME/predictions.json --topk 1000 --prf-depth 5 --prf-method rocchio --rocchio-topk 5 --rocchio-alpha 0.4 --rocchio-beta 0.6

# evaluate the results!
cd ../
srun python eval_predictions_file.py --root dataset/ --dataset Movies --predictions $MOVIES_MODEL_NAME/predictions.json
srun python eval_predictions_file.py --root dataset/ --dataset Books --predictions $BOOKS_MODEL_NAME/predictions.json
```



```
export CUDA_VISIBLE_DEVICES=4,5
python -u multihop_dense_retrieval/mdr/retrieval/train_single.py \
    --do_train \
    --prefix dpr \
    --predict_batch_size 125 \
    --model_name roberta-base \
    --train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --accumulate_gradients 1 \
    --learning_rate 2e-5 \
    --output_dir ./models/Movies/DPR \
    --train_file ./dataset/Movies/DPR/train_dpr.json \
    --predict_file ./dataset/Movies/DPR/validation_dpr.json \
    --seed 16 \
    --eval-period -1 \
    --max_c_len 512 \
    --max_q_len 512 \
    --warmup-ratio 0.1 \
    --shared-encoder \
    --num_train_epochs 25


ls -ld ./models/Movies/DPR/*/*
MOVIES_MODEL_NAME=models/Movies/DPR/02-28-2023/dpr-seed16-bsz4-fp16False-lr2e-05-decay0.0-warm0.1-roberta-base
python encode_corpus.py \
    --do_predict \
    --predict_batch_size 100 \
    --model_name roberta-base \
    --predict_file ../dataset/Movies/DPR/id2doc2.json \
    --init_checkpoint ../$MOVIES_MODEL_NAME/checkpoint_best.pt    \
    --embed_save_path ../dataset/Movies/DPR/index \
    --max_c_len 512   \
    --num_workers 20


## book 

python -u multihop_dense_retrieval/mdr/retrieval/train_single.py \
    --do_train \
    --prefix dpr \
    --predict_batch_size 125 \
    --model_name roberta-base \
    --train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --accumulate_gradients 1 \
    --learning_rate 2e-5 \
    --output_dir ./models/Books/DPR \
    --train_file ./dataset/Books/DPR/train_dpr.json \
    --predict_file ./dataset/Books/DPR/validation_dpr.json \
    --seed 16 \
    --eval-period -1 \
    --max_c_len 512 \
    --max_q_len 512 \
    --warmup-ratio 0.1 \
    --shared-encoder \
    --num_train_epochs 25

```