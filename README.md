# Environment
To create the environment with conda:
```bash
conda create -n esm_env
conda activate esm_env

conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda install fire fair-esm xgboost -y
pip install fairscale
```

# Inference

You can download the models from the google drive[Here](https://drive.google.com/drive/folders/1CRKzq3DGFjlTH3MzTZ8a3AQaB23YrfCn?usp=sharing)

```bash
python3 code/predict.py --input_csv sequences.csv --seq_col sequence --model_fname weights/model_xgboost --output_csv sequences_scored.csv
```

# Train

To obtain embeddings from ESM-2 protein LLM (takes ~50Mb for 10k sequences):

```bash
python3 esm_embeddings/ml/dataloader_v1.py --input_csv data/datasets/brenda_new.csv --seq_col sequence --target_col mean_pH --output_emb data/embeddings/brenda_new_emb.npy
```

Train your model.
JSON file containing info about your model should look as follows:
```json
{
    "1": {
        "type": "knn",
        "params": {
            "n_neighbors": 8
        }
    }
}
```

```bash
python3 code/final_train.py --models experiments/best_models.jso --data data/embeddings/brenda_new_emb.npy --output_dir weights/ --prefix model --csv_input data/datasets/brenda_new.csv
```
