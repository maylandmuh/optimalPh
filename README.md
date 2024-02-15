# Inference
```bash
python3 code/predict.py --input_csv sequences.csv --seq_col sequence --model_fname weights/model_xgboost --output_csv sequences_scored.csv
```

# Train

```bash
python3 esm_embeddings/ml/dataloader_v1.py --input_csv data/datasets/brenda_new.csv --seq_col sequence --target_col mean_pH --output_emb data/embeddings/brenda_new_emb.npy
```

```bash
python3 code/final_train.py --models exeriments/best_models.jso --data data/embeddings/brenda_new_emb.npy --output_dir weights/ --prefix model --csv_input data/datasets/brenda_new.csv
```