from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 200, 
        "lr": 1e-4,
        "seq_len": 128,    
        "d_model": 512,     
        "lang_src": "en",
        "lang_tgt": "id",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,     
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "datasource": 'opus_books',
        "d_ff": 2048, 
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "num_heads": 8,
        "dropout": 0.1
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)