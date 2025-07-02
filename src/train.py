import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
import warnings

from config import get_config, get_weights_file_path
from dataset import BilingualDataset, causal_mask
from model import build_transformer
from tokenizer import train_tokenizer
from tokenizers import Tokenizer

def get_or_build_tokenizer(config, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not tokenizer_path.exists():
        print(f"Tokenizer for {lang} not found. Training now...")
        train_tokenizer(config)
    return Tokenizer.from_file(str(tokenizer_path))

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    tokenizer_src = get_or_build_tokenizer(config, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, config['lang_tgt'])

    full_ds = BilingualDataset(
        'data/en_id_corpus.txt', 
        tokenizer_src, 
        tokenizer_tgt, 
        config['lang_src'], 
        config['lang_tgt'], 
        config['seq_len']
    )

    train_ds_size = int(0.9 * len(full_ds))
    val_ds_size = len(full_ds) - train_ds_size
    train_ds, val_ds = random_split(full_ds, [train_ds_size, val_ds_size])

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config['seq_len'],
        config['seq_len'],
        d_model=config['d_model']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    initial_epoch = 0
    if config['preload'] == 'latest':
        pass

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_filename)
        print(f"Model saved to {model_filename}")

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)