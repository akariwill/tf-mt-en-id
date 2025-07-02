import os
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

def train_tokenizer(config):
    data_path = Path('data/en_id_corpus.txt')
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")
        
    tokenizer_path_src = Path(config['tokenizer_file'].format(config['lang_src']))
    tokenizer_path_tgt = Path(config['tokenizer_file'].format(config['lang_tgt']))

    tokenizer_src = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer_src.pre_tokenizer = Whitespace()
    trainer_src = WordPieceTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)

    tokenizer_tgt = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer_tgt.pre_tokenizer = Whitespace()
    trainer_tgt = WordPieceTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)

    def get_all_sentences(lang):
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    if lang == config['lang_src']:
                        yield parts[0]
                    else:
                        yield parts[1]

    if not tokenizer_path_src.exists():
        print(f"Training tokenizer for {config['lang_src']}...")
        tokenizer_src.train_from_iterator(get_all_sentences(config['lang_src']), trainer=trainer_src)
        tokenizer_src.save(str(tokenizer_path_src))
        print(f"Tokenizer for {config['lang_src']} saved to {tokenizer_path_src}")
    else:
        print(f"Tokenizer for {config['lang_src']} already exists.")

    if not tokenizer_path_tgt.exists():
        print(f"Training tokenizer for {config['lang_tgt']}...")
        tokenizer_tgt.train_from_iterator(get_all_sentences(config['lang_tgt']), trainer=trainer_tgt)
        tokenizer_tgt.save(str(tokenizer_path_tgt))
        print(f"Tokenizer for {config['lang_tgt']} saved to {tokenizer_path_tgt}")
    else:
        print(f"Tokenizer for {config['lang_tgt']} already exists.")

if __name__ == '__main__':
    from config import get_config
    config = get_config()
    train_tokenizer(config)