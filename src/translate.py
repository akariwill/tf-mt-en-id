# src/translate.py
import torch
from pathlib import Path
from tokenizers import Tokenizer

from model import build_transformer
from config import get_config, get_weights_file_path

def translate(sentence: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config()
    
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))

    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config['seq_len'],
        config['seq_len'],
        d_model=config['d_model']
    ).to(device)

    model_filename = get_weights_file_path(config, "100")
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    
    model.eval()
    with torch.no_grad():
        source = tokenizer_src.encode(sentence)
        source = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64),
            torch.tensor(source.ids, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (config['seq_len'] - len(source.ids) - 2), dtype=torch.int64)
        ]).to(device)
        source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)

        encoder_output = model.encode(source, source_mask)

        decoder_input = torch.empty(1, 1).fill_(tokenizer_tgt.token_to_id('[SOS]')).type_as(source).to(device)
        
        while decoder_input.size(1) < config['seq_len']:
            decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.int).type_as(source_mask).to(device) == 0
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
            
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)
            
            if next_word == tokenizer_tgt.token_to_id('[EOS]'):
                break
    
    return tokenizer_tgt.decode(decoder_input[0].tolist())

if __name__ == '__main__':
    input_sentence = "Data does not always equal knowledge."
    translation = translate(input_sentence)
    print(f"Input: {input_sentence}")
    print(f"Translation: {translation}")

    input_sentence_2 = "We categorize and classify data."
    translation_2 = translate(input_sentence_2)
    print(f"\nInput: {input_sentence_2}")
    print(f"Translation: {translation_2}")