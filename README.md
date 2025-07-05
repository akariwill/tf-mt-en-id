<div align="center">
<a href="https://akariwill.github.io/tf-mt-en-id/">
  <img src="https://github.com/akariwill/Otaku/blob/main/assets/images/akari.jpg" alt="logo" width="180" style="border-radius: 50%;"/>
</a>
</div>
---

# Transformer Machine Translation (EN-ID) from Scratch

This project is an implementation of a Machine Translation (MT) model from scratch using the original Transformer architecture ("Attention Is All You Need") with PyTorch. The model is trained to translate from English to Indonesian.

## Architecture

This model implements the core components of the Transformer:
-   **Input Embeddings & Positional Encoding**: To represent input tokens and their positions.
-   **Encoder**: Consists of N blocks, each with a Multi-Head Self-Attention layer and a Feed-Forward Network.
-   **Decoder**: Consists of N blocks, each with a Masked Multi-Head Self-Attention layer, a Cross-Attention layer (with the Encoder's output), and a Feed-Forward Network.
-   **Projection Layer**: To map the decoder's output to vocabulary probabilities.

## Project Structure
```
transformer-mt-en-id/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   └── en_id_corpus.txt
├── weights/
└── src/
    ├── config.py
    ├── dataset.py
    ├── model.py
    ├── tokenizer.py
    ├── train.py
    └── translate.py
```

## Setup & Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/akariwill/tf-mt-en-id.git
    cd transformer-mt-en-id
    ```

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## How to Use

The process consists of three main steps: training the tokenizer, training the model, and performing inference (translation).

### 1. Train the Tokenizer (Optional, runs automatically)
The training script will automatically create the tokenizers if they don't exist. However, you can also run it manually:
```bash
python src/tokenizer.py
```
This will create `tokenizer_en.json` and `tokenizer_id.json` files in the project's root folder.

### 2. Train the Model

To start the training process:
```bash
python src/train.py
```

This process will:
- Load the data and tokenizers.
- Build the Transformer model.
- Train the model for the number of epochs specified in `src/config.py`.
- Save a model checkpoint in the `weights/` folder after each epoch.

**Note**: The dataset used is very small, so the model will overfit quickly. The main goal is to demonstrate the architecture and workflow. For good results, a much larger dataset (e.g., from [OPUS](https://opus.nlpl.eu/)) is required.

### 3. Translate New Sentences

After the model is trained, use the `translate.py` script to translate sentences.

You can change the input sentence directly within the `src/translate.py` file. The script will load the trained model and generate the translation.

**Important**: Make sure to adjust the model weight filename loaded in `src/translate.py` to match the best or latest epoch from your training run.

```python
# Example in src/translate.py
model_filename = get_weights_file_path(config, "199") 
```

### Configuration
All hyperparameters and paths can be configured in the `src/config.py` file.

---

### **Additional Explanations**

1.  **A Note on "Multiple Models Based on the Data"**: The initial request for this project may have been slightly ambiguous. Instead of creating several different models for various tasks (e.g., classification, NER), this project focuses on building **one solid translation model** trained on the provided data. A from-scratch Transformer implementation is already highly complex and serves as a powerful foundation for many other NLP tasks. Creating other models (like for NER) would require different architectures and task-specific annotated datasets. This project provides the best possible foundation for such endeavors.

2.  **Optimality**: The standard Transformer architecture is an excellent starting point. To achieve truly optimal, production-level results, the next steps would involve:
    *   Using a much larger dataset (millions of sentence pairs).
    *   Training for a longer duration on a powerful GPU.
    *   Implementing more advanced regularization techniques and learning rate scheduling.
    *   Using **beam search** instead of **greedy search** during inference for higher-quality translations (though greedy search is sufficient for demonstration purposes).

## Contact

Thank You for passing by!!
If you have any questions or feedback, please reach out to us at [contact@akariwill.id](mailto:mwildjrs23@gmail.com?subject=[tf-mt]%20-%20Your%20Subject).
<br>
or you can DM me on Discord `wildanjr_` or Instagram `akariwill`. (just contact me on one of these account)

## Contributing

Feel free to contribute to this project by submitting pull requests or opening issues in the repository.

---
