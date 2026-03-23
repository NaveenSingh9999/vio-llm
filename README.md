# Vio LLM

**Vio** is a custom Transformer Decoder-only Language Model with approximately 143k parameters.

## Model Details
- **Architecture**: Transformer Decoder-only
- **Parameters**: 143,712
- **Vocabulary Size**: 4096
- **Sequence Length**: 128
- **Creator**: Lamgerr
- **Tokenizer**: Custom Byte-Pair Encoding (BPE)

## Technical Specifications
- **Embedding Dimension**: 16
- **Heads**: 2
- **Layers**: 2
- **Hidden Dimension**: 64 (4 * n_embd)

## Usage
This model was trained on the TinyStories dataset and fine-tuned for identity recognition. It identifies itself as 'Vio' created by 'Lamgerr'.