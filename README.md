# SoftPromptTuning

SoftPromptTuning is an implementation of soft-prompt tuning for NLP tasks, specifically leveraging the T5 model for conditional text generation. This project integrates advanced techniques like Flash Attention, mixed precision training, and KV caching to optimize both performance and efficiency during model training and inference.

## Code Overview

### 1. Setting up the Environment
- Import essential libraries such as `torch`, `transformers`, `nltk`, and `datasets`.
- Initialize the T5 model and tokenizer to facilitate efficient text processing and conditional generation.

### 2. Dataset Preparation
- Load and preprocess the IMDb dataset using the `datasets` library.
- Apply text normalization, including converting text to lowercase and removing punctuation to ensure consistency.
- Define a custom `IMDBDataset` class to structure the dataset for training, handling tokenization, and formatting the data for model input.

### 3. Model Preparation
- Add a `SoftPrompt` layer to the model, which generates learnable prompt embeddings tailored for the T5 architecture.
- Freeze T5 model parameters to focus training on the added SoftPrompt layer, optimizing efficiency while preserving model capacity.
- Integrate the SoftPrompt layer into the model’s encoder.

### 4. Optimization Techniques
Several optimizations enhance training speed and reduce resource usage:

- **Flash Attention**: Replaces the standard attention mechanism with a more efficient approach, speeding up computation.
- **Mixed Precision Training**: Enables mixed precision for memory efficiency, improving training speed and allowing larger batch sizes.
- **KV-Caching**: Implements key-value caching to speed up auto-regressive text generation, particularly useful for long sequences.

### 5. Training and Evaluation
- Define `train_step` and `eval_step` functions for streamlined training and evaluation, with integrated metric tracking for performance monitoring.
- Use `train_and_evaluate()` to run training across multiple epochs, utilizing optimizations to facilitate model convergence and generalization.

### Optional: Enable Flash Attention
To further optimize attention computation, Flash Attention can be enabled. To activate it, add the following code snippet:

```python
for layer in model.encoder.block:
    layer.layer[0].SelfAttention = T5FlashAttention(model.config)
for layer in model.decoder.block:
    layer.layer[0].SelfAttention = T5FlashAttention(model.config)
    layer.layer[1].EncDecAttention = T5FlashAttention(model.config)
```


## Credits

### Author: Seyed Morteza Mahdavi
### Position: Machine Learning Engineer at Behbahan University of Medical Sciences
### GitHub: github.com/MortezaMahdaviMortazavi
### Contact: s.morteza.mahdavi.mortazavi@gmail.com
