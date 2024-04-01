# Apple-Silicon-NN-Trainer
Train Neural Networks natively on Apple Silicon! Take advantage of Apple Silicon's Unified Memory using for lightning fast neural net training! 

## Quick Start

1: Install dependencies: `pip3 install mlx`

2: Download base model from Huggingface: `python3  src/mlx_train/convert.py --hf-path mistralai/Mistral-7B-Instruct-v0.2 -q`

3: Train Model! `python3 main.py train`

4: Convert to GGUF format: `python3 main.py convert`

5: Start Server: `python3 main.py start_server`
