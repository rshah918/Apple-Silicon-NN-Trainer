# Apple-Silicon-NN-Trainer
Train Neural Networks natively on Apple Silicon! Take advantage of Apple Silicon's Unified Memory using for lightning fast neural net training! 

## Quick Start

0: Set up virtual env:

`python3 -m venv venv ` 

`source .venv/bin/activate `

1: Install dependencies: 
`pip3 install sentencepiece`
`pip3 install numpy`
`pip3 install mlx`

2: Download base model from Huggingface: mistralai/Mistral-7B-Instruct-v0.2 and store in `/models`

3: Train Model! `python3 main.py train`

4: Convert to GGUF format: `python3 main.py convert`

5: Start Server: `python3 main.py start_server`
