'''
Script to abstract away training, fusing, conversion, and server functionality.
'''

import argparse
import subprocess

def start_llama_server(gguf_model_name="ggml-model-q8_0.gguf", model_path="../models/lora_fused_model/"):
    '''
    Loads the NN and starts the llama.cpp server
    '''
    subprocess.run(["llama_server/server", "-m", model_path + gguf_model_name])

def train(model="mlx_model", iters=500, batch_size=1, lora_layers=4, resume_adapter_file="adapters.npz"):
    '''
    Invokes the LLM training script. resume_adapter_file is meant for continuing a previously terminated training run
    '''
    if resume_adapter_file is None:
        subprocess.run([
            "python3", "mlx_train/lora.py",
            "--model", model,
            "--train", "--iters", str(iters),
            "--batch-size", str(batch_size),
            "--lora-layers", str(lora_layers)
        ])
    else:
            subprocess.run([
            "python3", "mlx_train/lora.py",
            "--model", model,
            "--train", "--iters", str(iters),
            "--batch-size", str(batch_size),
            "--lora-layers", str(lora_layers),
            "--resume-adapter-file", resume_adapter_file
        ])

def fuse_lora_layers(model="mlx_model"):
    subprocess.run(["python3", "mlx_train/fuse.py", "--model", model, "--de-quantize"])

def convert_to_gguf(outtype="q8_0"):
    subprocess.run(["python3", "mlx_train/convert.py", "../models/lora_fused_model", "--outtype", outtype])

def get_base_model(huggingface_model):
    subprocess.run(["python3", "get_base_model/convert.py", "--hf-path",huggingface_model,"--mlx-path", "../models/mlx_model/", "-q"])
def main():
    parser = argparse.ArgumentParser(description="Script to abstract away training, fusing, conversion, and server functionality")
    parser.add_argument("action", choices=["train", "convert", "start_server", "get_base_model"], help="Action to perform: train, convert, get_base_model, or start_server")
    parser.add_argument("--model", default="mlx_model", help="Model name (default: mlx_model)")
    parser.add_argument("--iters", type=int, default=500, help="Number of iterations for training (default: 500)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training (default: 1)")
    parser.add_argument("--lora-layers", type=int, default=4, help="Number of LoRa layers (default: 4)")
    parser.add_argument("--resume-adapter-file", default=None, help="Resume adapter file for training (default: adapters.npz)")
    parser.add_argument("--gguf-model-name", default="ggml-model-q8_0.gguf", help="GGUF model name for starting server (default: ggml-model-q8_0.gguf)")
    parser.add_argument("--model-path", default="../models/lora_fused_model/", help="Path to model directory for server (default: ../models/lora_fused_model/)")
    parser.add_argument("--outtype", default="q8_0", help="Select quantization level (default: q8_0)")
    parser.add_argument("--hf-path", default="mistralai/Mistral-7B-Instruct-v0.2", help="path to huggingface model (default: Mistral Instruct)")
    args = parser.parse_args()

    if args.action == "train":
        train(args.model, args.iters, args.batch_size, args.lora_layers, args.resume_adapter_file)
    elif args.action == "convert":
        fuse_lora_layers(args.model)
        convert_to_gguf(args.outtype)
    elif args.action == "start_server":
        start_llama_server(args.gguf_model_name, args.model_path)
    elif args.action == "get_base_model":
        get_base_model(args.hf_path)

if __name__ == "__main__":
    main()
