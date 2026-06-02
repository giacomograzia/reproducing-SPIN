import torch
import torch.nn as nn
import re
import os
import pickle
import argparse
import numpy as np

from utils import get_llm
from utils import get_loaders
from functools import reduce
from transformers import AutoTokenizer, AutoModelForCausalLM


class ActLinear(nn.Module):
    """
    drop in replacement of nn.Linear
    """

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.activation_norms = torch.zeros(
            [base.in_features], device=self.base.weight.device, requires_grad=False
        )
        
        # Reproducibility study: grad×act accumulator per input feature (neuron)
        self.gxA = torch.zeros(
            [base.in_features], device=self.base.weight.device, requires_grad=False
        )
        
        self.n_samples = 0
        self.record_activation = True
        
        # Reproducibility study: only enabled for the new method
        self.record_gxA = False


    def clear_act_buffer(self):
        self.activation_norms.fill_(0.0)        
        self.n_samples = 0
        
        # Reproducibility study:
        self.gxA.fill_(0.0)  

    def forward(self, x):

        if self.record_activation:
            if hasattr(self, "mask") and self.mask is not None:
                x_ = x[self.mask]
            else:
                x_ = x

            bs = x_.nelement() // x_.shape[-1]
            self.activation_norms = self.activation_norms * (
                self.n_samples / (self.n_samples + bs)
            ) + (x_ * x_).view(-1, x_.shape[-1]).sum(dim=0) * (
                1.0 / (self.n_samples + bs)
            )
            self.n_samples += bs
            
        # Reproducibility study: attach hook to capture grad wrt this layer's input features
        if self.record_gxA:
            # ensure x has grad
            if not x.requires_grad:
                x = x.detach().requires_grad_(True)

            def _hook(grad):
                if hasattr(self, "mask") and self.mask is not None:
                    x_local = x[self.mask]
                    g_local = grad[self.mask]
                else:
                    x_local = x
                    g_local = grad

                bs = x_local.nelement() // x_local.shape[-1]
                contrib = (x_local * g_local).abs().view(-1, x_local.shape[-1]).sum(dim=0)

                # running mean (same style as activation_norms)
                self.gxA = self.gxA * (self.n_samples / (self.n_samples + bs)) + contrib * (
                    1.0 / (self.n_samples + bs)
                )
                # if activation recording is off, maintain n_samples here
                if not self.record_activation:
                    self.n_samples += bs

            x.register_hook(_hook)

        out = self.base(x)
        return out


class no_act_recording:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        for name, module in self.model.named_modules():
            if isinstance(module, ActLinear):
                module.record_activation = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, module in self.model.named_modules():
            if isinstance(module, ActLinear):
                module.record_activation = True
                
# Reproducibility study: context manager for grad×act recording
class record_gxA:
    def __init__(self, model, name_filter_fn=None):
        self.model = model
        self.name_filter_fn = name_filter_fn

    def __enter__(self):
        for name, m in self.model.named_modules():
            if isinstance(m, ActLinear) and (self.name_filter_fn is None or self.name_filter_fn(name)):
                m.record_gxA = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, m in self.model.named_modules():
            if isinstance(m, ActLinear) and (self.name_filter_fn is None or self.name_filter_fn(name)):
                m.record_gxA = False


def revert_Act_to_Linear(model):
    """
    Reverts ActLinear modules back to their original nn.Linear layers.
    """
    for name, module in model.named_modules():
        if isinstance(module, ActLinear):
            # Extract the base nn.Linear module from ActLinear
            linear_module = module.base
            # Navigate to the parent module of the ActLinear module
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            print(f"Reverting {name}, parent: {parent_name}")
            parent_module = (
                model
                if parent_name == ""
                else reduce(getattr, parent_name.split("."), model)
            )
            # Replace the ActLinear module with the extracted nn.Linear module
            setattr(parent_module, name.split(".")[-1], linear_module)

    return model


def make_Act(model, verbose=False):
    replace_map = dict()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            replace_map[name] = ActLinear(module)

    for name, module in model.named_modules():
        if verbose:
            print("current:", name)
        for k, v in replace_map.items():
            k_ = k.split(".")
            name_prefix, name_suffix = ".".join(k_[:-1]), k_[-1]
            if name_prefix == "":  # outer layer
                if name == name_suffix:
                    if verbose:
                        print(" not modifying ", name_suffix)
                    # setattr(model, name_suffix, v)
            elif name == name_prefix:
                if verbose:
                    print("    modifying ", name_suffix, "inside", name)
                setattr(module, name_suffix, v)
    return model


def compute_importance_score(
    args,
    model,
    tokenizer,
    device=torch.device("cuda:0"),
):
    model = make_Act(model, verbose=False)

    dataloader, _ = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
        disentangle=True,
    )
    print("dataset loading complete")

    num_hidden_layers = model.config.num_hidden_layers
    for layer in range(num_hidden_layers):
        layer_filter_fn = (
            lambda x: f"layers.{layer}." in x
        )

        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                module.clear_act_buffer()

        if args.locating_method == "random":
            _compute_and_save(args, model, name_filter_fn=layer_filter_fn)
            continue  
        
        if args.locating_method in ["wanda", "sparsegpt"]:
            with torch.no_grad():
                for batch in dataloader:
                    inp = batch[0].to(device)
                    _ = model(input_ids=inp)
            _compute_and_save(args, model, name_filter_fn=layer_filter_fn)
            continue

        if args.locating_method == "grad_x_act":
            model.zero_grad(set_to_none=True)
            model.requires_grad_(False)

            with no_act_recording(model):
                with record_gxA(model, name_filter_fn=layer_filter_fn):
                    for batch in dataloader:
                        inp, tar = batch[0].to(device), batch[1].to(device)
                        model.zero_grad(set_to_none=True)
                        loss = model(input_ids=inp, labels=tar)[0]
                        loss.backward()

            _compute_and_save(args, model, name_filter_fn=layer_filter_fn)
            continue

        saved_grad = {}
        model.zero_grad()
        model.requires_grad_(False)
        for name, module in model.named_modules():  
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                print("enabling grad for ", name)
                module.base.requires_grad_(True)
                saved_grad[name] = torch.zeros_like(
                    module.base.weight, device=module.base.weight.device
                )
                module.base.zero_grad()
            
        for batch in dataloader:
            inp, tar = batch[0].to(device), batch[1].to(device)
            model.zero_grad()
            with no_act_recording(model):
                loss = model(input_ids=inp, labels=tar)[0]
            loss.backward()
            for name, module in model.named_modules():
                if layer_filter_fn(name) and isinstance(module, ActLinear):
                    saved_grad[name] += module.base.weight.grad.abs()   

        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                module.base.weight.grad.copy_(saved_grad[name])

        _compute_and_save(
            args,
            model,
            name_filter_fn=layer_filter_fn,
        )

    model = revert_Act_to_Linear(model)
    model.zero_grad()  # freeze gradient to save cuda memory


def _compute_and_save(
    args,
    model,
    name_filter_fn=None,
):
    dataset = args.dataset
    for name, module in model.named_modules():
        if name_filter_fn is not None and not name_filter_fn(name):
            continue
        if isinstance(module, ActLinear):
            m = re.search(r"layers\.(\d+)\.", name)
            i = int(m.group(1)) if m else 0

            # magnitude = torch.abs(module.base.weight.data)  #  weight W

            # act = module.base.weight.grad.abs()  #  dW

            # importance_score = magnitude * act  # importance score


            # Reproducibility study: For implementing different neuron locating methods
            W = module.base.weight.data

            if args.locating_method == "importance":
                # W_ij * grad_Wij L(s)
                score = W.abs() * module.base.weight.grad.abs()

            elif args.locating_method == "random":
                # random baseline
                score = torch.rand_like(W)

            elif args.locating_method == "wanda":
                # |W_ij| * ||X_j||_2
                # activation_norms stores aggregated sum of x^2 per input feature (or averaged)
                x2 = module.activation_norms  # shape [in_features]
                xnorm = torch.sqrt(x2 + 1e-8).view(1, -1)
                score = W.abs() * xnorm

            elif args.locating_method == "sparsegpt":
                # |W|^2 / diag((XX^T + λI)^-1)
                # diag inverse approx: 1/(x2 + λ) => divide by that == multiply by (x2 + λ)
                x2 = module.activation_norms  # [in_features]
                lambda_damp = 1e-2
                scale = (x2 + lambda_damp).view(1, -1)
                score = (W * W).abs() * scale

            elif args.locating_method == "grad_x_act":
                # neuron saliency per input feature: |a * dL/da|
                sal = module.gxA  # [in_features]
                # keep same 2D shape as other methods
                score = W.abs() * sal.view(1, -1)
            
            ##############

            # save the importance score
            save_folder = os.path.join( 
                args.score_dir, #--> Reproducibility study: for the shared scratch dir
                args.locating_method,
                str(args.nsamples),
                args.dataset,
                args.model.lower(),
            )
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            target_file = os.path.join(
                save_folder, f"layer_{i}_name_{name}_weight.pkl"
            )
            with open(target_file, "wb") as f:
                print(
                    f"Writing {args.locating_method} in layer {i} and name {name} with {dataset} to the file"
                )
                pickle.dump(score, f) # Reproducibility study: replaced importance_score with score 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama2-7b-chat-hf")
    parser.add_argument("--model_path", type=str, default="temp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--score_dir", type=str, default="significance_score") # Reproducibility study: added for scratch
    parser.add_argument("--locating_method", type=str, default="importance",
                    choices=["importance", "random", "wanda", "sparsegpt", "grad_x_act"])

    args = parser.parse_args()
    print(args)

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # model = get_llm(args.model) 
    model = get_llm(args.model_path) # Reproducibility study: fixed bug
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=False, trust_remote_code=True
    )
    # Compute and save the importance score
    compute_importance_score(
        args=args,
        model=model,
        tokenizer=tokenizer
    )

# Reproducibility study: this was missing in the original code
if __name__ == "__main__":
    main()