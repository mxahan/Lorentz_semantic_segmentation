'''
code for normalized filter loss visualization 

https://arxiv.org/abs/1712.09913


'''

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ----------------------------
# 0) Utility: filter-normalized directions (Li et al., 2018)
# ----------------------------
@torch.no_grad()
def get_filter_normalized_directions(model, module_name=None, seed=42):
    """
    Generate filter-normalized random directions for loss surface visualization.
    
    Args:
        model (nn.Module): PyTorch model.
        module_name (str or None): If None, perturb all layers.
                                   If string, only perturb layers whose name contains this substring.
        seed (int): Random seed for reproducibility.

    Returns:
        list of torch.Tensor: Normalized directions for each selected parameter.
    """
    g = torch.Generator(device=next(model.parameters()).device)
    g.manual_seed(seed)

    directions = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if (module_name is None) or (module_name in name):
            # sample random direction
            d = torch.randn(p.shape, device=p.device, dtype=p.dtype, generator=g)

            # filter normalization (per filter/row)
            if d.ndim >= 2:
                flat = d.view(d.size(0), -1)
                d = (flat / flat.norm(dim=1, keepdim=True)).view_as(p)
            else:
                d = d / d.norm()

            directions.append(d)
        else:
            # fill with zeros so parameter count matches
            directions.append(torch.zeros_like(p))

    return directions

# ----------------------------
# 1) Apply a 2D perturbation (alpha, beta) for multiple modules
# ----------------------------
@torch.no_grad()
def perturb_model(base_model: nn.Module,
                  module_names,  # list of module names or None
                  dirs1, dirs2,
                  alpha: float, beta: float) -> nn.Module:
    """
    Returns a deep-copied model where parameters are perturbed by alpha*dirs1 + beta*dirs2.
    If module_names=None, perturb ALL layers.
    """
    model = copy.deepcopy(base_model)

    if module_names is None:
        # Apply to all parameters
        for p, d1, d2 in zip(model.parameters(), dirs1, dirs2):
            if p.requires_grad and d1 is not None and d2 is not None:
                p.add_(alpha * d1 + beta * d2)
    else:
        # Apply to all selected modules
        idx = 0
        for module_name in module_names:
            submod = dict(model.named_modules())[module_name]
            for p in submod.parameters():
                if p.requires_grad:
                    d1, d2 = dirs1[idx], dirs2[idx]
                    if d1 is not None and d2 is not None:
                        p.add_(alpha * d1 + beta * d2)
                    idx += 1
    return model

# ----------------------------
# 2) Evaluate loss on a 2D grid for multiple modules
# ----------------------------
def loss_surface_for_modules(model: nn.Module,
                             module_names,  # list of str or None
                             x: torch.Tensor,
                             labels: torch.Tensor,
                             grid_size: int = 21,
                             span: float = 1.0,
                             seed1: int = 123, seed2: int = 456):
    """
    Computes a grid Z of losses by perturbing the specified modules along two
    independent filter-normalized directions.
    If module_names=None, perturbs ALL parameters in the model.
    """
    device = next(model.parameters()).device
    model.eval()

    # Build directions for all selected modules
    dirs1, dirs2 = [], []
    if module_names is None:
        dirs1 = get_filter_normalized_directions(model, seed=seed1)
        dirs2 = get_filter_normalized_directions(model, seed=seed2)
    else:
        for module_name in module_names:
            submod = dict(model.named_modules())[module_name]
            dirs1.extend(get_filter_normalized_directions(submod, seed=seed1))
            dirs2.extend(get_filter_normalized_directions(submod, seed=seed2))

    alphas = np.linspace(-span, span, grid_size)
    betas = np.linspace(-span, span, grid_size)
    Z = np.zeros((grid_size, grid_size), dtype=np.float64)

    with torch.no_grad():
        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                perturbed = perturb_model(model, module_names, dirs1, dirs2, a, b)
                perturbed.eval()
                
                # modify the loss function as necessary
                # for hyperbolic MERU
                try:
                    out = perturbed(x, labels)
                    loss_val = out["logging"]['supervised_loss'].item() if isinstance(out, dict) and "loss" in out else float(out)
                
                # for baseline model custom deeplab
                except:
                    outputs = perturbed(x)
                    outputs = outputs['logits']
                    outputs = torch.clamp(outputs, -10, 15)
                    loss = nn.functional.cross_entropy(outputs, labels,  reduction="mean")
                    loss_val = loss.item()
                
                
                
                Z[i, j] = loss_val

    return alphas, betas, Z

# ----------------------------
# 3) Plot helper
# ----------------------------
def plot_loss_contour_filled(alphas, betas, Z, title: str):
    A, B = np.meshgrid(alphas, betas)
    plt.figure(figsize=(7, 6))
    cp = plt.contourf(A, B, Z.T, levels=30)
    plt.colorbar(cp, label="Loss")
    plt.xlabel("alpha")
    plt.ylabel("beta")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_loss_contour_lines(alphas, betas, Z, title: str, filename: str, cmap = 'Accent'):
    A, B = np.meshgrid(alphas, betas)
    plt.figure(figsize=(7, 6))
    # line-based contour plot
    cp = plt.contour(A, B, Z.T, levels=30, cmap=cmap, vmin=0, vmax=16)
    plt.clabel(cp, inline=True, fontsize=8)  # add labels on contour lines
    # plt.xlabel("alpha")
    # plt.ylabel("beta")
    # plt.title(title)
    plt.tight_layout()
    plt.savefig('figure/'+filename+'.svg', format="svg", dpi=300)
    plt.show()

# ----------------------------
# Example usage:
# ----------------------------
# Perturb multiple layers
# layers_to_plot = ['backbone']
# a, b, Z_all = loss_surface_for_modules(model, layers_to_plot, inputs, labels, grid_size=15, span=0.2)
# plot_loss_contour(a, b, Z_all, f"Loss Surface — {layers_to_plot}")