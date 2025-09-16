
with torch.no_grad():
        outputs = model.backbone(inputs)
        hidden_states = outputs['out']
        re_feats = hidden_states.permute(0,2,3,1).reshape(-1, 2048).cpu()
        

# delta hyperbolicity computations


def delta_hyp_torch(dismat):
    p = 0
    row = dismat[p, :][None, :]
    col = dismat[:, p][:, None]
    XY_p = 0.5 * (row + col - dismat)
    maxmin = torch.minimum(XY_p[:, :, None], XY_p[None, :, :]).max(1).values
    return (maxmin - XY_p).max()
import numpy as np

idx = np.random.choice(len(re_feats), 2000)
all_features_small = re_feats[idx]
# dists = distance_matrix(all_features_small, all_features_small)
dists = torch.cdist(all_features_small, all_features_small)
delta = delta_hyp_torch(dists)
diam = torch.max(dists)
del_rel =  2*delta/diam
print('delta relative: ', del_rel)
c = (0.144 / del_rel) ** 2
print('c value: ', c)


#%%

import torch

print("PyTorch δ-hyperbolicity computation (matches 1904.02239)")

@torch.no_grad()
def pairwise_euclid(x: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Euclidean distance matrix for x: (n,d)."""
    x2 = (x**2).sum(-1, keepdim=True)
    D2 = x2 + x2.T - 2*(x @ x.T)
    D2.clamp_(min=0)
    return D2.sqrt()

@torch.no_grad()
def gromov_product_matrix(D: torch.Tensor, r: int) -> torch.Tensor:
    """Compute Gromov product matrix at basepoint r."""
    Dr = D[:, r].unsqueeze(0)         # (1,n)
    A = (Dr.T + Dr - D) * 0.5         # (n,n)
    return A

@torch.no_grad()
def max_min_product(A: torch.Tensor, k_chunk: int = 512) -> torch.Tensor:
    """Memory-efficient max-min product C[i,j] = max_k min(A[i,k], A[k,j])."""
    n = A.shape[0]
    device = A.device
    C = torch.full((n, n), -torch.inf, device=device, dtype=A.dtype)

    for start in range(0, n, k_chunk):
        stop = min(start + k_chunk, n)
        left  = A[:, start:stop].unsqueeze(2)   # (n, b, 1)
        right = A[start:stop, :].unsqueeze(0)   # (1, b, n)
        cand = torch.minimum(left, right)       # (n, b, n)
        C = torch.maximum(C, cand.amax(dim=1))
        del left, right, cand
    return C

@torch.no_grad()
def delta_r_from_D(D: torch.Tensor, r: int, k_chunk: int = 2048) -> float:
    """Compute δ_r at a given basepoint r."""
    A = gromov_product_matrix(D, r)
    C = max_min_product(A, k_chunk=k_chunk)
    delta_r = (C - A).max().item()
    return max(delta_r, 0.0)

@torch.no_grad()
def delta_star_2approx_from_D(D: torch.Tensor, r: int | None = None, k_chunk: int = 2048) -> float:
    """2-approximation of δ* using one basepoint."""
    n = D.shape[0]
    if r is None:
        # Pick farthest-from-median heuristic
        s = D.sum(dim=1)
        m = int(s.argmin().item())
        r = int(D[m].argmax().item())
    return delta_r_from_D(D, r, k_chunk=k_chunk)

@torch.no_grad()
def delta_from_points(x: torch.Tensor, exact: bool = False, k_chunk: int = 2048, base: int | None = None):
    """Compute δ (exact or 2-approx) from points x: (n,d)."""
    D = pairwise_euclid(x)
    return (delta_star_2approx_from_D(D, r=base, k_chunk=k_chunk) if not exact
            else delta_star_exact_from_D(D, k_chunk=k_chunk))

@torch.no_grad()
def relative_delta_from_D(D: torch.Tensor, delta_val: float) -> float:
    """Compute relative δ_hyperbolicity: δ_rel = 2 * δ / diam."""
    diam = float(D.max().item())
    return (2 * delta_val / diam) if diam > 0 else 0.0


# usage of the earlier section.
from math import ceil
# 2-approximation δ (fast)
delta_val = delta_from_points(all_features_small, exact=False, k_chunk=512)  
print("δ (2-approx):", delta_val)

D = pairwise_euclid(all_features_small)               # distance matrix (2000x2000)
delta_rel = relative_delta_from_D(D, delta_val)
