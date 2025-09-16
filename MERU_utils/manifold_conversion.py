import torch

def spatial_to_lorentz(x_spatial, c=1.0, eps=1e-6):
    """
    Convert spatial components (bs, h, w, n) to full Lorentz coords (bs, h, w, n+1)
    with curvature c > 0, numerically stable.
    """
    norm_sq = torch.sum(x_spatial**2, dim=-1, keepdim=True)
    # clamp to avoid negative values under sqrt
    norm_sq = norm_sq.clamp(min=0.0)
    time = torch.sqrt(norm_sq + 1.0 / c + eps)  # add eps for stability
    return torch.cat([time, x_spatial], dim=-1)

# -----------------------------
# Lorentz -> Poincaré
# -----------------------------
def lorentz_to_poincare(x_lorentz, c=1.0, eps=1e-6):
    """
    x_lorentz: (..., n+1) on hyperboloid, x0>0, <x,x>_L = -1/c
    Returns p: (..., n) Poincare ball coordinates, ||p|| < 1/sqrt(c)
    """
    sqrt_c = torch.sqrt(torch.tensor(c, dtype=x_lorentz.dtype, device=x_lorentz.device))
    x0 = x_lorentz[..., :1]
    xs = x_lorentz[..., 1:]
    denom = 1 + sqrt_c * x0
    return xs / denom


# -----------------------------
# Poincaré -> Lorentz
# -----------------------------
def poincare_to_lorentz(p, c=1.0, eps=1e-7):
    """
    p: (..., n), ||p|| < 1/sqrt(c)
    Returns x_lorentz: (..., n+1) on hyperboloid
    """
    sqrt_c = torch.sqrt(torch.tensor(c, dtype=p.dtype, device=p.device))
    cp2 = c * torch.sum(p**2, dim=-1, keepdim=True)
    
    # Soft clamping that preserves gradients
    cp2_safe = cp2 - (cp2 - (1 - eps)).clamp(min=0)
    
    denom = 1 - cp2_safe
    x0 = (1 + cp2_safe) / (sqrt_c * denom)
    xs = (2 * p) / (sqrt_c * denom)
    
    return torch.cat([x0, xs], dim=-1)


# -----------------------------
# Lorentz -> Klein
# -----------------------------
def lorentz_to_klein(x_lorentz, c=1.0, eps=1e-6):
    """
    x_lorentz: (..., n+1) on hyperboloid, x0>0
    Returns k: (..., n), c*||k||^2 < 1
    """
    sqrt_c = torch.sqrt(torch.tensor(c, dtype=x_lorentz.dtype, device=x_lorentz.device))
    x0 = x_lorentz[..., :1]
    xs = x_lorentz[..., 1:]
    return xs / (sqrt_c * x0)


# -----------------------------
# Klein -> Lorentz
# -----------------------------
def klein_to_lorentz(k, c=1.0, eps=1e-6):
    """
    k: (..., n) with c‖k‖² < 1
    Returns x_lorentz: (..., n+1) on hyperboloid with ⟨x,x⟩_L = -1/c
    """
    sqrt_c = torch.sqrt(torch.tensor(c, dtype=k.dtype, device=k.device))
    
    # Scale to standard Klein coordinates
    k_standard = k * sqrt_c
    
    # Ensure ‖k_standard‖² < 1
    norm_sq = torch.sum(k_standard**2, dim=-1, keepdim=True)
    norm_sq = torch.clamp(norm_sq, max=1-eps)
    
    # Correct formula
    x0 = 1 / (sqrt_c * torch.sqrt(1 - norm_sq))
    xs = k_standard * x0
    
    return torch.cat([x0, xs], dim=-1)


# -----------------------------
# Klein -> Poincaré
# -----------------------------
def klein_to_poincare(k, c=1.0, eps=1e-7):
    """
    k: (..., n) with c‖k‖² < 1
    Returns p: (..., n) with ‖p‖ < 1/√c
    """
    sqrt_c = torch.sqrt(torch.tensor(c, dtype=k.dtype, device=k.device))
    
    # Scale to standard Klein coordinates
    k_standard = k * sqrt_c
    
    # Ensure ‖k_standard‖² < 1
    kk = torch.sum(k_standard**2, dim=-1, keepdim=True)
    kk = torch.clamp(kk, max=1-eps)
    
    # Direct conversion
    denom = 1 + torch.sqrt(1 - kk)
    p_standard = k_standard / denom
    
    # Scale to Poincaré ball with radius 1/√c
    p = p_standard / sqrt_c
    return p

# -----------------------------
# Poincaré -> Klein
# -----------------------------
def poincare_to_klein(p, c=1.0, eps=1e-7):
    """
    p: (..., n), ||p|| < 1/sqrt(c)
    Returns k: (..., n), c*||k||^2 < 1
    """
    sqrt_c = torch.sqrt(torch.tensor(c, dtype=p.dtype, device=p.device))
    
    # Scale to standard Poincaré coordinates (‖p_standard‖ < 1)
    p_standard = p * sqrt_c
    
    # Ensure ‖p_standard‖² < 1
    p_norm_sq = torch.sum(p_standard**2, dim=-1, keepdim=True)
    p_norm_sq = torch.clamp(p_norm_sq, max=1-eps)
    
    # Direct conversion
    k_standard = 2 * p_standard / (1 + p_norm_sq)
    
    # Scale to Klein coordinates with c‖k‖² < 1
    k = k_standard / sqrt_c
    
    return k


# ============================================================
# Compute Einstein/Klein weighted mean of Lorentz points
# ============================================================
def klein_mean_lorentz_factor_curvature(points_lorentz, c=1.0, eps=1e-5):
    """
    Compute Einstein/Klein weighted mean with Lorentz factor and curvature c.

    Args:
        points_lorentz: (N, n+1), points on Lorentz hyperboloid (x0>0, <x,x>_L=-1/c)
        c: hyperbolic curvature
        eps: small value for numerical stability
    Returns:
        mean_lorentz: (n+1,), Lorentz coordinates of the mean
    Notes:
        - Maps Lorentz -> Klein -> weighted mean -> Lorentz
        - Domain: Lorentz points on upper sheet
        - Klein domain: c*||k||^2 < 1
        - Numerical stability ensured via eps in sqrt and denominator
    """
    # Step 1: Lorentz -> Klein (curvature scaled)
    points_klein = lorentz_to_klein(points_lorentz, c=c, eps=eps)  # (N, n)

    # Step 2: Lorentz factor with curvature c
    norm_sq = torch.sum(points_klein**2, dim=-1, keepdim=True)
    norm_sq = norm_sq.clamp(max=1 - eps)  # Klein domain: c*||k||^2 < 1
    gamma = 1.0 / torch.sqrt(1 - c * norm_sq + eps)

    # Step 3: Weighted mean in Klein coordinates
    weighted_sum = torch.sum(points_klein * gamma, dim=0)
    gamma_sum = torch.sum(gamma, dim=0) + eps
    mean_klein = weighted_sum / gamma_sum

    # Step 4: Map back to Lorentz coordinates
    mean_lorentz = klein_to_lorentz(mean_klein, c=c, eps=eps)
    return mean_lorentz


# ============================================================
# Compute masked Lorentz mean from spatial components
# ============================================================

def masked_lorentz_mean_klein(x_spatial, mask, c=1.0, eps=1e-6):
    """
    Compute mean of Lorentz points using Klein transformation method.
    
    Args:
        x_spatial: (bs, h, w, n) spatial Lorentz features
        mask: (bs, h, w), binary mask (0 or 1)
        c: hyperbolic curvature
        eps: small epsilon for numerical stability
    
    Returns:
        mean_lorentz: (n+1,), Lorentz coordinates of mean
    """
    bs, h, w, n = x_spatial.shape
    
    # Flatten spatial dims
    x_flat = x_spatial.reshape(-1, n)
    mask_flat = mask.reshape(-1)

    # Select only masked points
    selected = x_flat[mask_flat.bool()]
    if selected.shape[0] == 0:
        raise ValueError("Mask has no 1s, cannot compute mean.")

    # Compute time components
    norm_sq = torch.sum(selected**2, dim=-1, keepdim=True).clamp(min=0.0)
    time = torch.sqrt(norm_sq + 1.0 / c + eps)
    
    # Convert to Klein coordinates
    sqrt_c = torch.sqrt(torch.tensor(c, dtype=selected.dtype, device=selected.device))
    klein = selected / (sqrt_c * time)
    
    # Compute weights (Lorentz factors)
    weights = 1.0 / torch.sqrt(1 - c * torch.sum(klein**2, dim=-1, keepdim=True) + eps)
    
    # Compute weighted mean in Klein coordinates
    weighted_sum = torch.sum(klein * weights, dim=0)
    weights_sum = torch.sum(weights, dim=0) + eps
    mean_klein = weighted_sum / weights_sum
    
    # Convert back to Lorentz coordinates
    norm_sq_mean = torch.sum(mean_klein**2, dim=-1, keepdim=True).clamp(max=1-eps)
    time_mean = 1.0 / (sqrt_c * torch.sqrt(1 - norm_sq_mean + eps))
    spatial_mean = mean_klein * time_mean * sqrt_c
    
    return torch.cat([time_mean, spatial_mean], dim=-1)



def lorentz_dot(x, y, c=1.0):
    """
    Compute the Lorentz inner product: -c*x₀y₀ + x₁y₁ + ... + xₙyₙ
    """
    return -c*x[0]*y[0] + torch.sum(x[1:]*y[1:])


def masked_lorentz_mean_direct(x_spatial, mask, c=1.0, eps=1e-6):
    """
    Compute mean of Lorentz points using direct Lorentz method.
    
    Args:
        x_spatial: (bs, h, w, n) spatial Lorentz features
        mask: (bs, h, w), binary mask (0 or 1)
        c: hyperbolic curvature
        eps: small epsilon for numerical stability
    
    Returns:
        mean_lorentz: (n+1,), Lorentz coordinates of mean
    """
    bs, h, w, n = x_spatial.shape
    
    # Flatten spatial dims
    x_flat = x_spatial.reshape(-1, n)
    mask_flat = mask.reshape(-1)

    # Select only masked points
    selected = x_flat[mask_flat.bool()]
    if selected.shape[0] == 0:
        raise ValueError("Mask has no 1s, cannot compute mean.")

    # Compute full Lorentz coordinates
    norm_sq = torch.sum(selected**2, dim=-1, keepdim=True).clamp(min=0.0)
    time = torch.sqrt(norm_sq + 1.0 / c + eps)
    lorentz_points = torch.cat([time, selected], dim=-1)

    # Compute weights (Lorentz factors)
    weights = lorentz_points[..., :1]

    # Compute weighted sum
    weighted_sum = torch.sum(lorentz_points * weights, dim=0)
    weights_sum = torch.sum(weights, dim=0) + eps

    # Normalize to hyperboloid
    mean_unnormalized = weighted_sum / weights_sum
    
    # Ensure the result is on the hyperboloid
    x0 = mean_unnormalized[0]
    xs = mean_unnormalized[1:]
    lorentz_norm_sq = -c * x0**2 + torch.sum(xs**2)
    normalization_factor = torch.sqrt(-1 / (c * lorentz_norm_sq) + eps)
    
    return mean_unnormalized * normalization_factor

