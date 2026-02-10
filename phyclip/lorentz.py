# ---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------

# Modified from github.com/facebookresearch/meru

"""
Implementation of common operations for the Lorentz model of hyperbolic geometry.
This model represents a hyperbolic space of `d` dimensions on the upper-half of
a two-sheeted hyperboloid in a Euclidean space of `(d+1)` dimensions.

Hyperbolic geometry has a direct connection to the study of special relativity
theory -- implementations in this module borrow some of its terminology. The axis
of symmetry of the Hyperboloid is called the _time dimension_, while all other
axes are collectively called _space dimensions_.

All functions implemented here only input/output the space components, while
while calculating the time component according to the Hyperboloid constraint:

    `x_time = torch.sqrt(1 / curv + torch.norm(x_space) ** 2)`
"""

from __future__ import annotations

import math

import torch
from loguru import logger
from torch import Tensor


def pairwise_inner(x: Tensor, y: Tensor, curv: float | Tensor = 1.0):
    """
    Compute pairwise Lorentzian inner product between input vectors.

    Args:
        x: Tensor of shape `(B1, D)` giving a space components of a batch
            of vectors on the hyperboloid.
        y: Tensor of shape `(B2, D)` giving a space components of another
            batch of points on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B1, B2)` giving pairwise Lorentzian inner product
        between input vectors.
    """

    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1, keepdim=True))
    xyl = x @ y.T - x_time @ y_time.T
    return xyl


def pairwise_inner_batch(x: Tensor, y: Tensor, curv: Tensor) -> Tensor:
    """
    Compute pairwise Lorentzian inner product for multiple hyperbolic spaces.
    This version expects batch-first format (B, N, D) instead of (N, B, D).

    Args:
        x: Tensor of shape `(B1, N, D)` giving space components of first batch
        y: Tensor of shape `(B2, N, D)` giving space components of second batch
        curv: Tensor of shape `(N,)` or `(N, 1)` giving curvature for each hyperbolic space

    Returns:
        Tensor of shape `(N, B1, B2)` giving pairwise Lorentzian inner products
    """
    # Transpose to get (N, B, D) format for computation
    x_t = x.transpose(0, 1)  # (B1, N, D) -> (N, B1, D)
    y_t = y.transpose(0, 1)  # (B2, N, D) -> (N, B2, D)

    # Ensure curv has proper shape for broadcasting: (N,) -> (N, 1, 1)
    if curv.dim() == 1:
        curv_reshaped = curv.view(-1, 1, 1)  # (N,) -> (N, 1, 1)
    elif curv.dim() == 2 and curv.shape[1] == 1:
        curv_reshaped = curv.view(-1, 1, 1)  # (N, 1) -> (N, 1, 1)
    else:
        curv_reshaped = curv

    # Time components calculation for each hyperbolic space
    # x_time: (N, B1, 1), y_time: (N, B2, 1)
    x_time = torch.sqrt(1 / curv_reshaped + torch.sum(x_t**2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1 / curv_reshaped + torch.sum(y_t**2, dim=-1, keepdim=True))

    # Batch matrix multiplication: (N, B1, D) @ (N, D, B2) -> (N, B1, B2)
    spatial_inner = torch.bmm(x_t, y_t.transpose(-2, -1))

    # Time component: (N, B1, 1) @ (N, 1, B2) -> (N, B1, B2)
    time_inner = torch.bmm(x_time, y_time.transpose(-2, -1))

    # Lorentzian inner product
    xyl = spatial_inner - time_inner
    return xyl


def pairwise_dist(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
) -> Tensor:
    """
    Compute the pairwise geodesic distance between two batches of points on
    the hyperboloid.

    Args:
        x: Tensor of shape `(B1, D)` giving a space components of a batch
            of point on the hyperboloid.
        y: Tensor of shape `(B2, D)` giving a space components of another
            batch of points on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B1, B2)` giving pairwise distance along the geodesics
        connecting the input points.
    """

    # Ensure numerical stability in arc-cosh by clamping input.
    c_xyl = -curv * pairwise_inner(x, y, curv)
    _distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return _distance / curv**0.5


def pairwise_dist_batch(
    x: Tensor, y: Tensor, curv: Tensor, eps: float = 1e-8
) -> Tensor:
    """
    Compute pairwise hyperbolic distances between two sets of points for multiple hyperbolic spaces.
    This version expects batch-first format (B, N, D) .

    Args:
        x: First set of points in the Hyperboloid of shape `(B1, N, D)` where N is the number of hyperbolic spaces.
        y: Second set of points in the Hyperboloid of shape `(B2, N, D)` where B2 is the number of points.
        curv: Negative curvature of each Hyperboloid of shape `(N,)` or `(N, 1)`.
        eps: Small value to prevent numerical instability.

    Returns:
        Pairwise distances of shape `(N, B1, B2)`.
    """
    # Compute pairwise inner products for all hyperbolic spaces
    inner_products = pairwise_inner_batch(x, y, curv)  # (N, B1, B2)

    # Ensure curv has proper shape for broadcasting: (N,) -> (N, 1, 1)
    if curv.dim() == 1:
        curv_reshaped = curv.view(-1, 1, 1)  # (N,) -> (N, 1, 1)
    elif curv.dim() == 2 and curv.shape[1] == 1:
        curv_reshaped = curv.view(-1, 1, 1)  # (N, 1) -> (N, 1, 1)
    else:
        curv_reshaped = curv

    # Apply curvature scaling: curv_reshaped has shape (N, 1, 1), broadcasts to (N, B1, B2)
    c_xyl = -curv_reshaped * inner_products

    # Ensure numerical stability in acosh by clamping input
    clamped_input = torch.clamp(c_xyl, min=1 + eps)

    # Compute hyperbolic distances
    _distance = torch.acosh(clamped_input)

    # Scale by square root of curvature
    distances = _distance / torch.sqrt(curv_reshaped)

    return distances


def exp_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Map points from the tangent space at the vertex of hyperboloid, on to the
    hyperboloid. This mapping is done using the exponential map of Lorentz model.

    Args:
        x: Tensor of shape `(B, D)` giving batch of Euclidean vectors to project
            onto the hyperboloid. These vectors are interpreted as velocity
            vectors in the tangent space at the hyperboloid vertex.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of same shape as `x`, giving space components of the mapped
        vectors on the hyperboloid.
    """

    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)

    # Ensure numerical stability in sinh by clamping input.
    sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2**15))
    _output = torch.sinh(sinh_input) * x / torch.clamp(rc_xnorm, min=eps)
    return _output


def exp_map0_batch(x: Tensor, curv: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Map points from the tangent space at the vertex of hyperboloid onto multiple
    hyperboloids with different curvatures. This is a batch version of exp_map0
    that efficiently handles multiple hyperbolic spaces.

    Args:
        x: Tensor of shape `(B, N, D)` giving batch of Euclidean vectors for N
            hyperbolic spaces to project onto their respective hyperboloids.
            These vectors are interpreted as velocity vectors in the tangent
            space at the hyperboloid vertex.
        curv: Tensor of shape `(N,)` or `(N, 1)` giving positive scalars denoting
            negative hyperboloid curvature for each space.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of shape `(B, N, D)` giving space components of the mapped
        vectors on their respective hyperboloids.
    """
    if curv.dim() == 1:
        curv = curv.view(1, -1, 1)
    elif curv.dim() == 2 and curv.shape[1] == 1:
        curv = curv.view(1, -1, 1)

    # Apply exp_map0
    return exp_map0(x, curv, eps)


def log_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Inverse of the exponential map: map points from the hyperboloid on to the
    tangent space at the vertex, using the logarithmic map of Lorentz model.

    Args:
        x: Tensor of shape `(B, D)` giving space components of points
            on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of same shape as `x`, giving Euclidean vectors in the tangent
        space of the hyperboloid vertex.
    """

    # Calculate distance of vectors to the hyperboloid vertex.
    rc_x_time = torch.sqrt(1 + curv * torch.sum(x**2, dim=-1, keepdim=True))
    _distance0 = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))

    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)
    _output = _distance0 * x / torch.clamp(rc_xnorm, min=eps)
    return _output


def log_map0_batch(x: Tensor, curv: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Inverse of the exponential map: map points from the hyperboloid on to the
    tangent space at the vertex, using the logarithmic map of Lorentz model.

    Args:
        x: Tensor of shape `(B, N, D)` giving space components of points
            on the hyperboloid.
        curv: Tensor of shape `(N,)` or `(N, 1)` giving positive scalars denoting
            negative hyperboloid curvature for each space.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of shape `(B, N, D)` giving Euclidean vectors in the tangent
        space of the hyperboloid vertex.
    """
    if curv.dim() == 1:
        curv = curv.view(1, -1, 1)
    elif curv.dim() == 2 and curv.shape[1] == 1:
        curv = curv.view(1, -1, 1)

    # Apply log_map0
    return log_map0(x, curv, eps)


def half_aperture(
    x: Tensor, curv: float | Tensor = 1.0, min_radius: float = 0.1, eps: float = 1e-8
) -> Tensor:
    """
    Compute the half aperture angle of the entailment cone formed by vectors on
    the hyperboloid. The given vector would meet the apex of this cone, and the
    cone itself extends outwards to infinity.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        min_radius: Radius of a small neighborhood around vertex of the hyperboloid
            where cone aperture is left undefined. Input vectors lying inside this
            neighborhood (having smaller norm) will be projected on the boundary.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B, )` giving the half-aperture of entailment cones
        formed by input vectors. Values of this tensor lie in `(0, pi/2)`.
    """

    # Ensure numerical stability in arc-sin by clamping input.
    asin_input = 2 * min_radius / (torch.norm(x, dim=-1) * curv**0.5 + eps)
    _half_aperture = torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))

    return _half_aperture


def oxy_angle(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8):
    """
    Given two vectors `x` and `y` on the hyperboloid, compute the exterior
    angle at `x` in the hyperbolic triangle `Oxy` where `O` is the origin
    of the hyperboloid.

    This expression is derived using the Hyperbolic law of cosines.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        y: Tensor of same shape as `x` giving another batch of vectors.
        curv: Positive scalar denoting negative hyperboloid curvature.

    Returns:
        Tensor of shape `(B, )` giving the required angle. Values of this
        tensor lie in `(0, pi)`.
    """

    # Calculate time components of inputs (multiplied with `sqrt(curv)`):
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1))

    # Calculate lorentzian inner product multiplied with curvature. We do not use
    # the `pairwise_inner` implementation to save some operations (since we only
    # need the diagonal elements).
    c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)

    # Make the numerator and denominator for input to arc-cosh, shape: (B, )
    acos_numer = y_time + c_xyl * x_time
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))

    acos_input = acos_numer / (torch.norm(x, dim=-1) * acos_denom + eps)
    _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

    return _angle


def oxy_angle_eval(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8):
    """
    Given two vectors `x` and `y` on the hyperboloid, compute the exterior
    angle at `x` in the hyperbolic triangle `Oxy` where `O` is the origin
    of the hyperboloid.

    This expression is derived using the Hyperbolic law of cosines.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        y: Tensor of same shape as `x` giving another batch of vectors.
        curv: Positive scalar denoting negative hyperboloid curvature.

    Returns:
        Tensor of shape `(B, )` giving the required angle. Values of this
        tensor lie in `(0, pi)`.
    """

    # Calculate time components of inputs (multiplied with `sqrt(curv)`):
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1, keepdim=True))

    logger.info(f"x_time shape: {x_time.size()}")
    logger.info(f"y_time shape: {y_time.size()}")

    # Calculate lorentzian inner product multiplied with curvature. We do not use
    # the `pairwise_inner` implementation to save some operations (since we only
    # need the diagonal elements).

    # c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)
    c_xyl = curv * (y @ x.T - y_time @ x_time.T)
    logger.info(f"c_xyl shape: {c_xyl.size()}")

    # Make the numerator and denominator for input to arc-cosh, shape: (B, )
    acos_numer = y_time + c_xyl * x_time.T
    logger.info(f"acos_numer shape: {acos_numer.size()}")
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))
    logger.info(f"acos_denom shape: {acos_denom.size()}")

    acos_input = acos_numer / (torch.norm(x, dim=-1, keepdim=True).T * acos_denom + eps)
    _angle = -torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

    return _angle


def get_space_norm(x: Tensor) -> Tensor:
    """
    Compute the Euclidean norm of the space components along the last dimension.

    This function calculates the L2 norm of vectors along the last dimension
    while keeping the dimension structure intact (keepdim=True).

    Args:
        x: Tensor of any shape `(..., D)` where D is the dimension to compute norm over.
           Common shapes include:
           - `(B, D)`: Batch of vectors → returns `(B, 1)`
           - `(B, K, D)`: Batch of K-dimensional concept vectors → returns `(B, K, 1)`
           - `(D,)`: Single vector → returns `(1,)`

    Returns:
        Tensor with same number of dimensions as input, where the last dimension
        becomes 1. The norm is computed along the last dimension with keepdim=True.

        Output shapes:
        - Input `(B, D)` → Output `(B, 1)`
        - Input `(B, K, D)` → Output `(B, K, 1)`
        - Input `(D,)` → Output `(1,)`

    Example:
        >>> x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
        >>> get_space_norm(x)  # Returns (2, 1)
        tensor([[3.7417], [8.7750]])
    """
    return torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True))


def get_distance_from_origin(x: Tensor, curv: float | Tensor) -> Tensor:
    """
    Compute the distance from the origin of the hyperboloid.

    Args:
        x: Tensor of shape `(B, num_subspaces, subspace_dim)` giving space components of vectors
        curv: Tensor of shape `(num_subspaces,)` or `(num_subspaces, 1)` giving curvature for each hyperbolic space.

    Returns:
        Tensor of shape `(B, num_subspaces, 1)` giving the distance from the origin.
    """
    # Ensure curv has proper shape for broadcasting: (1, num_subspaces, 1)
    if curv.dim() == 1:
        curv_reshaped = curv.view(1, -1, 1)
    elif curv.dim() == 2 and curv.shape[1] == 1:
        curv_reshaped = curv.view(1, -1, 1)
    else:
        curv_reshaped = curv

    # Compute time component: (B, num_subspaces, 1)
    time_component = torch.sqrt(
        1 / curv_reshaped + torch.sum(x**2, dim=-1, keepdim=True)
    )

    # Pre-compute sqrt(c) for efficiency and clarity
    sqrt_curv = torch.sqrt(curv_reshaped)

    # Compute distance from origin: (B, num_subspaces, 1)
    # The argument to acosh must be >= 1. Clamp to avoid NaN due to numerical precision.
    acosh_argument = torch.clamp(time_component * sqrt_curv, min=1.0)
    distances = torch.acosh(acosh_argument) / sqrt_curv

    return distances


def oxy_angle_batch(x: Tensor, y: Tensor, curv: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Compute exterior angles at x in hyperbolic triangles Oxy for multiple hyperbolic spaces.
    This version expects batch-first format (B, N, D) instead of (N, B, D).

    Args:
        x: Tensor of shape `(B, N, D)` giving space components of vectors
        y: Tensor of shape `(B, N, D)` giving space components of vectors
        curv: Tensor of shape `(N,)` or `(N, 1)` giving curvature for each hyperbolic space
        eps: Small value to prevent numerical instability

    Returns:
        Tensor of shape `(N, B)` giving angles for each hyperbolic space
    """
    # Transpose to get (N, B, D) format for computation
    x_t = x.transpose(0, 1)  # (B, N, D) -> (N, B, D)
    y_t = y.transpose(0, 1)  # (B, N, D) -> (N, B, D)

    # Ensure curv has proper shape
    if curv.dim() == 1:
        curv = curv.view(-1, 1)  # (N,) -> (N, 1)

    # Calculate time components: (N, B)
    x_time = torch.sqrt(
        1 / curv + torch.sum(x_t**2, dim=-1)
    )  # (N, 1) + (N, B) -> (N, B)
    y_time = torch.sqrt(
        1 / curv + torch.sum(y_t**2, dim=-1)
    )  # (N, 1) + (N, B) -> (N, B)

    # Calculate Lorentzian inner product multiplied with curvature: (N, B)
    spatial_inner = torch.sum(x_t * y_t, dim=-1)  # (N, B)
    c_xyl = curv * (spatial_inner - x_time * y_time)  # (N, 1) * (N, B) -> (N, B)

    # Compute numerator and denominator for acos input: (N, B)
    acos_numer = y_time + c_xyl * x_time
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))

    # Compute x norms: (N, B)
    x_norms = torch.norm(x_t, dim=-1)

    # Final acos input: (N, B)
    acos_input = acos_numer / (x_norms * acos_denom + eps)

    # Clamp and compute angle
    clamped_input = torch.clamp(acos_input, min=-1 + eps, max=1 - eps)
    _angle = torch.acos(clamped_input)

    return _angle


def half_aperture_batch(
    x: Tensor, curv: Tensor, min_radius: float = 0.1, eps: float = 1e-8
) -> Tensor:
    """
    Compute the half aperture angle of entailment cones for multiple hyperbolic spaces.
    This version expects batch-first format (B, N, D) instead of (N, B, D).

    Args:
        x: Tensor of shape `(B, N, D)` giving space components of vectors
        curv: Tensor of shape `(N,)` or `(N, 1)` giving curvature for each hyperbolic space
        min_radius: Minimum radius for numerical stability
        eps: Small value to prevent numerical instability

    Returns:
        Tensor of shape `(N, B)` giving half aperture angles
    """
    # Transpose to get (N, B, D) format for computation
    x_t = x.transpose(0, 1)  # (B, N, D) -> (N, B, D)

    # Ensure curv has proper shape
    if curv.dim() == 1:
        curv = curv.view(-1, 1)  # (N,) -> (N, 1)

    # Compute norms for each batch in each hyperbolic space: (N, B)
    x_norms = torch.norm(x_t, dim=-1)  # (N, B)

    # Compute asin input with broadcasting: (N, 1) broadcasts with (N, B)
    asin_input = 2 * min_radius / (x_norms * torch.sqrt(curv) + eps)

    # Clamp and compute half aperture
    clamped_input = torch.clamp(asin_input, min=-1 + eps, max=1 - eps)
    _half_aperture = torch.asin(clamped_input)

    return _half_aperture
