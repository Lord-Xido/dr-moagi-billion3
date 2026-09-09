"""
Discrete mathematical derivations executed directly on PyTorch tensors.
Implements differential geometry operations for volumetric fields.
"""

import torch
import torch.nn.functional as F


class GeometryOps:
    """Differential geometry operators on 5D tensors (batch, channel, x, y, z)."""

    @staticmethod
    def grad3(z: torch.Tensor) -> tuple:
        """Compute 3D spatial gradient using finite differences.
        
        Args:
            z: Tensor of shape (B, C, X, Y, Z)
            
        Returns:
            (dx, dy, dz): Gradient components, same shape as z
        """
        # Central differences with zero-padding at boundaries
        dz = F.pad(z[:, :, 2:, :, :] - z[:, :, :-2, :, :], (0, 0, 0, 0, 1, 1)) * 0.5
        dy = F.pad(z[:, :, :, 2:, :] - z[:, :, :, :-2, :], (0, 0, 1, 1, 0, 0)) * 0.5
        dx = F.pad(z[:, :, :, :, 2:] - z[:, :, :, :, :-2], (1, 1, 0, 0, 0, 0)) * 0.5
        return dx, dy, dz

    @staticmethod
    def laplacian(z: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian ∇²z = ∂²z/∂x² + ∂²z/∂y² + ∂²z/∂z².
        
        Uses replicate padding to handle boundaries.
        
        Args:
            z: Tensor of shape (B, C, X, Y, Z)
            
        Returns:
            Laplacian field, same shape as z
        """
        zp = F.pad(z, (1, 1, 1, 1, 1, 1), mode="replicate")
        lap = (
            zp[:, :, 1:-1, 1:-1, 2:] + zp[:, :, 1:-1, 1:-1, :-2] +  # x-direction
            zp[:, :, 1:-1, 2:, 1:-1] + zp[:, :, 1:-1, :-2, 1:-1] +  # y-direction
            zp[:, :, 2:, 1:-1, 1:-1] + zp[:, :, :-2, 1:-1, 1:-1] -  # z-direction
            6.0 * z
        )
        return lap

    @staticmethod
    def curvature_proxy(z: torch.Tensor) -> torch.Tensor:
        """Compute scalar curvature proxy: κ ≈ Δz / (1 + ||z||).
        
        Useful for detecting geometric bending.
        
        Args:
            z: Tensor of shape (B, C, X, Y, Z)
            
        Returns:
            Curvature field, same shape as z
        """
        lap = GeometryOps.laplacian(z)
        scale = torch.sqrt(torch.mean(z * z, dim=1, keepdim=True) + 1e-6)
        return lap / (1.0 + scale)

    @staticmethod
    def divergence_3d(vx: torch.Tensor, vy: torch.Tensor, vz: torch.Tensor) -> torch.Tensor:
        """Compute 3D divergence ∇·v = ∂vx/∂x + ∂vy/∂y + ∂vz/∂z.
        
        Args:
            vx, vy, vz: Vector field components, each shape (B, C, X, Y, Z)
            
        Returns:
            Divergence field, shape (B, C, X, Y, Z)
        """
        dvx_dx = F.pad(vx[:, :, :, :, 1:] - vx[:, :, :, :, :-1], (0, 1, 0, 0, 0, 0))
        dvy_dy = F.pad(vy[:, :, :, 1:, :] - vy[:, :, :, :-1, :], (0, 0, 0, 1, 0, 0))
        dvz_dz = F.pad(vz[:, :, 1:, :, :] - vz[:, :, :-1, :, :], (0, 0, 0, 0, 0, 1))
        return dvx_dx + dvy_dy + dvz_dz

    @staticmethod
    def spectral_radius(J: torch.Tensor, max_iters: int = 20) -> float:
        """Estimate spectral radius of Jacobian via power iteration.
        
        Args:
            J: Jacobian matrix or linear operator
            max_iters: Number of power iterations
            
        Returns:
            Estimated spectral radius (largest eigenvalue magnitude)
        """
        v = torch.randn(J.shape[-1], 1, device=J.device)
        for _ in range(max_iters):
            v = J @ v
            v = v / (torch.norm(v) + 1e-8)
        eigenvalue = torch.norm(J @ v) / (torch.norm(v) + 1e-8)
        return eigenvalue.item()

    @staticmethod
    def local_voxel_update(
        S: torch.Tensor,
        U_grad: torch.Tensor,
        nu: float = 0.01,
        F_mem: torch.Tensor = None,
        F_attn: torch.Tensor = None,
        F_control: torch.Tensor = None,
    ) -> torch.Tensor:
        """Apply local voxel update rule for 3D state.
        
        ∂S/∂t = -∇U + ν∇²S + F_mem + F_attn + F_control
        
        Args:
            S: Current state (B, C, X, Y, Z)
            U_grad: Gradient of potential (B, C, X, Y, Z)
            nu: Diffusion coefficient
            F_mem, F_attn, F_control: External forcing terms
            
        Returns:
            Updated state derivative
        """
        laplacian_S = GeometryOps.laplacian(S)
        dS = -U_grad + nu * laplacian_S
        
        if F_mem is not None:
            dS = dS + F_mem
        if F_attn is not None:
            dS = dS + F_attn
        if F_control is not None:
            dS = dS + F_control
            
        return dS
