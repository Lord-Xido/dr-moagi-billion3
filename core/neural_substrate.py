"""
Neural Substrate: Deep network mapping structural representations
in the latent manifold with learnable geometric operators.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.geometry_ops import GeometryOps


class NeuralSubstrate(nn.Module):
    """Deep 3D convolutional network with explicit geometric operators.
    
    Architecture:
    - Encoder: 32³ → 16³ → 8³ → 4³ (spatial compression)
    - Latent operators: ∇U, Δz, curvature, memory mixing
    - Decoder: 4³ → 8³ → 16³ → 32³ (reconstruction)
    
    The substrate maintains learnable weight matrices for:
    - δ (gradient descent strength)
    - κ (curvature sensitivity)
    - γ (gradient weighting)
    - ω (memory/control mixing)
    """

    def __init__(self, in_channels: int = 1, latent_channels: int = 8, spatial_size: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.spatial_size = spatial_size

        # =====================================================================
        # ENCODER: V_t → Z_t (32³ → 4³ with 3 stages)
        # =====================================================================
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv3d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv3d(16, latent_channels, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
        )

        # =====================================================================
        # LATENT DYNAMICS OPERATORS
        # =====================================================================
        # Expert residual block for latent state transformation
        self.expert_residual = nn.Sequential(
            nn.Conv3d(latent_channels, latent_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        # Router for multi-expert selection (4 experts)
        self.router_dense = nn.Linear(latent_channels * 4 * 4 * 4, 4)

        # Learnable operator weights
        # These scale the contributions of different forces in latent kinetics
        self.w_delta = nn.Parameter(torch.ones(1, latent_channels, 1, 1, 1) * 0.1)
        self.w_kappa = nn.Parameter(torch.ones(1, latent_channels, 1, 1, 1) * 0.1)
        self.w_grad = nn.Parameter(torch.ones(1, latent_channels, 1, 1, 1) * 0.05)
        self.w_omega = nn.Parameter(torch.ones(1, latent_channels, 1, 1, 1) * 0.2)

        # Global scaling parameters for memory and control
        self.g_Z = nn.Parameter(torch.tensor([0.5]))
        self.g_omega = nn.Parameter(torch.tensor([0.2]))

        # Memory permeation network: fuses latent state with memory
        self.permeate_net = nn.Sequential(
            nn.Conv3d(latent_channels * 2, latent_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        # =====================================================================
        # DECODER: Z_t → X̂_t (4³ → 32³ with 3 stages)
        # =====================================================================
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(latent_channels, 16, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose3d(8, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, V: torch.Tensor) -> torch.Tensor:
        """Encode volumetric input to latent representation.
        
        Args:
            V: Input tensor (B, C, 32, 32, 32)
            
        Returns:
            Z: Latent tensor (B, latent_channels, 4, 4, 4)
        """
        return self.encoder(V)

    def decode(self, Z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to volumetric space.
        
        Args:
            Z: Latent tensor (B, latent_channels, 4, 4, 4)
            
        Returns:
            X̂: Reconstructed tensor (B, in_channels, 32, 32, 32)
        """
        return self.decoder(Z)

    def latent_kinetics(
        self,
        Z: torch.Tensor,
        Omega: torch.Tensor,
        dt: float = 0.01,
        num_steps: int = 5,
    ) -> tuple:
        """Apply inward contraction dynamics in latent space.
        
        Implements:
        ∂Z/∂t = -∇U(Z) + ν∇²Z + F_mem(Omega) + F_attn(Z) + F_control
        
        Args:
            Z: Latent state (B, C, 4, 4, 4)
            Omega: Memory field (B, C, 4, 4, 4)
            dt: Time step
            num_steps: Number of integration steps
            
        Returns:
            (Z_evolved, Omega_updated): Updated latent and memory
        """
        Z_curr = Z.clone()
        Omega_curr = Omega.clone() if Omega is not None else torch.zeros_like(Z)

        for step in range(num_steps):
            # Compute potential gradient (via implicit differentiation)
            Z_curr.requires_grad_(True)
            
            # Reconstruction through decoder (defines implicit potential)
            X_recon = self.decode(Z_curr)
            U = X_recon.mean()  # Scalar potential
            U.backward()
            
            grad_U = Z_curr.grad.detach()
            Z_curr.requires_grad_(False)

            # Laplacian for diffusion
            lap_Z = GeometryOps.laplacian(Z_curr)

            # Curvature term
            curv_Z = GeometryOps.curvature_proxy(Z_curr)

            # Memory forcing: permeate latent state with Omega
            if Omega_curr is not None:
                Z_Omega = torch.cat([Z_curr, Omega_curr], dim=1)
                F_mem = self.permeate_net(Z_Omega)
            else:
                F_mem = torch.zeros_like(Z_curr)

            # Attention-based control (use expert router)
            Z_flat = Z_curr.view(Z_curr.size(0), -1)
            expert_scores = torch.softmax(self.router_dense(Z_flat), dim=1)  # (B, 4)
            F_attn = expert_scores[:, 0:1].view(-1, 1, 1, 1, 1) * self.expert_residual(Z_curr)

            # Inward contraction dynamics
            dZ_dt = (
                -self.w_delta * grad_U
                + self.w_kappa * curv_Z
                + self.w_grad * lap_Z
                + self.w_omega * F_mem
                + F_attn
            )

            # Euler step
            Z_curr = Z_curr + dt * dZ_dt

            # Update memory: leaky integration with error signal
            Omega_curr = 0.95 * Omega_curr + 0.05 * F_mem

        return Z_curr, Omega_curr

    def compute_reconstruction_loss(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """L²reconstruction loss between input and reconstruction.
        
        Args:
            X: Input tensor (B, C, 32, 32, 32)
            Z: Latent tensor (B, C, 4, 4, 4)
            
        Returns:
            Scalar loss
        """
        X_recon = self.decode(Z)
        return F.mse_loss(X, X_recon)

    def compute_cycle_loss(self, Z: torch.Tensor) -> torch.Tensor:
        """Cycle consistency loss: E(D(Z)) ≈ Z.
        
        Args:
            Z: Latent tensor (B, C, 4, 4, 4)
            
        Returns:
            Scalar loss
        """
        X_recon = self.decode(Z)
        Z_cycle = self.encode(X_recon)
        return F.mse_loss(Z, Z_cycle)

    def compute_geometry_loss(self, Z: torch.Tensor, lambda_geo: float = 0.01) -> torch.Tensor:
        """Geometric regularization: penalize high curvature.
        
        Args:
            Z: Latent tensor
            lambda_geo: Weighting coefficient
            
        Returns:
            Scalar loss
        """
        curv = GeometryOps.curvature_proxy(Z)
        return lambda_geo * torch.mean(torch.abs(curv))

    def forward(
        self,
        X: torch.Tensor,
        Omega: torch.Tensor = None,
        num_contraction_steps: int = 5,
        compute_losses: bool = True,
    ) -> dict:
        """Full forward pass through entire inward loop.
        
        Args:
            X: Input volume (B, C, 32, 32, 32)
            Omega: Memory field (B, C, 4, 4, 4)
            num_contraction_steps: Iterations of latent dynamics
            compute_losses: Whether to compute loss terms
            
        Returns:
            dict with:
                Z: Encoded latent state
                Z_star: After inward contraction
                X_recon: Reconstruction
                Omega_next: Updated memory
                losses: Dictionary of loss components
                metrics: Reconstruction error, spectral radius, etc.
        """
        batch_size = X.shape[0]

        # 1. ENCODE: V → Z
        Z = self.encode(X)

        # 2. Initialize memory if needed
        if Omega is None:
            Omega = torch.zeros_like(Z)

        # 3. INWARD CONTRACTION: Z → Z*
        Z_star, Omega_next = self.latent_kinetics(Z, Omega, dt=0.01, num_steps=num_contraction_steps)

        # 4. DECODE: Z* → X̂
        X_recon = self.decode(Z_star)

        # 5. VERIFY: Compute error and metrics
        recon_error = X - X_recon
        metrics = {
            "recon_mse": F.mse_loss(X, X_recon).item(),
            "recon_l1": F.l1_loss(X, X_recon).item(),
            "z_mean": Z_star.mean().item(),
            "z_std": Z_star.std().item(),
            "z_max": Z_star.max().item(),
            "z_min": Z_star.min().item(),
        }

        # 6. Compute losses if requested
        losses = {}
        if compute_losses:
            losses["L_recon"] = self.compute_reconstruction_loss(X, Z_star)
            losses["L_cycle"] = self.compute_cycle_loss(Z_star)
            losses["L_geo"] = self.compute_geometry_loss(Z_star)
            losses["L_total"] = (
                losses["L_recon"] + 0.1 * losses["L_cycle"] + 0.01 * losses["L_geo"]
            )

        return {
            "Z": Z,
            "Z_star": Z_star,
            "X_recon": X_recon,
            "recon_error": recon_error,
            "Omega_next": Omega_next,
            "losses": losses,
            "metrics": metrics,
        }
