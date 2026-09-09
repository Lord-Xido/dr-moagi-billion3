"""
Machine State: Carries active numerical values and metrics across execution loops.
Represents the complete computational state of the Dr Moagi system.
"""

import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class MachineState:
    """Complete state vector S_t of the inward loop machine.
    
    Tracks:
    - pc: program counter
    - X: physical world observation
    - V: volumetric embedding
    - Z: latent representation
    - Omega: memory field
    - X_hat: reconstruction
    - e: reconstruction error
    - Pi: hypothesis predictions
    - J: total loss
    - L_rec: reconstruction loss
    - L_geo: geometric loss
    - L_cycle: cycle consistency loss
    - spec_radius: spectral radius of transition operator
    - V: convergence flag
    - Theta: model parameters (reference only)
    - iteration: current micro-step within macro iteration
    - macro_epoch: outer loop iteration
    """
    
    pc: int = 0  # Program counter
    
    # Physical/volumetric state
    X: Optional[torch.Tensor] = None  # Input observation (B, C, 32, 32, 32)
    V: Optional[torch.Tensor] = None  # Volumetric embedding (B, C, 32, 32, 32)
    
    # Latent state
    Z: Optional[torch.Tensor] = None  # Latent encoding (B, C, 4, 4, 4)
    Z_star: Optional[torch.Tensor] = None  # After contraction
    
    # Memory and control
    Omega: Optional[torch.Tensor] = None  # Memory field (B, C, 4, 4, 4)
    Lambda: Optional[torch.Tensor] = None  # Constraints (typically not used in this impl)
    
    # Reconstruction
    X_hat: Optional[torch.Tensor] = None  # Reconstruction (B, C, 32, 32, 32)
    
    # Error and discrepancy
    e: Optional[torch.Tensor] = None  # Reconstruction error: X - X_hat
    
    # Hypothesis predictions
    Pi: Optional[List[torch.Tensor]] = field(default_factory=list)  # Hypothesis predictions
    
    # Loss components
    J: float = 0.0  # Total loss
    L_rec: float = 0.0  # Reconstruction loss
    L_geo: float = 0.0  # Geometric loss
    L_cycle: float = 0.0  # Cycle consistency loss
    L_pred: float = 0.0  # Prediction loss
    L_reality: float = 0.0  # Reality comparison loss
    
    # Spectral analysis
    spec_radius: float = 0.95  # Spectral radius (contractivity condition ρ < 1)
    
    # Convergence indicators
    converged_internal: bool = False  # ||S_{n+1} - S_n|| < ε_i
    converged_external: bool = False  # d(X_world, X_hat) < ε_e
    converged_structural: bool = False  # L_Θ < ε_Θ
    
    # Execution state
    iteration: int = 0  # Micro-iteration within contraction loop
    macro_epoch: int = 1  # Outer loop iteration number
    
    # Historical tracking
    loss_history: List[float] = field(default_factory=list)
    error_history: List[float] = field(default_factory=list)
    spectral_history: List[float] = field(default_factory=list)
    
    # Metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize with default tensor if not provided."""
        if self.X is None:
            # Default 32³ seed geometry
            self.X = torch.sin(torch.randn(1, 1, 32, 32, 32)) * 0.5
        
        if self.V is None:
            self.V = self.X.clone()
        
        if self.Omega is None:
            # Latent space is 4³
            self.Omega = torch.zeros(self.X.shape[0], 8, 4, 4, 4)
    
    def reset_loss(self):
        """Zero out all loss terms."""
        self.J = 0.0
        self.L_rec = 0.0
        self.L_geo = 0.0
        self.L_cycle = 0.0
        self.L_pred = 0.0
        self.L_reality = 0.0
    
    def update_from_forward(self, forward_output: Dict):
        """Update machine state from neural substrate forward pass.
        
        Args:
            forward_output: Dict returned from NeuralSubstrate.forward()
        """
        self.Z = forward_output.get("Z")
        self.Z_star = forward_output.get("Z_star")
        self.X_hat = forward_output.get("X_recon")
        self.recon_error = forward_output.get("recon_error")
        
        if self.X_hat is not None and self.X is not None:
            self.e = self.X - self.X_hat
        
        self.Omega = forward_output.get("Omega_next", self.Omega)
        
        # Extract losses
        losses = forward_output.get("losses", {})
        self.L_rec = losses.get("L_recon", 0.0).item() if isinstance(losses.get("L_recon"), torch.Tensor) else losses.get("L_recon", 0.0)
        self.L_cycle = losses.get("L_cycle", 0.0).item() if isinstance(losses.get("L_cycle"), torch.Tensor) else losses.get("L_cycle", 0.0)
        self.L_geo = losses.get("L_geo", 0.0).item() if isinstance(losses.get("L_geo"), torch.Tensor) else losses.get("L_geo", 0.0)
        self.J = losses.get("L_total", 0.0).item() if isinstance(losses.get("L_total"), torch.Tensor) else losses.get("L_total", 0.0)
        
        # Extract metrics
        self.metrics.update(forward_output.get("metrics", {}))
    
    def record_convergence(self, eps_internal: float = 1e-4, eps_external: float = 1e-3):
        """Check and record convergence criteria.
        
        Args:
            eps_internal: Threshold for internal convergence ||S_{n+1} - S_n|| < eps
            eps_external: Threshold for external convergence d(X, X_hat) < eps
        """
        # Track loss history
        self.loss_history.append(self.J)
        if self.e is not None:
            self.error_history.append(self.e.norm().item())
        self.spectral_history.append(self.spec_radius)
        
        # Internal convergence: check loss change
        if len(self.loss_history) > 1:
            loss_change = abs(self.loss_history[-1] - self.loss_history[-2])
            self.converged_internal = loss_change < eps_internal
        
        # External convergence: check reconstruction error
        if self.e is not None:
            error_norm = self.e.norm().item()
            self.converged_external = error_norm < eps_external
        
        # Structural convergence: check spectral radius
        self.converged_structural = self.spec_radius < 1.0
    
    @property
    def is_fully_converged(self) -> bool:
        """Return True if all three convergence criteria met."""
        return (
            self.converged_internal
            and self.converged_external
            and self.converged_structural
        )
    
    def to_dict(self) -> Dict:
        """Serialize state to dictionary (tensors converted to shapes)."""
        return {
            "pc": self.pc,
            "X_shape": self.X.shape if self.X is not None else None,
            "Z_shape": self.Z.shape if self.Z is not None else None,
            "Omega_shape": self.Omega.shape if self.Omega is not None else None,
            "X_hat_shape": self.X_hat.shape if self.X_hat is not None else None,
            "J": self.J,
            "L_rec": self.L_rec,
            "L_geo": self.L_geo,
            "L_cycle": self.L_cycle,
            "spec_radius": self.spec_radius,
            "converged": self.is_fully_converged,
            "iteration": self.iteration,
            "macro_epoch": self.macro_epoch,
            "metrics": self.metrics,
        }
