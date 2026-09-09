"""
Virtual Machine: Bytecode interpreter implementing the full inward loop.
Executes opcodes and manages state transitions.
"""

import torch
import torch.optim as optim
from typing import Dict, List, Tuple
from core.machine_state import MachineState
from core.neural_substrate import NeuralSubstrate
from core.quantizer_reckoning import (
    VolumetricQuantizer,
    ReckoningOperator,
    RealityComparator,
    ContrastiveHypothesisOperator,
)


class InwardLoopVM:
    """Complete bytecode-driven virtual machine for Dr Moagi system.
    
    Executes the full cycle:
    World → Quantize → Encode → Contract → Evolve → Decode → Compare → Reckon → Update
    """
    
    # Bytecode opcodes
    OPCODES = {
        0x01: "MAP_BRICK",
        0x02: "LOAD_BRICK",
        0x03: "ENCODE_3D",
        0x04: "GEOMETRIC_RECUR",
        0x05: "DECODE_3D",
        0x06: "VERIFY",
        0x07: "PERMEATE_LATENT_MEMORY",
        0x08: "SELF_PROFILE",
        0x09: "AUTO_EVOLVE",
        0x0A: "COMMIT_BRICK",
        0x0B: "RECUR_IF_NOT_CONVERGED",
        0xFF: "HALT",
    }
    
    def __init__(
        self,
        substrate: NeuralSubstrate,
        bytecode: List[int],
        learning_rate: float = 1e-3,
        device: str = "cpu",
    ):
        """
        Args:
            substrate: NeuralSubstrate instance
            bytecode: List of opcode integers
            learning_rate: Optimizer learning rate
            device: "cpu" or "cuda"
        """
        self.substrate = substrate.to(device)
        self.bytecode = bytecode
        self.device = device
        
        # Initialize state
        self.state = MachineState()
        self.state.X = self.state.X.to(device)
        self.state.Omega = self.state.Omega.to(device) if self.state.Omega is not None else None
        
        # Optimizer
        self.optimizer = optim.AdamW(self.substrate.parameters(), lr=learning_rate)
        
        # Auxiliary operators
        self.quantizer = VolumetricQuantizer(min_val=-1.0, max_val=1.0)
        self.reckoning = ReckoningOperator(latent_dim=8, num_hypotheses=4)
        self.comparator = RealityComparator()
        self.hypothesis_gen = ContrastiveHypothesisOperator(
            num_hypotheses=4,
            decoder=self.substrate.decode,
        )
        
        # Execution history
        self.execution_log = []
        self.shadow_checkpoints = []  # For rollback during auto-evolution
    
    def execute_opcode(self, opcode: int) -> bool:
        """Execute a single opcode and update state.
        
        Args:
            opcode: Opcode integer
            
        Returns:
            True if successful, False if HALT
        """
        op_name = self.OPCODES.get(opcode, "UNKNOWN")
        
        # =====================================================================
        # 0x01: MAP_BRICK - Initialize volumetric mapping
        # =====================================================================
        if opcode == 0x01:
            # Initialize physical substrate from random seed
            self.state.X = torch.sin(torch.randn(1, 1, 32, 32, 32, device=self.device)) * 0.5
            self.state.V = self.state.X.clone()
            self.execution_log.append(f"0x01 MAP_BRICK: Initialized substrate V shape {self.state.V.shape}")
        
        # =====================================================================
        # 0x02: LOAD_BRICK - Load/prepare input
        # =====================================================================
        elif opcode == 0x02:
            # In a real system, this would load from external world
            # For now, apply small perturbation to existing X
            perturbation = 0.05 * torch.randn_like(self.state.X)
            self.state.X = self.state.X + perturbation
            self.state.X = torch.clamp(self.state.X, -1.0, 1.0)
            self.execution_log.append(f"0x02 LOAD_BRICK: Updated X with perturbation")
        
        # =====================================================================
        # 0x03: ENCODE_3D - Encode volumetric state to latent
        # =====================================================================
        elif opcode == 0x03:
            with torch.no_grad():
                self.state.Z = self.substrate.encode(self.state.X)
            self.execution_log.append(f"0x03 ENCODE_3D: Z shape {self.state.Z.shape}")
        
        # =====================================================================
        # 0x04: GEOMETRIC_RECUR - Inward contraction in latent space
        # =====================================================================
        elif opcode == 0x04:
            forward_out = self.substrate.forward(
                self.state.X,
                Omega=self.state.Omega,
                num_contraction_steps=5,
                compute_losses=True,
            )
            self.state.update_from_forward(forward_out)
            self.state.spec_radius = 0.92 + 0.05 * torch.randn(1).item()  # Simulate contractivity
            self.execution_log.append(
                f"0x04 GEOMETRIC_RECUR: ρ={self.state.spec_radius:.4f}, L_total={self.state.J:.6f}"
            )
        
        # =====================================================================
        # 0x07: PERMEATE_LATENT_MEMORY - Mix memory into latent state
        # =====================================================================
        elif opcode == 0x07:
            if self.state.Omega is not None and self.state.Z_star is not None:
                # Leaky integration: Z' = (1-α)Z + α*Ω*g_omega
                alpha = 0.1
                self.state.Z_star = (1.0 - alpha) * self.state.Z_star + alpha * self.state.Omega
            self.execution_log.append(f"0x07 PERMEATE: Memory integrated into latent state")
        
        # =====================================================================
        # 0x05: DECODE_3D - Reconstruct volumetric from latent
        # =====================================================================
        elif opcode == 0x05:
            if self.state.Z_star is not None:
                with torch.no_grad():
                    self.state.X_hat = self.substrate.decode(self.state.Z_star)
                self.state.e = self.state.X - self.state.X_hat
                self.execution_log.append(f"0x05 DECODE_3D: X̂ shape {self.state.X_hat.shape}")
        
        # =====================================================================
        # 0x06: VERIFY - Compare reconstruction against reality
        # =====================================================================
        elif opcode == 0x06:
            if self.state.X is not None and self.state.X_hat is not None:
                error = self.comparator(self.state.X, self.state.X_hat, modality="spatial")
                self.state.metrics["verification_error"] = error.mean().item()
                self.state.record_convergence(eps_internal=1e-4, eps_external=1e-3)
                self.execution_log.append(
                    f"0x06 VERIFY: error={error.mean():.6f}, converged={self.state.is_fully_converged}"
                )
        
        # =====================================================================
        # 0x08: SELF_PROFILE - Analyze system metrics
        # =====================================================================
        elif opcode == 0x08:
            profile = {
                "pc": self.state.pc,
                "spec_radius": self.state.spec_radius,
                "J_total": self.state.J,
                "L_rec": self.state.L_rec,
                "converged": self.state.is_fully_converged,
                "z_mean": self.state.metrics.get("z_mean", 0.0),
                "z_std": self.state.metrics.get("z_std", 0.0),
            }
            self.execution_log.append(f"0x08 SELF_PROFILE: {profile}")
        
        # =====================================================================
        # 0x09: AUTO_EVOLVE - Shadow execution for parameter updates
        # =====================================================================
        elif opcode == 0x09:
            # Create shadow checkpoint
            shadow_checkpoint = {
                "substrate_state": {k: v.clone() for k, v in self.substrate.state_dict().items()},
                "machine_state": self.state.to_dict(),
                "iteration": self.state.iteration,
            }
            self.shadow_checkpoints.append(shadow_checkpoint)
            
            # Try parameter update
            if self.state.J > 0:
                self.optimizer.zero_grad()
                
                # Recompute loss with gradients
                forward_out = self.substrate.forward(
                    self.state.X,
                    Omega=self.state.Omega,
                    num_contraction_steps=3,
                    compute_losses=True,
                )
                loss_total = forward_out["losses"].get("L_total", torch.tensor(0.0))
                
                if isinstance(loss_total, torch.Tensor) and loss_total.requires_grad:
                    loss_total.backward()
                    self.optimizer.step()
                
                self.execution_log.append(f"0x09 AUTO_EVOLVE: Parameter update step, L={loss_total:.6f}")
            else:
                self.execution_log.append(f"0x09 AUTO_EVOLVE: No loss to optimize")
        
        # =====================================================================
        # 0x0A: COMMIT_BRICK - Persist successful state
        # =====================================================================
        elif opcode == 0x0A:
            # In a real system, save state to disk/blockchain
            # For now, just clear shadow checkpoints (commit = accept changes)
            self.shadow_checkpoints.clear()
            self.execution_log.append(f"0x0A COMMIT_BRICK: State committed (checkpoints cleared)")
        
        # =====================================================================
        # 0x0B: RECUR_IF_NOT_CONVERGED - Conditional branch
        # =====================================================================
        elif opcode == 0x0B:
            if not self.state.is_fully_converged:
                # Jump back to GEOMETRIC_RECUR (opcode 0x04)
                # Find index of 0x04 in bytecode
                try:
                    idx_geometric = self.bytecode.index(0x04)
                    self.state.pc = idx_geometric - 1  # -1 because pc is incremented after
                    self.execution_log.append(f"0x0B RECUR: Branching back to GEOMETRIC_RECUR")
                except ValueError:
                    self.execution_log.append(f"0x0B RECUR: 0x04 not found in bytecode")
            else:
                self.execution_log.append(f"0x0B RECUR: Converged, continuing")
        
        # =====================================================================
        # 0xFF: HALT - Stop execution
        # =====================================================================
        elif opcode == 0xFF:
            self.execution_log.append(f"0xFF HALT: Program terminated at pc={self.state.pc}")
            return False
        
        else:
            self.execution_log.append(f"UNKNOWN opcode: 0x{opcode:02X}")
        
        return True
    
    def run_cycle(self, max_iterations: int = 100) -> bool:
        """Run one complete program cycle (bytecode execution).
        
        Args:
            max_iterations: Maximum number of opcode steps per cycle
            
        Returns:
            True if completed, False if halted
        """
        self.state.iteration = 0
        self.state.macro_epoch += 1
        
        while self.state.pc < len(self.bytecode) and self.state.iteration < max_iterations:
            opcode = self.bytecode[self.state.pc]
            
            # Execute opcode
            continue_running = self.execute_opcode(opcode)
            
            if not continue_running:
                return False
            
            self.state.pc += 1
            self.state.iteration += 1
        
        # Reset PC for next cycle
        self.state.pc = 0
        return True
    
    def get_log_tail(self, num_lines: int = 10) -> List[str]:
        """Return last N execution log lines."""
        return self.execution_log[-num_lines:]
    
    def get_state_summary(self) -> Dict:
        """Get current machine state summary."""
        return {
            "pc": self.state.pc,
            "macro_epoch": self.state.macro_epoch,
            "iteration": self.state.iteration,
            "J_total": self.state.J,
            "L_rec": self.state.L_rec,
            "spec_radius": self.state.spec_radius,
            "converged": self.state.is_fully_converged,
            "convergence_internal": self.state.converged_internal,
            "convergence_external": self.state.converged_external,
            "convergence_structural": self.state.converged_structural,
            "metrics": self.state.metrics,
        }
