"""
Quantizer and World Interface: Maps continuous world observations to discrete substrate.
Implements 4-bit quantization and inverse quantization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VolumetricQuantizer:
    """Maps continuous observations to 4-bit discrete voxel substrate.
    
    Each voxel q_xyz ∈ {0, ..., 15} (4 bits).
    
    Quantization formula:
        Q(x) = clip(⌊15 * (x - x_min)/(x_max - x_min) + 0.5⌋, 0, 15)
    """
    
    def __init__(self, min_val: float = -1.0, max_val: float = 1.0):
        """
        Args:
            min_val: Minimum expected value in input
            max_val: Maximum expected value in input
        """
        self.min_val = min_val
        self.max_val = max_val
        self.range = max_val - min_val
    
    def quantize(self, X: torch.Tensor) -> torch.Tensor:
        """Quantize continuous tensor to 4-bit representation.
        
        Args:
            X: Continuous tensor, shape (..., H, W, D)
            
        Returns:
            Q: Quantized tensor with integer values in [0, 15], same shape
        """
        # Normalize to [0, 1]
        X_norm = (X - self.min_val) / (self.range + 1e-8)
        X_norm = torch.clamp(X_norm, 0.0, 1.0)
        
        # Scale to [0, 15] and round
        X_q = torch.round(X_norm * 15.0)
        
        return X_q.int()
    
    def dequantize(self, Q: torch.Tensor) -> torch.Tensor:
        """Dequantize 4-bit representation back to continuous.
        
        Args:
            Q: Quantized tensor with values in [0, 15], shape (..., H, W, D)
            
        Returns:
            X: Reconstructed continuous tensor, same shape
        """
        # Normalize from [0, 15] to [0, 1]
        Q_float = Q.float()
        X_norm = Q_float / 15.0
        
        # Scale back to [min_val, max_val]
        X_recon = self.min_val + X_norm * self.range
        
        return X_recon


class ReckoningOperator(nn.Module):
    """Evaluates claims, predictions, observations, and evidence.
    
    Implements the comparison and hypothesis selection mechanism:
    
    R_t = R(X_t, X̂_t, H_t, E_t, C_t)
    
    For each hypothesis h_i:
        r_i = [c_i, p_i, o_i, e_i, s_i, q_i, c̄_i, σ_i]
    
    where:
    - c_i: claim/hypothesis
    - p_i: prediction
    - o_i: observation
    - e_i: error
    - s_i: support (evidence strength)
    - q_i: quality score
    - c̄_i: counterevidence
    - σ_i: confidence/uncertainty
    """
    
    def __init__(self, latent_dim: int = 8, num_hypotheses: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_hypotheses = num_hypotheses
        
        # Hypothesis encoder: maps latent state to hypothesis space
        self.hypothesis_encoder = nn.Sequential(
            nn.Linear(latent_dim * 4 * 4 * 4, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, num_hypotheses),  # One score per hypothesis
        )
        
        # Evidence aggregator
        self.evidence_net = nn.Sequential(
            nn.Linear(num_hypotheses * 2, 64),  # hypothesis scores + error
            nn.SiLU(),
            nn.Linear(64, num_hypotheses),  # Confidence per hypothesis
        )
        
        # Quality scorer
        self.quality_scorer = nn.Sequential(
            nn.Linear(num_hypotheses + 2, 64),  # hypotheses + complexity + uncertainty
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        X_obs: torch.Tensor,
        X_recon: torch.Tensor,
        Z: torch.Tensor,
        return_selected: bool = True,
    ) -> dict:
        """Reckon hypotheses against observations.
        
        Args:
            X_obs: Observed world state (B, C, 32, 32, 32)
            X_recon: Reconstructed state (B, C, 32, 32, 32)
            Z: Latent state (B, C_latent, 4, 4, 4)
            return_selected: If True, return best hypothesis index
            
        Returns:
            dict with:
                hypothesis_scores: Unnormalized scores per hypothesis
                confidence: Confidence in each hypothesis
                quality_scores: Quality metric per hypothesis
                selected_hypothesis: Best hypothesis index (if return_selected=True)
                evidence: Total evidence strength
                error: Reconstruction error
                claims: Claim vectors for each hypothesis
        """
        batch_size = X_obs.shape[0]
        device = X_obs.device
        
        # Flatten latent for encoding
        Z_flat = Z.view(batch_size, -1)
        
        # 1. Generate hypothesis scores from latent state
        hyp_scores = self.hypothesis_encoder(Z_flat)  # (B, num_hypotheses)
        
        # 2. Compute reconstruction error
        error = X_obs - X_recon  # (B, C, 32, 32, 32)
        error_norm = torch.norm(error.view(batch_size, -1), dim=1, keepdim=True)  # (B, 1)
        
        # 3. Aggregate evidence: combine hypothesis scores with error signal
        hyp_and_error = torch.cat([hyp_scores, error_norm.expand_as(hyp_scores)], dim=1)
        confidence = self.evidence_net(hyp_and_error)  # (B, num_hypotheses)
        confidence = F.softmax(confidence, dim=1)
        
        # 4. Score quality: penalize complexity and high uncertainty
        entropy = -torch.sum(confidence * torch.log(confidence + 1e-8), dim=1, keepdim=True)
        complexity = torch.sum(torch.abs(hyp_scores), dim=1, keepdim=True)
        quality_input = torch.cat([confidence, complexity, entropy], dim=1)
        quality_scores = self.quality_scorer(quality_input)  # (B, 1)
        
        # 5. Select best hypothesis
        selected_idx = torch.argmax(confidence, dim=1)  # (B,)
        
        # 6. Build claim records: [claim, prediction, observation, error, support, quality, counterevidence, uncertainty]
        claims = []
        for b in range(batch_size):
            claim_records = []
            for h in range(self.num_hypotheses):
                claim_record = {
                    "hypothesis_id": h,
                    "score": hyp_scores[b, h].item(),
                    "confidence": confidence[b, h].item(),
                    "quality": quality_scores[b, 0].item(),
                    "error": error_norm[b, 0].item(),
                    "entropy": entropy[b, 0].item(),
                }
                claim_records.append(claim_record)
            claims.append(claim_records)
        
        result = {
            "hypothesis_scores": hyp_scores,  # (B, num_hypotheses)
            "confidence": confidence,  # (B, num_hypotheses)
            "quality_scores": quality_scores,  # (B, 1)
            "error": error_norm,  # (B, 1)
            "entropy": entropy,  # (B, 1)
            "claims": claims,
        }
        
        if return_selected:
            result["selected_hypothesis"] = selected_idx  # (B,)
            result["best_confidence"] = torch.max(confidence, dim=1)[0]  # (B,)
        
        return result


class RealityComparator(nn.Module):
    """Compares reconstructed model against actual world observations.
    
    Implements:
        C_i = d(X_{t+1}, X̂_{t+1}^{(i)})
    
    with multimodal support (text, image, audio, video, sensor, action).
    """
    
    def __init__(self):
        super().__init__()
        
        # Modal-specific loss functions can be learned
        self.spatial_L2 = nn.Identity()  # L2 distance for spatial data
        self.temporal_DTW = nn.Identity()  # Can implement DTW for temporal
    
    def forward(
        self,
        X_actual: torch.Tensor,
        X_predicted: torch.Tensor,
        modality: str = "spatial",
    ) -> torch.Tensor:
        """Compute discrepancy between actual and predicted observations.
        
        Args:
            X_actual: Actual world observation (B, C, 32, 32, 32)
            X_predicted: Predicted/reconstructed observation (B, C, 32, 32, 32)
            modality: Type of data ("spatial", "temporal", "discrete", etc.)
            
        Returns:
            Discrepancy tensor (B,) or scalar
        """
        if modality == "spatial":
            # L2 distance for continuous spatial fields
            diff = X_actual - X_predicted
            dist = torch.norm(diff.view(diff.shape[0], -1), dim=1)
            return dist
        
        elif modality == "discrete":
            # Hamming distance for discrete/binary states
            match = torch.eq(X_actual, X_predicted).float()
            hamming = 1.0 - torch.mean(match.view(match.shape[0], -1), dim=1)
            return hamming
        
        elif modality == "temporal":
            # Could implement dynamic time warping
            dist = torch.norm((X_actual - X_predicted).view(X_actual.shape[0], -1), dim=1)
            return dist
        
        else:
            raise ValueError(f"Unknown modality: {modality}")


class ContrastiveHypothesisOperator:
    """Generates and contrasts multiple hypotheses.
    
    Given Z_t, generate K hypotheses and predict X̂_{t+1}^{(i)}.
    Then contrast against evidence.
    """
    
    def __init__(self, num_hypotheses: int = 4, decoder=None):
        """
        Args:
            num_hypotheses: Number of hypotheses to maintain
            decoder: Neural substrate decoder for reconstruction
        """
        self.num_hypotheses = num_hypotheses
        self.decoder = decoder
        self.hypothesis_cache = []
    
    def generate_hypotheses(self, Z: torch.Tensor) -> list:
        """Generate K hypotheses from latent state.
        
        Args:
            Z: Latent state (B, C, 4, 4, 4)
            
        Returns:
            List of K hypothesis tensors
        """
        batch_size = Z.shape[0]
        hypotheses = []
        
        for i in range(self.num_hypotheses):
            # Add noise or transformation to latent state
            noise_scale = 0.01 * (i + 1)  # Increasing noise per hypothesis
            Z_i = Z + noise_scale * torch.randn_like(Z)
            
            # Decode to prediction
            if self.decoder is not None:
                X_pred_i = self.decoder(Z_i)
            else:
                X_pred_i = Z_i
            
            hypotheses.append(X_pred_i)
        
        self.hypothesis_cache = hypotheses
        return hypotheses
    
    def contrast(
        self,
        hypotheses: list,
        X_actual: torch.Tensor,
        comparator: RealityComparator = None,
    ) -> dict:
        """Contrast hypotheses against actual observation.
        
        Args:
            hypotheses: List of K prediction tensors
            X_actual: Actual observation (B, C, 32, 32, 32)
            comparator: RealityComparator instance
            
        Returns:
            dict with contrast scores C_i for each hypothesis
        """
        if comparator is None:
            comparator = RealityComparator()
        
        contrast_scores = []
        for i, X_pred_i in enumerate(hypotheses):
            C_i = comparator(X_actual, X_pred_i, modality="spatial")
            contrast_scores.append(C_i)
        
        # Stack and return
        contrast_tensor = torch.stack(contrast_scores, dim=1)  # (B, num_hypotheses)
        
        return {
            "contrast_scores": contrast_tensor,
            "best_hypothesis": torch.argmin(contrast_tensor, dim=1),  # (B,)
            "mean_contrast": torch.mean(contrast_tensor, dim=1),  # (B,)
        }
