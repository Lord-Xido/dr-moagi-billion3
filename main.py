#!/usr/bin/env python3
"""
Dr Moagi Billion³ - Main Entry Point

Launches the complete inward loop system with PyQt5 dashboard visualization.
"""

import sys
import torch

from ui.dashboard import InwardLoopDashboard
from PyQt5.QtWidgets import QApplication


def main():
    """Launch the Dr Moagi system."""
    print("="*70)
    print("Dr Moagi Billion³ Geometric Intelligence System")
    print("Closed 3D Recursive Volumetric Autoencoder")
    print("="*70)
    print()
    print("Initializing neural substrate...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()
    
    app = QApplication(sys.argv)
    dashboard = InwardLoopDashboard()
    
    print("Dashboard initialized. Opening window...")
    print()
    print("Controls:")
    print("  ▶ Start  - Begin VM execution")
    print("  ⏸ Pause - Pause execution")
    print("  ⟲ Reset - Reset system to initial state")
    print()
    print("System Status:")
    print("  - Left panel: State metrics, convergence, execution log")
    print("  - Right panel: 3D latent manifold visualization")
    print()
    
    dashboard.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
