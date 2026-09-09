"""
PyQt5 Dashboard: Real-time visualization of the Dr Moagi inward loop machine.
Displays 3D geometry, state metrics, execution logs, and convergence tracking.
"""

import sys
import torch
import numpy as np
import random
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QProgressBar, QTextEdit, QGroupBox, QGridLayout, QPushButton
)
from PyQt5.QtGui import QFont
from PyQt5.QtDataVisualization import Q3DScatter, QScatter3DSeries, QScatterDataItem

from core.neural_substrate import NeuralSubstrate
from core.machine_state import MachineState
from vm.inward_loop_vm import InwardLoopVM


class InwardLoopDashboard(QMainWindow):
    """Real-time visualization dashboard for Dr Moagi system."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dr Moagi Billion³ Geometric Intelligence - Live Inward Loop")
        self.resize(1400, 800)
        
        # Stylesheet
        self.setStyleSheet("""
            QMainWindow { background-color: #05070F; }
            QGroupBox {
                background-color: #0B0E17; border: 1px solid #1A2333;
                border-radius: 6px; margin-top: 12px; font-weight: bold; color: #0EA5E9;
            }
            QLabel { color: #94A3B8; font-family: 'Courier New'; font-size: 11px; }
            QTextEdit {
                background-color: #010307; color: #10B981; border: 1px solid #1A2333;
                font-family: 'Courier New'; font-size: 10px;
            }
            QProgressBar { border: 1px solid #1A2333; background-color: #010307; text-align: center; color: #FFF; }
            QProgressBar::chunk { background-color: #0284C7; }
            QPushButton {
                background-color: #0284C7; color: white; border: none; padding: 6px 12px;
                border-radius: 4px; font-weight: bold;
            }
            QPushButton:hover { background-color: #0369A1; }
        """)
        
        # Initialize neural substrate and VM
        self.substrate = NeuralSubstrate(in_channels=1, latent_channels=8, spatial_size=32)
        self.bytecode = [0x01, 0x02, 0x03, 0x04, 0x07, 0x04, 0x07, 0x04, 0x07, 0x05, 0x06, 0x08, 0x09, 0x0A, 0x0B, 0xFF]
        self.vm = InwardLoopVM(self.substrate, self.bytecode, learning_rate=1e-3, device="cpu")
        
        # UI components
        self.init_ui()
        
        # Execution timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.evolve_vm_one_step)
        self.timer.start(250)  # 250ms per cycle
    
    def init_ui(self):
        """Initialize user interface layout."""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # =====================================================================
        # LEFT PANEL: State & Logs
        # =====================================================================
        left_panel = QVBoxLayout()
        
        title = QLabel("INWARD LOOP VM - STATE & EXECUTION")
        title.setFont(QFont("Courier New", 13, QFont.Bold))
        title.setStyleSheet("color: #0EA5E9;")
        left_panel.addWidget(title)
        
        # State metrics box
        state_box = QGroupBox("Machine State Vector S_k")
        grid = QGridLayout(state_box)
        
        self.lbl_pc = QLabel("pc_k       : 0x00")
        self.lbl_epoch = QLabel("Epoch     : 1")
        self.lbl_loss = QLabel("J_total   : 0.0000")
        self.lbl_rec = QLabel("L_rec     : 0.0000")
        self.lbl_radius = QLabel("ρ(J_T)    : 0.9500")
        self.lbl_converge = QLabel("Convergence: ◯ ◯ ◯")
        
        grid.addWidget(self.lbl_pc, 0, 0)
        grid.addWidget(self.lbl_epoch, 0, 1)
        grid.addWidget(self.lbl_loss, 1, 0)
        grid.addWidget(self.lbl_rec, 1, 1)
        grid.addWidget(self.lbl_radius, 2, 0)
        grid.addWidget(self.lbl_converge, 2, 1)
        
        left_panel.addWidget(state_box)
        
        # Convergence progress
        conv_box = QGroupBox("Convergence Indicators")
        conv_layout = QGridLayout(conv_box)
        
        self.prog_internal = QProgressBar()
        self.prog_external = QProgressBar()
        self.prog_structural = QProgressBar()
        
        for pb in [self.prog_internal, self.prog_external, self.prog_structural]:
            pb.setRange(0, 100)
            pb.setValue(0)
        
        conv_layout.addWidget(QLabel("Internal (||ΔS||)"), 0, 0)
        conv_layout.addWidget(self.prog_internal, 0, 1)
        conv_layout.addWidget(QLabel("External (d(X,X̂))"), 1, 0)
        conv_layout.addWidget(self.prog_external, 1, 1)
        conv_layout.addWidget(QLabel("Structural (ρ<1)"), 2, 0)
        conv_layout.addWidget(self.prog_structural, 2, 1)
        
        left_panel.addWidget(conv_box)
        
        # Execution log
        log_box = QGroupBox("VM Execution Log")
        log_layout = QVBoxLayout(log_box)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(250)
        log_layout.addWidget(self.log_text)
        
        left_panel.addWidget(log_box)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.btn_start = QPushButton("▶ Start")
        self.btn_pause = QPushButton("⏸ Pause")
        self.btn_reset = QPushButton("⟲ Reset")
        
        self.btn_start.clicked.connect(lambda: self.timer.start())
        self.btn_pause.clicked.connect(lambda: self.timer.stop())
        self.btn_reset.clicked.connect(self.reset_system)
        
        button_layout.addWidget(self.btn_start)
        button_layout.addWidget(self.btn_pause)
        button_layout.addWidget(self.btn_reset)
        
        left_panel.addLayout(button_layout)
        left_panel.addStretch()
        
        layout.addLayout(left_panel, stretch=1)
        
        # =====================================================================
        # RIGHT PANEL: 3D Visualization
        # =====================================================================
        right_box = QGroupBox("Active Latent Manifold - Z* Geometry")
        right_layout = QVBoxLayout(right_box)
        
        self.scatter = Q3DScatter()
        self.container = QWidget.createWindowContainer(self.scatter)
        right_layout.addWidget(self.container)
        
        # Configure 3D axes
        self.scatter.activeAxisX().setRange(0, 32)
        self.scatter.activeAxisY().setRange(0, 32)
        self.scatter.activeAxisZ().setRange(0, 32)
        self.scatter.setShadowQuality(Q3DScatter.ShadowQualityNone)
        
        # Add series
        self.series = QScatter3DSeries()
        self.series.setItemSize(0.18)
        self.scatter.addSeries(self.series)
        
        layout.addWidget(right_box, stretch=2)
    
    def push_tensor_to_3d_scatter(self, tensor_3d: torch.Tensor):
        """Extract and visualize significant spatial coordinates."""
        if tensor_3d is None or tensor_3d.numel() == 0:
            return
        
        arr = tensor_3d[0, 0].detach().cpu().numpy()
        
        # Sample indices where activation > threshold
        indices = np.argwhere(np.abs(arr) > 0.15)
        
        # Cap to avoid visualization overhead
        if len(indices) > 600:
            indices = indices[random.sample(range(len(indices)), 600)]
        
        # Create 3D scatter items
        data_items = [
            QScatterDataItem(np.array([idx[0], idx[1], idx[2]], dtype=np.float32))
            for idx in indices
        ]
        
        if data_items:
            self.series.dataProxy().resetArray(data_items)
    
    def evolve_vm_one_step(self):
        """Execute one VM cycle and update dashboard."""
        # Run one complete bytecode cycle
        self.vm.run_cycle(max_iterations=50)
        
        # Get current state
        summary = self.vm.get_state_summary()
        state = self.vm.state
        
        # Update UI labels
        self.lbl_pc.setText(f"pc_k       : 0x{summary['pc']:02X}")
        self.lbl_epoch.setText(f"Epoch     : {summary['macro_epoch']}")
        self.lbl_loss.setText(f"J_total   : {summary['J_total']:.6f}")
        self.lbl_rec.setText(f"L_rec     : {summary['L_rec']:.6f}")
        self.lbl_radius.setText(f"ρ(J_T)    : {summary['spec_radius']:.4f}")
        
        # Convergence symbols
        sym_internal = "✓" if state.converged_internal else "◯"
        sym_external = "✓" if state.converged_external else "◯"
        sym_structural = "✓" if state.converged_structural else "◯"
        self.lbl_converge.setText(f"Convergence: {sym_internal} {sym_external} {sym_structural}")
        
        # Update progress bars
        internal_progress = min(100, int(100 * (1.0 - abs(summary["J_total"]))))
        external_progress = min(100, int(100 * (1.0 - summary["metrics"].get("recon_mse", 1.0))))
        structural_progress = min(100, int(100 * (1.0 - summary["spec_radius"])))
        
        self.prog_internal.setValue(internal_progress)
        self.prog_external.setValue(external_progress)
        self.prog_structural.setValue(structural_progress)
        
        # Update log
        log_lines = self.vm.get_log_tail(5)
        self.log_text.clear()
        for line in log_lines:
            self.log_text.append(line)
        
        # Update 3D visualization
        if state.Z_star is not None:
            self.push_tensor_to_3d_scatter(state.Z_star)
    
    def reset_system(self):
        """Reset VM to initial state."""
        self.substrate = NeuralSubstrate(in_channels=1, latent_channels=8, spatial_size=32)
        self.vm = InwardLoopVM(self.substrate, self.bytecode, learning_rate=1e-3, device="cpu")
        
        self.log_text.clear()
        self.log_text.append("System reset. Ready to run.")
        
        self.lbl_pc.setText("pc_k       : 0x00")
        self.lbl_epoch.setText("Epoch     : 1")
        self.lbl_loss.setText("J_total   : 0.0000")
        self.lbl_converge.setText("Convergence: ◯ ◯ ◯")
        
        for pb in [self.prog_internal, self.prog_external, self.prog_structural]:
            pb.setValue(0)


def main():
    """Launch the dashboard application."""
    app = QApplication(sys.argv)
    dashboard = InwardLoopDashboard()
    dashboard.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
