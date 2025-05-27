import sys
import numpy as np
import torch
from pylint.config import save_results
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from PyQt6 import QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtCore import QTimer
import pickle
from packages_own.NSGAII import MultiObjectiveProblem, select_uniform_points
# Assuming custom modules are properly implemented
from packages_own.models import Regressor, expotionalp, calculate_linear_properties
from packages_own.ensemble_models import EnsembleRegressor
from packages_own.Navigation import Navigation
from packages_ui.main_window import Ui_Form
import csv
from matplotlib import cm
from packages_ui.pics import backgr_rc

class OptimizationApp(QtWidgets.QWidget):
    def __init__(self, select_number=1):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # Initialize variables
        self.history = []
        self.best_front_curves = []
        self.current_gen = 0
        self.select_number = select_number

        # Setup UI interactions
        self.ui.pushButton.clicked.connect(self.start_optimization)
        self.init_plots()

    def init_plots(self):
        """Initialize matplotlib figure and canvas"""
        self.fig2 = Figure(figsize=(3, 3))
        self.fig2.subplots_adjust(left=0.15, right=0.95, bottom=0.17, top=0.9)
        self.canvas2 = FigureCanvas(self.fig2)
        layout = QtWidgets.QVBoxLayout(self.ui.frame)
        layout.addWidget(self.canvas2)
        self.ax2 = self.fig2.add_subplot(111)
        # Set plot formatting
        self.ax2.set_title("Optimization Process Visualization")
        self.ax2.set_xlabel("Linearity")
        self.ax2.set_ylabel("Sensitivity (V/kPa)")

    def start_optimization(self):
        """Start optimization process"""
        try:
            self.target_pressure = []
            # Get target pressure range
            start_point = float(self.ui.textEdit.toPlainText())
            end_point = float(self.ui.textEdit_2.toPlainText())
            self.target_pressure = np.linspace(start_point, end_point, 25)
            sensitivity_limit = float(self.ui.textEdit_3.toPlainText())
            linearity_limit = float(self.ui.textEdit_4.toPlainText())

            # Reset state
            self.history.clear()
            self.best_front_curves.clear()
            self.current_gen = 0
            self.ax2.clear()
            self.ax2.set_title("Optimization Process Visualization")
            self.ax2.set_xlabel("Linearity")
            self.ax2.set_ylabel("Sensitivity (V/kPa)")

            # Define optimization problem
            problem = MultiObjectiveProblem(self.target_pressure, ensemble, svm_model, sensitivity_limit, linearity_limit)

            # Configure NSGA-II algorithm
            algorithm = NSGA2(
                pop_size=50,
                eliminate_duplicates=True,
            )

            def optimization_callback(algorithm):
                """Callback to record generation data"""
                gen_data = {
                    "params": algorithm.pop.get("X")[::],
                    "objectives": algorithm.pop.get("F")[::],
                    "generation": algorithm.n_gen
                }
                self.history.append(gen_data)

            # Run optimization
            res = minimize(
                problem,
                algorithm,
                ('n_gen', 50),
                seed=42,
                verbose=True,
                callback=optimization_callback
            )
            print(-res.F[-1])

            # Start visualization updates
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_visualization_parato)
            self.timer.start(500)  # Update every 500ms

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Optimization failed: {str(e)}")

    def update_visualization_parato(self):
        """Update Pareto front visualization"""
        if self.current_gen >= len(self.history):
            self.timer.stop()
            final_pareto_front = self.history[self.current_gen-1]["objectives"]
            selected_params_last = self.history[self.current_gen - 1]["params"]
            select_indices = select_uniform_points(final_pareto_front)
            selected_optimal_six = selected_params_last[select_indices]
            selected_optimal_six_denorm = selected_optimal_six * (navigator.param_max - navigator.param_min) + navigator.param_min

            select_points = -final_pareto_front[select_indices]
            self.ax2.scatter(
                select_points[:, 0], select_points[:, 1],
                c='red', alpha=0.8, edgecolors='none', label='recommended parameters', s=50
            )
            self.canvas2.draw()
            self.display_final_results(selected_optimal_six_denorm[self.select_number], select_points[self.select_number:, 0],
                                      select_points[self.select_number:, 1])
            with open(f'./data_cache/data/selected_params{self.target_pressure[-1]}.csv', 'w',
                      newline='') as f:
                writer = csv.writer(f)
                writer.writerow(
                    ['crosslinking', 'height', 'side length', 'density', 'R^2', 'Slope', 'SVM Success Rate'])
                for params, (r2, slope) in zip(selected_optimal_six_denorm, select_points):
                    svm_success_rate = svm_model.predict_proba([[params[1], params[2]]])[0][1]
                    writer.writerow([*params, r2, slope, svm_success_rate])
            return

        gen_data = self.history[self.current_gen]
        F = -gen_data["objectives"]

        # Fade previous points
        for collection in self.ax2.collections:
            collection.set_color('lightblue')

        # Plot new Pareto front
        self.ax2.scatter(
            F[:, 0], F[:, 1],
            c='royalblue', alpha=0.8, edgecolors='none',
            label=f'pareto frontier'
        )
        if self.current_gen == 0:
            self.ax2.legend(loc='lower left')
        self.canvas2.draw()
        self.current_gen += 1

    def display_final_results(self, best_params_original, slope, r_squared):
        """Display final optimization results"""
        print(best_params_original)
        self.ui.lineEdit.setText(f'{best_params_original[0]:.3f}')
        self.ui.lineEdit_2.setText(f'{best_params_original[1]:.2f}')
        self.ui.lineEdit_3.setText(f'{best_params_original[2]:.2f}')
        self.ui.lineEdit_4.setText(f'{best_params_original[3]:.2f}')

if __name__ == "__main__":
    navigator = Navigation()
    ensemble = EnsembleRegressor(num_models=5, model_class=Regressor)
    ensemble.load_model_weights(loop=9)
    # Load pre-trained SVM model
    with open('params/SVM_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    app = QtWidgets.QApplication(sys.argv)
    window = OptimizationApp(select_number=0)  # Corresponding to largest linearity in Pareto front
    window.setWindowTitle("NSGA-II Material Design Optimizer")
    window.show()
    sys.exit(app.exec())