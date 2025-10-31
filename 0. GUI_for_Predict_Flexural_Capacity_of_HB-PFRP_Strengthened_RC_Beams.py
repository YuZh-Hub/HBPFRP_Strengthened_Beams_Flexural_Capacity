# -*- coding: utf-8 -*-
import sys 
import os
import numpy as np 
import shap 
import matplotlib 
import matplotlib.pyplot  as plt
from matplotlib.backends.backend_qt5agg  import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QGroupBox, QLabel, QSlider, QLineEdit, QPushButton, QTextEdit, QSplitter,
    QScrollArea, QSizePolicy, QTabWidget, QMessageBox, QFrame, QMenu, QAction, QFileDialog)
from PyQt5.QtCore import Qt, QSize 
from PyQt5.QtGui import QFont, QDoubleValidator, QIcon 
import joblib 
import shutil 
matplotlib.use('Agg')

try:
    estimator = joblib.load('1. Trained_GBDT_model.pkl') 
    scaler = joblib.load('2. Data_Scaler.pkl') 
    explainer = shap.TreeExplainer(estimator)
    feature_columns = ['$f_c$', '$b_c$', '$h_0$', '$ρ_t$', '$ρ_c$', '$f_{yt}$', 
                       '$E_f$', '$ρ_f$', '$f_f$', '$p_l$', '$N_a$', '$n$']
    print("✅ Model and scaler loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    feature_columns = ['$f_c$', '$b_c$', '$h_0$', '$ρ_t$', '$ρ_c$', '$f_{yt}$', 
                       '$E_f$', '$ρ_f$', '$f_f$', '$p_l$', '$N_a$', '$n$' ]
    estimator = None 
    scaler = None 
    explainer = None 

FEATURE_BOUNDS = {
    '$f_c$': (13.3, 62.3),      # Concrete compressive strength (MPa)
    '$b_c$': (100, 500),        # Beam width (mm)
    '$h_0$': (90, 580),         # Effective height (mm)
    '$ρ_t$': (0.124, 4.5),      # Tensile reinforcement ratio 
    '$ρ_c$': (0, 2.58),         # Compression reinforcement ratio
    '$f_{yt}$': (282, 788),     # Steel yield strength (MPa)
    '$E_f$': (17.8, 270),       # FRP elastic modulus (MPa)
    '$ρ_f$': (0.014, 1.45),     # FRP reinforcement ratio 
    '$f_f$': (411, 4950),       # FRP tensile strength (MPa)
    '$p_l$': (0, 80),           # Load level (%)
    '$N_a$': (0, 22.9),         # Bolt force (kN)
    '$n$': (0, 21),             # Number of segments
}

FEATURE_DESCRIPTIONS = {
    '$f_c$':    "01  Concrete strength (MPa)",
    '$b_c$':    "02  Beam width (mm)",
    '$h_0$':    "03  Effective height (mm)",
    '$ρ_t$':    "04  Tensile reinforcement ratio (%)",
    '$ρ_c$':    "05  Compression reinforcement ratio (%)",
    '$f_{yt}$': "06  Reinforcement yield strength (MPa)",
    '$E_f$':    "07  FRP elastic modulus (MPa)",
    '$ρ_f$':    "08  FRP reinforcement ratio (%)",
    '$f_f$':    "09  FRP tensile strength (MPa)",
    '$p_l$':    "10  FRP prestress level (%)",
    '$N_a$':    "11  Bolt anchor load (kN)",
    '$n$':      "12  Anchorage intervals",
}

class FeatureControl(QWidget):
    def __init__(self, feature_name, min_val, max_val, default_val, parent=None):
        super().__init__(parent)
        self.feature_name  = feature_name
        
        layout = QVBoxLayout()
        self.setLayout(layout) 
        
        self.label  = QLabel(FEATURE_DESCRIPTIONS[feature_name])
        self.label.setStyleSheet('''background-color: rgba(127, 140, 141, 0.1); 
                                 font-size: 15px;
                                 font-weight: bold;
                                 ''') 
        layout.addWidget(self.label) 
        
        h_layout = QHBoxLayout()
        
        self.min_label  = QLabel(f"{min_val:.2f}")
        self.min_label.setFixedWidth(50) 

        h_layout.addWidget(self.min_label) 
        
        self.slider  = QSlider(Qt.Horizontal)
        self.slider.setRange(0,  1000)  
        self.slider.setValue(self._scale_value(default_val,  min_val, max_val))
        self.slider.valueChanged.connect(self._on_slider_changed) 
        self.slider.setFixedWidth(120) 
        h_layout.addWidget(self.slider) 
        
        self.max_label  = QLabel(f"{max_val:.2f}")
        self.max_label.setFixedWidth(50) 
        self.max_label.setAlignment(Qt.AlignRight  | Qt.AlignVCenter) 
        h_layout.addWidget(self.max_label) 
        
        self.value_edit  = QLineEdit()
        self.value_edit.setFixedWidth(70) 
        self.value_edit.setAlignment(Qt.AlignCenter)
        self.value_edit.setValidator(QDoubleValidator(min_val,  max_val, 3))
        self.value_edit.setText(f"{default_val:.2f}") 
        self.value_edit.editingFinished.connect(self._on_text_changed) 
        h_layout.addWidget(self.value_edit) 
        layout.addLayout(h_layout) 
        
        self.min_val  = min_val 
        self.max_val  = max_val 
    
    def _get_unit(self, feature_name):
        if feature_name in ['$b_c$', '$h_0$']:
            return "mm"
        elif feature_name in ['$f_{yt}$', '$f_f$', '$f_c$']:
            return "MPa"
        elif feature_name == '$N_a$':
            return "kN"
        return ""
    
    def _scale_value(self, value, min_val, max_val):
        return int(((value - min_val) / (max_val - min_val)) * 1000)
    
    def _unscale_value(self, slider_value):
        return self.min_val  + (slider_value / 1000) * (self.max_val  - self.min_val) 
    
    def _on_slider_changed(self, value):
        actual_value = self._unscale_value(value)
        self.value_edit.setText(f"{actual_value:.2f}") 
    
    def _on_text_changed(self):
        try:
            value = float(self.value_edit.text()) 
            if value < self.min_val: 
                value = self.min_val  
                self.value_edit.setText(f"{value:.2f}") 
            elif value > self.max_val: 
                value = self.max_val  
                self.value_edit.setText(f"{value:.2f}") 
            self.slider.setValue(self._scale_value(value,  self.min_val,  self.max_val)) 
        except ValueError:
            pass
    
    def get_value(self):
        return float(self.value_edit.text()) 
 
class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=8, dpi=100):
        self.fig,  self.ax  = plt.subplots(figsize=(width,  height), dpi=dpi)
        super().__init__(self.fig) 
        self.setParent(parent) 
        self.setSizePolicy(QSizePolicy.Expanding,  QSizePolicy.Expanding)
        self.updateGeometry() 
        self.set_message("No  results generated yet. Please run the model.")
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu) 
        self.current_plot_path  = None  
    
    def set_message(self, message):
        self.ax.clear() 
        self.ax.text(0.5,  0.5, message, 
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=12,
                    transform=self.ax.transAxes) 
        self.ax.axis('off') 
        self.draw() 
    
    def display_plot(self, plot_file):
        if not os.path.exists(plot_file): 
            self.set_message(f"Plot  file not found: {plot_file}")
            return
        
        self.ax.clear() 
        img = plt.imread(plot_file) 
        self.ax.imshow(img,  aspect='auto', extent=[0, 1, 0, 1]) 
        self.ax.axis('off') 
        self.fig.tight_layout(pad=0) 
        self.draw() 
        self.current_plot_path  = plot_file
    
    def show_context_menu(self, pos):
        if not self.current_plot_path: 
            return
            
        menu = QMenu(self)
        save_action = QAction("Save Image As...", self)
        save_action.triggered.connect(self.save_current_image) 
        menu.addAction(save_action) 
        menu.exec_(self.mapToGlobal(pos)) 
 
    def save_current_image(self):
        if not self.current_plot_path: 
            return 
            
        file_dialog = QFileDialog()
        default_name = os.path.basename(self.current_plot_path) 
        save_path, _ = file_dialog.getSaveFileName( 
            self,
            "Save Image",
            os.path.expanduser(f"~/Desktop/{default_name}"), 
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )
        
        if save_path:
            try:
                shutil.copy2(self.current_plot_path,  save_path)
                QMessageBox.information(self,  "Success", f"Image saved to:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self,  "Error", f"Failed to save image:\n{str(e)}")
 
class PredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        font = QFont("Times New Roman") 
        matplotlib.rcParams["font.family"] = "serif"   
        matplotlib.rcParams["font.serif"] = "Times New Roman"   
        QApplication.setFont(font)   
        
        self.setWindowTitle("GUI for Predicting Flexural Capacity of HB-PFRP Strengthened RC Beams")
        self.setGeometry(100,  100, 1300, 400)
        
        self.setStyleSheet(""" 
            QMainWindow {
                background-color: #2c3e50;
            }
            QGroupBox {
                background-color: #34495e;
                border: 2px solid #3498db;
                border-radius: 8px;
                margin-top: 1ex;
                color: #ecf0f1;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                background-color: #3498db;
                color: white;
                border-radius: 4px;
            }
            QLabel {
                color: #ecf0f1;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1c6ea4;
            }
            QLineEdit {
                background-color: #ecf0f1;
                border: 1px solid #3498db;
                border-radius: 4px;
                padding: 4px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: white;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                border: 1px solid #3498db;
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            QTextEdit {
                background-color: #ecf0f1;
                border: 1px solid #3498db;
                border-radius: 4px;
                padding: 4px;
                color: #2c3e50;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget) 
        main_layout = QHBoxLayout(central_widget)
        
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter) 
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        splitter.addWidget(left_panel) 
        
        feature_group = QGroupBox("Feature Input")
        feature_group.setSizePolicy(QSizePolicy.Preferred,  QSizePolicy.Expanding)
        feature_layout = QGridLayout()
        
        self.feature_controls  = {}
        for i, feature in enumerate(feature_columns):
            min_val, max_val = FEATURE_BOUNDS[feature]
            default_val = (min_val + max_val) / 2 
            control = FeatureControl(feature, min_val, max_val, default_val)
            
            self.feature_controls[feature]  = control 
                
            row = i // 2 
            col = (i % 2) * 2
            feature_layout.addWidget(control,  row, col, 1, 2)
            
        feature_layout.setColumnStretch(0,  1)
        feature_layout.setColumnStretch(1,  1)
        feature_layout.setColumnStretch(2,  1)
        feature_layout.setColumnStretch(3,  1)
        feature_layout.setColumnStretch(4,  1)
        feature_layout.setColumnStretch(5,  1)
        
        feature_group.setLayout(feature_layout) 
        left_layout.addWidget(feature_group) 
        
        self.run_button  = QPushButton("Run Prediction and Explanation")
        self.run_button.clicked.connect(self.run_prediction) 
        self.run_button.setFont(QFont("Times New Roman",  15))
        left_layout.addWidget(self.run_button) 
        
        info_group = QGroupBox()
        info_layout = QVBoxLayout()
        self.info_text  = QTextEdit()
        self.info_text.setReadOnly(True) 
        self.info_text.setText("Application  initialized. Load model and scaler to begin.")
        self.info_text.setStyleSheet(""" 
            QTextEdit {
                background: transparent;
                border: 0px solid #3498db; 
                color: #ecf0f1; 
            }
        """)
        info_layout.addWidget(self.info_text) 
        info_group.setLayout(info_layout) 
        left_layout.addWidget(info_group) 
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        splitter.addWidget(right_panel) 
        
        results_group = QGroupBox("Prediction Results")
        results_layout = QGridLayout()
        results_layout.setContentsMargins(10,  20, 10, 10)
        
        results_layout.addWidget(QLabel("Predicted  Moment:"), 0, 0)
        self.prediction_label  = QLabel("--")
        self.prediction_label.setFont(QFont("Arial",  16, QFont.Bold))
        results_layout.addWidget(self.prediction_label,  0, 1)
        
        results_layout.addWidget(QLabel("Base  Value:"), 1, 0)
        self.base_value_label  = QLabel("--")
        results_layout.addWidget(self.base_value_label,  1, 1)
        
        results_layout.addWidget(QLabel("Status:"),  2, 0)
        self.status_label  = QLabel("Ready")
        results_layout.addWidget(self.status_label,  2, 1)
        
        results_group.setLayout(results_layout) 
        right_layout.addWidget(results_group) 
        
        self.plot_tabs  = QTabWidget()
        self.plot_tabs.setTabPosition(QTabWidget.North) 
        
        self.waterfall_canvas  = PlotCanvas(self, width=7, height=5)
        waterfall_tab = QWidget()
        waterfall_layout = QVBoxLayout(waterfall_tab)
        waterfall_layout.addWidget(self.waterfall_canvas) 
        self.plot_tabs.addTab(waterfall_tab,  "Waterfall Plot ")
        
        self.decision_canvas  = PlotCanvas(self, width=7, height=5)
        decision_tab = QWidget()
        decision_layout = QVBoxLayout(decision_tab)
        decision_layout.addWidget(self.decision_canvas) 
        self.plot_tabs.addTab(decision_tab,  "Decision Plot ")
        
        self.contribution_text  = QTextEdit()
        self.contribution_text.setReadOnly(True) 
        self.contribution_text.setFont(QFont("Times New Roman",  10))
        self.plot_tabs.addTab(self.contribution_text,  "Feature Contributions")
        
        right_layout.addWidget(self.plot_tabs) 
        
        splitter.setSizes([700,  600])
        
        self.generated_plots  = []
        
        if estimator is None:
            self.update_info("Error:  Could not load model or scaler. Running in demo mode.")
            self.run_button.setEnabled(False) 
    
    def update_info(self, message):
        current_text = self.info_text.toPlainText() 
        self.info_text.setText(f"{current_text}\n{message}") 
        self.info_text.verticalScrollBar().setValue( 
            self.info_text.verticalScrollBar().maximum()) 
    
    def explain_external_sample(self, sample_data):
        if estimator is None or scaler is None or explainer is None:
            return {
                'prediction': 128.5,
                'base_value': 115.2,
                'plots': [('waterfall', 'waterfall_demo.png'),  ('decision', 'decision_demo.png')], 
                'feature_contributions': {
                    f: {'value': v, 'shap_contribution': np.random.uniform(-5,  5), 'impact': "Increase" if np.random.rand()  > 0.5 else "Decrease"} 
                    for f, v in sample_data.items() 
                }
            }
        
        missing = set(feature_columns) - set(sample_data.keys()) 
        if missing:
            raise ValueError(f"Missing features: {', '.join(missing)}")
        
        ordered_sample = np.array([[sample_data[col]  for col in feature_columns]])
        scaled_sample = scaler.transform(ordered_sample) 
        
        try:
            # New API (0.44.0+)
            sample_shap = explainer(scaled_sample, check_additivity=False)
            base_values = sample_shap.base_values 
            shap_values_array = sample_shap.values 
        except AttributeError:
            # Old version compatibility (0.40.x)
            base_values = explainer.expected_value  
            shap_values_array = explainer.shap_values(scaled_sample) 
        
        model_output = estimator.predict(scaled_sample) 
        n_outputs = 1 if len(model_output.shape)  == 1 else model_output.shape[1] 
        predictions = model_output[0] if n_outputs == 1 else model_output[0]
        
        plot_files = []
        os.makedirs("temp_plots",  exist_ok=True)
        
        # 1. Waterfall plot 
        plt.rcParams['font.family']  = 'Times New Roman'
        for output_idx in range(n_outputs):
            try:
                plt.figure(figsize=(5,  3))
                
                if isinstance(shap_values_array, list):
                    curr_shap = shap_values_array[output_idx][0]
                elif len(shap_values_array.shape)  == 3:
                    curr_shap = shap_values_array[0, output_idx, :]
                else:
                    curr_shap = shap_values_array[0]
                
                if isinstance(base_values, (list, np.ndarray)): 
                    curr_base = base_values[output_idx]
                else:
                    curr_base = base_values
                
                shap.waterfall_plot( 
                    shap.Explanation(
                        values=curr_shap,
                        base_values=curr_base,
                        data=ordered_sample[0],
                        feature_names=feature_columns
                    ),
                    max_display=min(12, len(feature_columns)),
                    show=False
                )

                plt.tight_layout() 
                
                waterfall_file = f"temp_plots/waterfall_output{output_idx+1}.png"
                plt.savefig(waterfall_file,  dpi=300, bbox_inches='tight')
                plt.close() 
                
                plot_files.append(('waterfall',  waterfall_file))
            except Exception as e:
                self.update_info(f"Waterfall  plot error: {str(e)}")
        
        # 2. Decision plot 
        plt.rcParams['font.family']  = 'Times New Roman'
        try:
            plt.figure(figsize=(10,  8))
            
            if len(shap_values_array.shape)  == 2:
                shap_3d = shap_values_array.reshape(1,  1, -1)
            elif len(shap_values_array.shape)  == 1:
                shap_3d = shap_values_array.reshape(1,  1, -1)
            else:
                shap_3d = shap_values_array
            
            if not isinstance(base_values, (list, np.ndarray)): 
                base_array = np.array([base_values]) 
            else:
                base_array = np.array(base_values) 
            
            shap.decision_plot( 
                base_array, 
                shap_3d[0],
                features=ordered_sample[0],
                feature_names=feature_columns,
                highlight=None,
                ignore_warnings=True,
                show=False
            )

            plt.tight_layout() 
            
            decision_file = "temp_plots/decision_plot.png" 
            plt.savefig(decision_file,  dpi=300, bbox_inches='tight')
            plt.close() 
            
            plot_files.append(('decision',  decision_file))
        except Exception as e:
            self.update_info(f"Decision  plot error: {str(e)}")
            try:
                plt.figure(figsize=(10,  6))
                
                if len(shap_values_array) > 0:
                    if isinstance(shap_values_array, list):
                        use_shap = shap_values_array[0][0]
                    else:
                        use_shap = shap_values_array[0]
                    
                    use_base = base_values[0] if isinstance(base_values, (list, np.ndarray))  else base_values 
                    
                    shap.bar_plot( 
                        shap.Explanation(
                            values=use_shap,
                            base_values=use_base,
                            feature_names=feature_columns
                        ),
                        show=False
                    )
                    
                    plt.tight_layout() 
                    
                    bar_file = "temp_plots/bar_plot.png" 
                    plt.savefig(bar_file,  dpi=300, bbox_inches='tight')
                    plt.close() 
                    
                    plot_files.append(('bar',  bar_file))
            except Exception as e2:
                self.update_info(f"All  visualization methods failed: {str(e2)}")
        
        return {
            'prediction': float(predictions[0]) if n_outputs > 1 else float(predictions),
            'base_value': float(base_values[0]) if isinstance(base_values, (list, np.ndarray))  else float(base_values),
            'plots': plot_files,
            'feature_contributions': {
                feature: {
                    'value': float(ordered_sample[0][i]),
                    'shap_contribution': float(shap_values_array[0][i]) if len(shap_values_array.shape)  == 2 else float(shap_values_array[0, 0, i]),
                    'impact': "Increase" if (shap_values_array[0][i] if len(shap_values_array.shape)  == 2 else shap_values_array[0, 0, i]) > 0 else "Decrease",
                } for i, feature in enumerate(feature_columns)
            }
        }
    
    def run_prediction(self):
        self.waterfall_canvas.set_message("Generating  waterfall plot...")
        self.decision_canvas.set_message("Generating  decision plot...")
        self.contribution_text.clear() 
        self.status_label.setText("Running...") 
        QApplication.processEvents() 
        
        sample_data = {}
        for feature, control in self.feature_controls.items(): 
            sample_data[feature] = control.get_value() 
        
        try:
            self.update_info("Starting  prediction and explanation...")
            result = self.explain_external_sample(sample_data) 
            
            prediction = result['prediction']
            base_value = result['base_value']
            self.prediction_label.setText(f"{prediction:.2f}  kN·m")
            self.base_value_label.setText(f"{base_value:.2f}") 
            self.status_label.setText("Completed") 
            
            waterfall_plot = None 
            decision_plot = None
            
            for plot_type, plot_file in result['plots']:
                if plot_type == 'waterfall' and waterfall_plot is None:
                    waterfall_plot = plot_file 
                elif plot_type in ['decision', 'bar'] and decision_plot is None:
                    decision_plot = plot_file 
            
            if waterfall_plot:
                self.waterfall_canvas.display_plot(waterfall_plot) 
                self.generated_plots.append(waterfall_plot) 
            
            if decision_plot:
                self.decision_canvas.display_plot(decision_plot) 
                self.generated_plots.append(decision_plot) 
            
            contrib_text = "Feature Contributions:\n"
            contrib_text += "----------------------------------------\n"
            contrib_text += "Feature\t\tValue\tContribution\tImpact\n"
            contrib_text += "----------------------------------------\n"
            
            for feature, contrib in result['feature_contributions'].items():
                desc = FEATURE_DESCRIPTIONS[feature]
                value = contrib['value']
                contribution = contrib['shap_contribution']
                impact = contrib['impact']
                contrib_text += f"{desc[:25]:<25}\t{value:.2f}\t{contribution:+.4f}\t{impact}\n"
            
            self.contribution_text.setText(contrib_text) 
            
            self.update_info("Prediction  and explanation completed successfully!")
        except Exception as e:
            self.status_label.setText("Error") 
            self.update_info(f"Error:  {str(e)}")
            QMessageBox.critical(self,  "Error", f"An error occurred:\n{str(e)}")
    
    def closeEvent(self, event):
        for plot_file in self.generated_plots: 
            try:
                if os.path.exists(plot_file): 
                    os.remove(plot_file) 
            except:
                pass 
        
        try:
            if os.path.exists("temp_plots")  and not os.listdir("temp_plots"): 
                os.rmdir("temp_plots") 
        except:
            pass
        
        event.accept()
 
if __name__ == "__main__":
    app = QApplication(sys.argv) 
    window = PredictionApp()
    window.show() 
    sys.exit(app.exec_()) 