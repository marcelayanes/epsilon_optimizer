# -*- coding: utf-8 -*-
"""CIFAR-10 Benchmark Execution Script (Epsilon vs Adam vs AMSGrad vs Adamax)

Reproducibility script for the research paper:
"Epsilon: An adaptive optimisation algorithm based on directional gradient similarity for image classification"
"""

# Instala todas las dependencias necesarias

# Reinicia el kernel después de instalar
from IPython.core.display import HTML
HTML("<script>Jupyter.notebook.kernel.restart()</script>")

# Celda de verificación
import torch
import torchvision
import sklearn
import thop
import pynvml
import matplotlib
import seaborn
import tqdm

print("✅ Todas las librerías instaladas correctamente")
print(f"PyTorch: {torch.__version__}")
print(f"TorchVision: {torchvision.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# =============================================================
# INSTALLATION, IMPORTS, AND GLOBAL CONFIGURATION
# =============================================================

# Install necessary libraries if not already present

# Import standard libraries for deep learning and data manipulation
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import os
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F 
from sklearn.preprocessing import label_binarize
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
import seaborn as sns
import thop
import pynvml

# =============================================================
# GLOBAL HYPERPARAMETERS AND DEVICE CONFIGURATION
# =============================================================

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Global hyperparameters for experiments
NUM_CLASSES = 10         # CIFAR-10 has 10 classes
num_classes = 10
NUM_EPOCHS = 100         # Set training epochs to 100
BATCH_SIZE = 128         # Set minibatch size to 128
EARLY_STOPPING_PATIENCE = 10 # Stop if validation loss doesn't improve for 10 epochs

# Standardized hyperparameters for all optimizers
HYPERPARAMS = {
    'lr': 0.0001,
    'betas': (0.9, 0.99),    # Se usa solo betas[0] como beta_max
    'eps': 1e-8,
    'alpha': 0.85,           # Sensibilidad momentum dinámico
    'beta_min': 0.5,         # Límite inferior para beta adaptativo
    'gamma': 0.5             # Balance para normalización del paso
}


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================
# DATASET AND DATALOADERS SETUP (CIFAR-10 WITH AUGMENTATION)
# =============================================================

# Define transformations with Data Augmentation for the training set
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# Define transformations for the test/validation set (without augmentation)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# Download and load the CIFAR-10 dataset
print("Downloading and preparing CIFAR-10 dataset...")
full_train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=train_transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=test_transform
)

# Split the training set into training and validation sets (45k train, 5k val)
train_size = 45000
val_size = len(full_train_dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(
    full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)
val_loader = torch.utils.data.DataLoader(
    val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)

# Define class names for CIFAR-10
class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print("Dataset prepared successfully.")

import torch
from torch.optim.optimizer import Optimizer

class Epsilon(Optimizer):
    def __init__(self, params, lr=1e-3, beta_max=0.9, alpha=10.0, epsilon=1e-8, beta_min=0.5, gamma=0.5):
        """
        Args:
            lr: learning rate
            beta_max: máximo valor de momentum decay
            alpha: sensibilidad para beta_t (sigmoide)
            epsilon: estabilidad numérica
            beta_min: mínimo valor clip para beta_t
            gamma: factor de balance para normalización que combina norma gradiente y momentum
        """
        defaults = dict(lr=lr, beta_max=beta_max, alpha=alpha, epsilon=epsilon, beta_min=beta_min, gamma=gamma)
        super(Epsilon, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta_max = group['beta_max']
            alpha = group['alpha']
            eps = group['epsilon']
            beta_min = group['beta_min']
            gamma = group['gamma']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # Inicialización de estado
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p.data)
                    state['prev_grad'] = torch.zeros_like(p.data)
                    state['norm_factor'] = torch.tensor(eps, device=p.device)

                m = state['momentum']
                prev_g = state['prev_grad']
                norm_factor = state['norm_factor']

                # Similitud direccional con coseno entre gradientes consecutivos
                grad_norm = grad.norm()
                prev_g_norm = prev_g.norm()
                denom = (grad_norm * prev_g_norm).clamp_min(eps)
                cos_dir = (grad * prev_g).sum() / denom

                # Cálculo de beta_t con sigmoid y clip
                beta_t_raw = beta_max * (1 / (1 + torch.exp(-alpha * cos_dir)))
                beta_t = torch.clamp(beta_t_raw, min=beta_min, max=beta_max)

                # Actualización momentum adaptativo
                m.mul_(beta_t).add_(grad, alpha=1 - beta_t)

                # Actualización media móvil para normalización
                state['norm_factor'] = norm_factor.mul(0.9).add(grad_norm * 0.1)

                # Normalización modificada usando norma gradiente y momentum
                s_t = grad_norm / (gamma + (1 - gamma) * m.norm() + eps)

                # Actualización parámetros: paso en dirección signo momentum escalado 
                p.data.addcmul_(m.sign(), s_t * (-lr))

                # Guardar gradiente actual para el siguiente paso
                state['prev_grad'] = grad.clone()

        return loss


# Bloque 3: Definición del modelo CNN
class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()
        self.features = nn.Sequential(
            # Bloque 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Bloque 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Bloque 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# =============================================================
# CORE TRAINING AND EVALUATION FUNCTION (MODIFIED)
# =============================================================

def train_and_evaluate_model(model, optimizer_name, hparams):
    """
    Orchestrate the complete training and evaluation process, now without weight_decay.
    """
    model.to(device)

    # Select the optimizer based on its name
    if optimizer_name == "Epsilon":
        optimizer = Epsilon(model.parameters(), lr=HYPERPARAMS['lr'], beta_max=HYPERPARAMS['betas'][0], epsilon=HYPERPARAMS['eps'], alpha=HYPERPARAMS['alpha'], beta_min=HYPERPARAMS['beta_min'], gamma=HYPERPARAMS['gamma'])
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=hparams['lr'], betas=hparams['betas'], eps=hparams['eps'])
    elif optimizer_name == "Adamax":
        optimizer = optim.Adamax(model.parameters(), lr=hparams['lr'], betas=hparams['betas'], eps=hparams['eps'])
    # Corregido para que coincida con el nombre en tu script principal (AMSGrad)
    elif optimizer_name == "AMSGrad": 
        # El optimizador correcto es Adam con amsgrad=True
        optimizer = optim.Adam(model.parameters(), lr=hparams['lr'], betas=hparams['betas'], eps=hparams['eps'], amsgrad=True)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' not recognized.")

    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

    # El diccionario history ya se estaba creando y llenando correctamente
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_auc': [], 'epoch_times': []}
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()

    print(f"\n--- Starting Training: {optimizer_name} ---")

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")

        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            train_pbar.set_postfix({'Loss': running_loss/total_train, 'Acc': correct_train/total_train * 100})
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / total_train
        
        # Validation phase
        model.eval()
        running_val_loss, correct_val, total_val = 0.0, 0, 0
        
        all_val_labels = []
        all_val_scores = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                all_val_labels.extend(labels.cpu().numpy())
                all_val_scores.extend(F.softmax(outputs, dim=1).cpu().numpy())

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val / total_val
        
        y_true_bin_val = label_binarize(all_val_labels, classes=range(NUM_CLASSES))
        # Ensure correct shape for roc_auc_score
        if y_true_bin_val.shape[1] < NUM_CLASSES:
             y_true_bin_val = label_binarize(all_val_labels, classes=list(range(NUM_CLASSES)))
        epoch_val_auc = roc_auc_score(y_true_bin_val, all_val_scores, average='macro', multi_class='ovr')

        epoch_time = time.time() - epoch_start_time
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['val_auc'].append(epoch_val_auc)
        history['epoch_times'].append(epoch_time)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, Val AUC: {epoch_val_auc:.4f}")

        scheduler.step()

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"cifar100best_model_{optimizer_name}.pth")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"--- Early stopping triggered at epoch {epoch+1} ---")
                break
    
    total_time = time.time() - start_time
    print(f"--- Training Finished for {optimizer_name} in {total_time:.2f}s ---")


    # Load the best model for final testing
    # Corregido para que cargue desde la CPU si es necesario
    model.load_state_dict(torch.load(f"cifar100best_model_{optimizer_name}.pth", map_location=device))

    # Final testing on test set
    model.eval()
    y_true, y_scores, y_pred = [], [], []
    test_loss, correct_test, total_test = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_scores.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    final_test_loss = test_loss / len(test_loader.dataset)
    final_test_acc = correct_test / total_test

    # --- ÚNICO CAMBIO NECESARIO AQUÍ ---
    # 1. Empaqueta los resultados finales en un diccionario.
    final_results = {
        "optimizer": optimizer_name,
        "test_loss": final_test_loss, "test_accuracy": final_test_acc,
        "y_true": y_true, "y_pred": y_pred, "y_scores": y_scores,
        "training_time": total_time, "params": count_parameters(model)
    }

    # 2. Devuelve los resultados finales Y el historial por separado.
    return final_results, history

# =============================================================
# OPTIMIZER COMPLEXITY ANALYSIS CLASS
# =============================================================
class OptimizerComplexity:
    """
    Calculate complexity metrics for the final set of optimizers.
    """
    def __init__(self, optimizer_name, num_params):
        self.optimizer_name = optimizer_name
        self.num_params = num_params
        self.ops_per_param = self._calculate_ops_per_param()

    def _calculate_ops_per_param(self):
        if self.optimizer_name == "Adam":
            return 12  # Mantenido: 1er momento (3) + 2do momento (4) + actualización (5)
        elif self.optimizer_name == "Adamax":
            return 10  # Corregido: 1er momento (3) + 2do momento con max (3) + actualización (4)
        elif self.optimizer_name == "AMSGrad":
            return 13  # Corregido: Es Adam (12) + 1 operación 'max' para el mecanismo AMSGrad
        elif self.optimizer_name == "Epsilon":
            return 11
        else:
            return 0
            
    def get_oop(self): return self.ops_per_param
    def get_total_ops(self): return self.ops_per_param * self.num_params
    def get_energy(self, total_time_s):
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            power_mW = pynvml.nvmlDeviceGetPowerUsage(handle)
            pynvml.nvmlShutdown()
            return (power_mW / 1000.0) * total_time_s
        except pynvml.NVMLError:
            return 0.0


# =============================================================
# VISUALIZATION FUNCTIONS (Adapted from MNIST Notebook)
# =============================================================

def plot_curves(results, metric='loss'):
    plt.figure(figsize=(12, 6))
    for res in results:
        # --- MODIFICACIÓN AQUÍ ---
        # Verificar si existe la métrica de entrenamiento antes de graficar
        train_metric_key = f'train_{metric}'
        if train_metric_key in res['history']:
            plt.plot(res['history'][train_metric_key], label=f"{res['optimizer']} Train")
        
        # Verificar si existe la métrica de validación antes de graficar
        val_metric_key = f'val_{metric}'
        if val_metric_key in res['history']:
            # Si solo existe la de validación, la graficamos como línea sólida
            linestyle = '--' if train_metric_key in res['history'] else '-'
            label = f"{res['optimizer']} Val" if train_metric_key in res['history'] else res['optimizer']
            plt.plot(res['history'][val_metric_key], label=label, linestyle=linestyle)
        # --- FIN DE LA MODIFICACIÓN ---

    plt.title(f'Model {metric.capitalize()} Curves')
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confusion_matrices(results):
    print("\n" + "CONFUSION MATRICES" + "\n")
    num_optimizers = len(results)
    num_cols = 3
    num_rows = (num_optimizers + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    axes = axes.flatten() if num_optimizers > 1 else [axes]

    for i, res in enumerate(results):
        cm = confusion_matrix(res['y_true'], res['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], xticklabels=class_names, yticklabels=class_names)
        axes[i].set_title(f"CM for {res['optimizer']}")
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

# =============================================================
# HELPER AND MAIN EXECUTION SCRIPT (CORRECTED)
# =============================================================

# Define a utility function to count trainable parameters
def count_parameters(model):
    """Count the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Check if the script is run directly
if __name__ == '__main__':
    # Use the ResNet18 function that you've already defined
    model_para_contar = CIFAR10_CNN()
    
    total_parametros = count_parameters(model_para_contar)
    
    print(f"Model Architecture: {model_para_contar.__class__.__name__}")
    print(f"Total Trainable Parameters: {total_parametros:,}")
    
    optimizers_to_compare = [
        "Epsilon", "Adam", "Adamax", "AMSGrad"
    ]
    
    all_results = []
    histories = {} # Diccionario para guardar los historiales de entrenamiento

    # --- BUCLE DE ENTRENAMIENTO CORREGIDO ---
    for opt_name in optimizers_to_compare:
        # Creamos una nueva instancia del modelo para cada optimizador
        model_instance = CIFAR10_CNN().to(device)

        # Llamamos a la función de entrenamiento UNA SOLA VEZ
        result, history = train_and_evaluate_model(model_instance, opt_name, HYPERPARAMS)
        result['history'] = history  
        # Guardamos los resultados finales y el historial de cada época
        all_results.append(result)
        histories[opt_name] = history
    
    # --- A partir de aquí, tu código ya era correcto ---
    
    # Calcular métricas detalladas para la tabla final
    full_summary = []
    # Creamos un diccionario para acceder a los resultados por nombre de optimizador
    results_by_opt = {res['optimizer']: res for res in all_results}

    for opt_name in optimizers_to_compare:
        res = results_by_opt[opt_name]
        precision = precision_score(res['y_true'], res['y_pred'], average='macro', zero_division=0)
        recall = recall_score(res['y_true'], res['y_pred'], average='macro', zero_division=0)
        f1 = f1_score(res['y_true'], res['y_pred'], average='macro', zero_division=0)
        
        y_true_bin = label_binarize(res['y_true'], classes=range(NUM_CLASSES))
        auc_score = roc_auc_score(y_true_bin, res['y_scores'], average='macro')
        complexity = OptimizerComplexity(res['optimizer'], res['params'])
        energy_kJ = complexity.get_energy(res['training_time']) / 1000.0
        
        full_summary.append({
            "Optimizer": res['optimizer'], "Accuracy (%)": res['test_accuracy'] * 100,
            "Loss": res['test_loss'], "AUC": auc_score, "Time (s)": res['training_time'],
            "F1-Score": f1, "Recall": recall, "Precision": precision,
            "OOP": complexity.get_oop(), "Total Ops": complexity.get_total_ops(), "Energy (kJ)": energy_kJ
        })

    # Imprimir la tabla de resumen de resultados
    print(" " * 50 + "FINAL EXPERIMENT SUMMARY")
    header = full_summary[0].keys()
    print(f"{'Optimizer':<15} | {'Accuracy (%)':<12} | {'Loss':<10} | {'AUC':<10} | {'Time (s)':<10} | {'F1-Score':<10} | {'Recall':<10} | {'Precision':<10} | {'OOP':<5} | {'Total Ops':<12} | {'Energy (kJ)':<12}")
    print("-"*140)
    for item in sorted(full_summary, key=lambda x: x['Accuracy (%)'], reverse=True):
        print(f"{item['Optimizer']:<15} | {item['Accuracy (%)']:<12.2f} | {item['Loss']:<10.4f} | {item['AUC']:.4f}{'':<6} | {item['Time (s)']:<10.2f} | {item['F1-Score']:.4f}{'':<4} | {item['Recall']:.4f}{'':<4} | {item['Precision']:.4f}{'':<3} | {item['OOP']:<5} | {item['Total Ops']:<12.2e} | {item['Energy (kJ)']:<12.2f}")

# =============================================================
# GRÁFICAS DE COMPARACIÓN (CORREGIDO)
# =============================================================

# --- 1. CREA EL DICCIONARIO 'results' A PARTIR DE 'full_summary' ---
# Esto convierte tu lista de resultados en el formato que necesitan las gráficas de barras.
results = {item['Optimizer']: item for item in full_summary}

optimizers = ['Epsilon', 'Adam', 'Adamax', 'AMSGrad']

plt.figure(figsize=(20, 15))

# Gráfica 1: Pérdida (Funciona como estaba)
plt.subplot(2, 3, 1)
for opt_name in optimizers:
    plt.plot(histories[opt_name]['train_loss'], label=f'{opt_name} Train', linewidth=2)
    plt.plot(histories[opt_name]['val_loss'], '--', label=f'{opt_name} Val', linewidth=2)
plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfica 2: Precisión (Funciona como estaba)
plt.subplot(2, 3, 2)
for opt_name in optimizers:
    # Multiplicamos por 100 para mostrar como porcentaje
    train_acc_percent = [acc * 100 for acc in histories[opt_name]['train_acc']]
    val_acc_percent = [acc * 100 for acc in histories[opt_name]['val_acc']]
    plt.plot(train_acc_percent, label=f'{opt_name} Train', linewidth=2)
    plt.plot(val_acc_percent, '--', label=f'{opt_name} Val', linewidth=2)
plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfica 3: AUC (Funciona como estaba)
plt.subplot(2, 3, 3)
for opt_name in optimizers:
    plt.plot(histories[opt_name]['val_auc'], label=opt_name, linewidth=2)
plt.title('Validation AUC Macro by Epoch', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('AUC Score')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfica 4: Tiempo (Funciona como estaba)
plt.subplot(2, 3, 4)
for opt_name in optimizers:
    plt.plot(histories[opt_name]['epoch_times'], label=opt_name, linewidth=2)
plt.title('Time per Epoch', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Time (s)')
plt.legend()
plt.grid(True, alpha=0.3)

# --- 2. CÓDIGO CORREGIDO PARA GRÁFICAS 5 Y 6 ---

# Gráfica 5: Métricas finales comparativas
plt.subplot(2, 3, 5)
# Usamos los nombres de las claves que SÍ existen en tu diccionario 'results'
final_metrics = ['Accuracy (%)', 'Precision', 'Recall', 'F1-Score'] 
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x_pos = np.arange(len(optimizers))

for i, metric in enumerate(final_metrics):
    # Extraemos los valores del diccionario 'results' que creamos
    values = [results[opt_name][metric] for opt_name in optimizers]
    
    # El F1, Recall y Precision ya no necesitan multiplicarse por 100 si ya lo hiciste al crear la tabla
    # Asumimos que Accuracy ya está en % y los demás en escala 0-1
    if metric not in ['Accuracy (%)', 'Time (s)', 'Loss']:
         values = [v * 100 for v in values]

    bar = plt.bar(x_pos + i*0.2, values, width=0.2, label=metric_names[i], alpha=0.8)
    
    # Añadir valores en las barras
    for j, v in enumerate(values):
        plt.text(x_pos[j] + i*0.2, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=8)

plt.title('Final Metrics Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Optimizer')
plt.ylabel('Score (%)')
plt.xticks(x_pos + 0.3, optimizers)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 105) # Ajusta el límite para que los textos no se corten

# Gráfica 6: AUC Macro final comparativo
plt.subplot(2, 3, 6)
# Extraemos el valor de AUC, que ya está en escala 0-1, y lo multiplicamos por 100
auc_values = [results[opt_name]['AUC'] * 100 for opt_name in optimizers]
bars = plt.bar(optimizers, auc_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'], alpha=0.8)
plt.title('Final AUC Macro Comparison', fontsize=14, fontweight='bold')
plt.ylabel('AUC Score (%)')
for i, v in enumerate(auc_values):
    plt.text(i, v + 1, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
plt.ylim(0, 105) # Ajusta el límite para que los textos no se corten

plt.tight_layout()
plt.show()

# =============================================================
# VISUALIZATION FUNCTIONS (Adapted from MNIST Notebook)
# =============================================================

import numpy as np
import random
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt

def plot_curves(results, metric='loss'):
    plt.figure(figsize=(12, 6))
    for res in results:
        # Resolver alias (ej: auc_macro → auc)
        metric_key = metric
        if metric == 'auc_macro':
            metric_key = 'auc'

        train_metric_key = f'train_{metric_key}'
        val_metric_key   = f'val_{metric_key}'
        
        if train_metric_key in res['history']:
            plt.plot(res['history'][train_metric_key], label=f"{res['optimizer']} Train")
        
        if val_metric_key in res['history']:
            linestyle = '--' if train_metric_key in res['history'] else '-'
            label = f"{res['optimizer']} Val" if train_metric_key in res['history'] else res['optimizer']
            plt.plot(res['history'][val_metric_key], label=label, linestyle=linestyle)

    plt.title(f'Model {metric.upper()} Curves')
    plt.xlabel('Epoch')
    plt.ylabel(metric.upper())
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_confusion_matrices(results):
    print("\n" + " " * 28 + "CONFUSION MATRICES" + "\n")
    num_optimizers = len(results)
    num_cols = 3
    num_rows = (num_optimizers + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    axes = axes.flatten() if num_optimizers > 1 else [axes]

    for i, res in enumerate(results):
        cm = confusion_matrix(res['y_true'], res['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], 
                    xticklabels=class_names, yticklabels=class_names)
        axes[i].set_title(f"CM for {res['optimizer']}")
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()


def plot_random_subset_cm(results, num_classes_to_show=10):
    """
    Selecciona un subconjunto aleatorio de clases de cifar-100 y 
    genera sus matrices de confusión.
    """
    title = f"CONFUSION MATRICES ({num_classes_to_show} RANDOM CLASSES)"
    print(f"\n--- {title} ---\n")
    
    random_indices = sorted(random.sample(range(NUM_CLASSES), num_classes_to_show))
    random_names = [class_names[i] for i in random_indices]
    
    num_optimizers = len(results)
    num_cols = 2 
    num_rows = (num_optimizers + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(7 * num_cols, 6 * num_rows))
    axes = axes.flatten() if num_optimizers > 1 else [axes]

    for i, res in enumerate(results):
        y_true = np.array(res['y_true'])
        y_pred = np.array(res['y_pred'])
        
        indices_to_keep = np.isin(y_true, random_indices)
        y_true_filtered = y_true[indices_to_keep]
        y_pred_filtered = y_pred[indices_to_keep]
        
        cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=random_indices)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], 
                    xticklabels=random_names, yticklabels=random_names)
        axes[i].set_title(f"CM for {res['optimizer']} (Random Subset)")
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
        axes[i].tick_params(axis='x', rotation=90)
        axes[i].tick_params(axis='y', rotation=0)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.show()


# LLAMAR A LAS FUNCIONES PARA GENERAR GRÁFICAS

print("\n--- Generating Plots ---")

plot_curves(all_results, metric='loss')
plot_curves(all_results, metric='acc')
plot_curves(all_results, metric='auc_macro')  # ahora sí funcionará

plot_random_subset_cm(all_results, num_classes_to_show=10)



