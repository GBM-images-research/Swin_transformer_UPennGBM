import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from monai.inferers import sliding_window_inference
from monai import transforms
from monai.data import DataLoader
from monai.networks.nets import SwinUNETR

# --- IMPORTACIONES DEL ENTORNO UNIFICADO ---
from src.get_data import UnifiedDataset
from src.custom_transforms import (
    ImputeMissingChannelsd,
    ConvertToMultiChannelPipeline2_Experimento_d
)

# ==========================================
# CONFIGURACIÓN DE RUTAS Y DISPOSITIVO
# ==========================================
UPENN_DIR = "./Dataset/Dataset_30_6/"
MUGLIOMA_DIR = "./Dataset/MU_glioma/"
PIPELINE = 2
MODEL_WEIGHTS_PATH = "./Dataset_Output/pipe2/expert-jazz-7/model_best.pt"
roi = (128, 128, 64)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Calculando Temperatura Óptima en: {device}")

# ==========================================
# TRANSFORMACIONES Y DATALOADER (Validación)
# ==========================================
val_transform = transforms.Compose([
    transforms.LoadImaged(keys=["image", "label"]),
    transforms.EnsureChannelFirstd(keys=["image", "label"]),
    ImputeMissingChannelsd(keys=["image"]),
    ConvertToMultiChannelPipeline2_Experimento_d(keys=["label"]),
    transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
])

# Cargamos TODO el set de validación (es la forma matemáticamente correcta de calibrar)
val_set = UnifiedDataset(
    upenn_dir=UPENN_DIR, 
    muglioma_dir=MUGLIOMA_DIR, 
    section="val", 
    pipeline=PIPELINE, 
    transform=val_transform
)
# Batch size 1 por la naturaleza del tamaño variable del volumen
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)

# ==========================================
# CARGA DEL MODELO (PIPELINE 2)
# ==========================================
model = SwinUNETR(img_size=roi, in_channels=11, out_channels=2, feature_size=48, use_checkpoint=True).to(device)
if os.path.exists(MODEL_WEIGHTS_PATH):
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device)["state_dict"])
    print(f"Pesos de P2 cargados desde {MODEL_WEIGHTS_PATH}")
else:
    raise FileNotFoundError(f"No se encontró el modelo en {MODEL_WEIGHTS_PATH}")

model.eval()

# ==========================================
# MÓDULO DE TEMPERATURE SCALING
# ==========================================
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        # PyTorch optimizará este único escalar. Empezamos en 1.5.
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        # PyTorch divide automáticamente toda la matriz [2, N] por el escalar
        return logits / self.temperature

# ==========================================
# 1. FASE DE EXTRACCIÓN DE LOGITS CRUDOS
# ==========================================
print("\n--- PASO 1: Extrayendo Logits del Set de Validación ---")
all_logits = []
all_labels = []

# Muestreo Sub-espacial: 
# Guardar el cerebro completo de validación satura la RAM. Guardaremos 1 de cada 10 vóxeles.
SUB_SAMPLE_RATE = 10 

with torch.no_grad():
    for batch_data in tqdm(val_loader):
        image = batch_data["image"].to(device)
        label = batch_data["label"].to(device) # [1, 2, H, W, D]
        
        # Inferencia con Autocast para velocidad y memoria
        with torch.cuda.amp.autocast():
            logits = sliding_window_inference(image, roi, sw_batch_size=1, predictor=model, overlap=0.5)
            
        # Aplanar tensores para no lidiar con formas 3D durante la optimización
        logits_flat = logits.view(2, -1) # [2, H*W*D]
        label_flat = label.squeeze(0).view(2, -1) # [2, H*W*D]
        
        # Muestreo (Nos quedamos con 1 de cada 10 vóxeles para no colapsar la RAM del optimizador)
        indices_muestreo = torch.randperm(logits_flat.shape[1])[:logits_flat.shape[1] // SUB_SAMPLE_RATE]
        
        logits_muestreados = logits_flat[:, indices_muestreo]
        label_muestreada = label_flat[:, indices_muestreo]
        
        all_logits.append(logits_muestreados.cpu())
        all_labels.append(label_muestreada.float().cpu()) # NLL/BCE requiere floats
        
        # Limpieza
        del image, label, logits, logits_flat, label_flat
        torch.cuda.empty_cache()

# Consolidamos todos los pacientes en un mega-tensor [2, N_voxeles_totales]
final_logits = torch.cat(all_logits, dim=1).to(device)
final_labels = torch.cat(all_labels, dim=1).to(device)

print(f"\nTotal de vóxeles de validación extraídos para calibración: {final_logits.shape[1]:,}")

# ==========================================
# 2. FASE DE OPTIMIZACIÓN (L-BFGS)
# ==========================================
print("\n--- PASO 2: Optimizando la Temperatura (L-BFGS) ---")

scaler = TemperatureScaler().to(device)
criterion = nn.BCEWithLogitsLoss()

# Calculamos NLL Inicial (T=1.0)
with torch.no_grad():
    # Nos aseguramos de quitar la metadata de MONAI con .as_tensor() por seguridad
    nll_inicial = criterion(final_logits.as_tensor(), final_labels.as_tensor()).item()
print(f"NLL Inicial (T=1.0, Sin calibrar): {nll_inicial:.6f}")

optimizer = optim.LBFGS([scaler.temperature], lr=0.01, max_iter=100)

def eval_closure():
    optimizer.zero_grad()
    # Pasamos los logits aplanados directamente, sin unsqueeze
    scaled_logits = scaler(final_logits.as_tensor()) 
    loss = criterion(scaled_logits, final_labels.as_tensor())
    loss.backward()
    return loss

# Ejecutar optimización
optimizer.step(eval_closure)

# Resultados Finales
optimal_T = scaler.temperature.item()
with torch.no_grad():
    nll_final = criterion(scaler(final_logits.as_tensor()), final_labels.as_tensor()).item()

print("\n" + "="*50)
print(" RESULTADOS DE CALIBRACIÓN")
print("="*50)
print(f"Temperatura Óptima (T): {optimal_T:.4f}")
print(f"NLL Final (Optimizado): {nll_final:.6f}")
print(f"Mejora en Error: {((nll_inicial - nll_final) / nll_inicial) * 100:.2f}%")
print("="*50)