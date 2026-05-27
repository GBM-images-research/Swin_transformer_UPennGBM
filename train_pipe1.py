import os
import time
import json
import shutil
import tempfile
import logging
from types import SimpleNamespace

import torch
import torch.nn.parallel
import numpy as np
import matplotlib.pyplot as plt
import wandb

from monai.data import DataLoader, decollate_batch
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.transforms import AsDiscrete, Activations
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR
from functools import partial

# --- IMPORTACIONES DEL NUEVO ENTORNO UNIFICADO ---
from src.get_data import UnifiedDataset
from src.custom_transforms import (
    ImputeMissingChannelsd,
    RandModalityDropoutd,
    ConvertToMultiChannelPipeline1d
)

logging.basicConfig(level=logging.INFO)

#################################
# CONFIGURACIÓN DE HIPERPARÁMETROS
#################################
roi = (128, 128, 64)  
batch_size = 1
sw_batch_size = 2
infer_overlap = 0.5
max_epochs = 100
val_every = 1
lr = 1e-4
weight_decay = 1e-5
feature_size = 48
use_v2 = False

config_train = SimpleNamespace(
    roi=roi,
    batch_size=batch_size,
    sw_batch_size=sw_batch_size,
    infer_overlap=infer_overlap,
    max_epochs=max_epochs,
    val_every=val_every,
    lr=lr,
    weight_decay=weight_decay,
    feature_size=feature_size,
    pipeline="Pipeline 1: Tumor Core + Whole Tumor",
    network="SwinUNETR",
    use_v2=use_v2,
)

#############################
# INICIALIZACIÓN DE WANDB
#############################
logging.info("Logging in WandB")
api_key = os.environ.get("WANDB_API_KEY")
if api_key:
    wandb.login(key=api_key)

run = wandb.init(
    project="Swin_Unified_Pipeline1", 
    job_type="train", 
    config=config_train
)
config_train = wandb.config

directory = "./Dataset_Output"
os.makedirs(directory, exist_ok=True)
print(f"Output directory: {directory}")

#############################
# CLASES Y UTILIDADES
#############################
class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)

def save_checkpoint(model, epoch, filename="model.pt", best_acc=0, dir_add=directory):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    filepath = os.path.join(dir_add, filename)
    torch.save(save_dict, filepath)
    print("Saving checkpoint:", filepath)

###############################
# PIPELINE DE TRANSFORMACIONES
###############################
train_transform = transforms.Compose([
    transforms.LoadImaged(keys=["image", "label"]),
    transforms.EnsureChannelFirstd(keys=["image", "label"]),
    
    # Unificación y Regularización
    ImputeMissingChannelsd(keys=["image"]),
    RandModalityDropoutd(keys=["image"], prob=0.5),
    
    # Formateo de Etiquetas a 2 Canales (TC, WT/Whole Tumor)
    ConvertToMultiChannelPipeline1d(keys=["label"]),
    
    # --- Recorte y Aumentación Espacial OPTIMIZADOS ---
    transforms.CropForegroundd(
        keys=["image", "label"], 
        source_key="image",
        margin=5,
        k_divisible=32,
        allow_smaller=False  
    ),
    
    transforms.RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=roi,
        pos=1,          
        neg=1,          
        num_samples=1,  
        image_key="image",
        image_threshold=0,
    ),
    # --------------------------------------------------

    # Aumentaciones geométricas y de intensidad
    transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    
    transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),            
    
    # Probabilidad bajada al 30%, factor bajado al 5%
    transforms.RandScaleIntensityd(keys="image", factors=0.05, prob=0.3),
    transforms.RandShiftIntensityd(keys="image", offsets=0.05, prob=0.3),
])

val_transform = transforms.Compose([
    transforms.LoadImaged(keys=["image", "label"]),
    transforms.EnsureChannelFirstd(keys=["image", "label"]),
    
    ImputeMissingChannelsd(keys=["image"]),
    ConvertToMultiChannelPipeline1d(keys=["label"]),
    
    transforms.CropForegroundd(
        keys=["image", "label"], 
        source_key="image",
        margin=5,
        k_divisible=32,
        allow_smaller=False
    ),
        
    transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
])

#############################
# CREACIÓN DEL MODELO
#############################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SwinUNETR(
    img_size=roi,
    in_channels=11,  
    out_channels=2,  
    feature_size=feature_size,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=0.0,
    use_checkpoint=True,
    use_v2=use_v2,
)

# model_path = "trained_models/vtzpbajf_best_model_pipe1/model.pt"
# if os.path.exists(model_path):
#     loaded_model = torch.load(model_path, map_location=device)["state_dict"]
#     model.load_state_dict(loaded_model)
#     print(f"Modelo preentrenado cargado desde {model_path}")

model.to(device)

###########################
# OPTIMIZADOR Y PÉRDIDA
###########################
torch.backends.cudnn.benchmark = True
dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)

post_sigmoid = Activations(sigmoid=True)
post_pred = AsDiscrete(argmax=False, threshold=0.5)

dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)

model_inferer = partial(
    sliding_window_inference,
    roi_size=roi,
    sw_batch_size=sw_batch_size,
    predictor=model,
    overlap=infer_overlap,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

###########################
# FUNCIONES DE ENTRENAMIENTO
###########################
def train_epoch(model, loader, optimizer, epoch, loss_func):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        data, target = batch_data["image"].to(device), batch_data["label"].to(device)
        logits = model(data)
        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() 
        
        # Corrección: usar data.shape[0] es más robusto que batch_size global
        run_loss.update(loss.item(), n=data.shape[0])
        print("Epoch {}/{} {}/{} loss: {:.4f} time {:.2f}s".format(
            epoch, max_epochs, idx, len(loader), run_loss.avg, time.time() - start_time))
        start_time = time.time()
    return run_loss.avg

def val_epoch(model, loader, epoch, acc_func, model_inferer, post_sigmoid, post_pred):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"].to(device), batch_data["label"].to(device)
            logits = model_inferer(data)
            
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
            
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            
            dice_tc = run_acc.avg[0]
            dice_wt = run_acc.avg[1]
            print("Val {}/{} {}/{} , dice_tc: {:.4f} , dice_wt: {:.4f} , time {:.2f}s".format(
                epoch, max_epochs, idx, len(loader), dice_tc, dice_wt, time.time() - start_time))
            start_time = time.time()
    return run_acc.avg

def trainer(model, train_loader, val_loader, optimizer, loss_func, acc_func, scheduler, model_inferer, start_epoch, post_sigmoid, post_pred):
    val_acc_max = 0.0
    for epoch in range(start_epoch, max_epochs):
        print(time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, epoch, loss_func)
        print("Final training {}/{} loss: {:.4f} time {:.2f}s".format(
            epoch, max_epochs - 1, train_loss, time.time() - epoch_time))
        
        wandb.log({
            "loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch": epoch,
        })

        if (epoch + 1) % val_every == 0 or epoch == 0:
            epoch_time = time.time()
            val_acc = val_epoch(model, val_loader, epoch, acc_func, model_inferer, post_sigmoid, post_pred)
            dice_tc = val_acc[0]
            dice_wt = val_acc[1]
            val_avg_acc = np.mean(val_acc)
            
            print("Final validation stats {}/{} , dice_tc: {:.4f} , dice_wt: {:.4f} , Dice_Avg: {:.4f} , time {:.2f}s".format(
                epoch, max_epochs - 1, dice_tc, dice_wt, val_avg_acc, time.time() - epoch_time))
            
            # Corrección: Cambiado 'val_dice_edema' a 'val_dice_wt' para evitar confusiones topológicas
            wandb.log({
                "val_dice_tc": dice_tc,
                "val_dice_wt": dice_wt,
                "val_dice_avg": val_avg_acc,
            })
            
            save_checkpoint(model, epoch, filename="model_last.pt", best_acc=val_avg_acc)
            
            if val_avg_acc > val_acc_max:
                print("New best ({:.6f} --> {:.6f}).".format(val_acc_max, val_avg_acc))
                val_acc_max = val_avg_acc
                save_checkpoint(model, epoch, filename="model_best.pt", best_acc=val_acc_max)

        scheduler.step()
        
    print("Training Finished! Best Accuracy: ", val_acc_max)
    
    if wandb.run is not None:
        artifact_name = f"{wandb.run.id}_best_model"
        at = wandb.Artifact(artifact_name, type="model")
        at.add_file(os.path.join(directory, "model_best.pt"))
        wandb.log_artifact(at, aliases=["final"])

    return val_acc_max

####################################
# Carga de Datos y Ejecución
####################################
def main(config_train):
    UPENN_DIR = "./Dataset/Dataset_331_30_casos/"
    MUGLIOMA_DIR = "./Dataset/MU_glioma/"
    PIPELINE = 1 

    print(f"\n--- INICIANDO CARGA PARA PIPELINE {PIPELINE} ---")
    train_set = UnifiedDataset(
        upenn_dir=UPENN_DIR, 
        muglioma_dir=MUGLIOMA_DIR, 
        section="train", 
        pipeline=PIPELINE,
        transform=train_transform
    )
    
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    val_set = UnifiedDataset(
        upenn_dir=UPENN_DIR, 
        muglioma_dir=MUGLIOMA_DIR, 
        section="val", 
        pipeline=PIPELINE,
        transform=val_transform
    )
    
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )

    print("\n--- INICIANDO ENTRENAMIENTO ---")
    start_epoch = 0
    trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        scheduler=scheduler,
        model_inferer=model_inferer,
        start_epoch=start_epoch,
        post_sigmoid=post_sigmoid,
        post_pred=post_pred,
    )
    print("Script completado.")

if __name__ == "__main__":
    main(config_train)