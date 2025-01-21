# Loading functions
import os
import time
from monai.data import DataLoader, decollate_batch

import torch
import torch.nn.parallel

from src.get_data import CustomDataset
import numpy as np
from scipy import ndimage
from types import SimpleNamespace
import wandb
import logging

#####
import json
import shutil
import tempfile

import matplotlib.pyplot as plt
import nibabel as nib

from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.transforms import (
    AsDiscrete,
    Activations,
    MapTransform,
)
from monai.utils.enums import TransformBackends
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR


# from monai.data import decollate_batch
from functools import partial

# from src.custom_transforms import ConvertToMultiChannelBasedOnN_Froi, ConvertToMultiChannelBasedOnBratsClassesdI
from src.custom_transforms import ConvertToMultiChannelBasedOnAnotatedInfiltration

####

logging.basicConfig(level=logging.INFO)


#################################
# HIPER PARAMETER CONFIGURATION #
#################################

### Hyperparameter
roi = (128, 128, 128) # (128, 128, 128) - (96, 96, 96)
batch_size = 1
sw_batch_size = 2
fold = 1
infer_overlap = 0.5
max_epochs = 50
val_every = 1
lr = 1e-4  # default 1e-4
weight_decay = 1e-8  # default 1e-5 (proporcional a la regularización que se aplica)
feature_size = 72 # default 48 - 72 - 96
use_v2=False
source_k = "image" # label - image
dataset_k=("train_all", "train_all") # ("train_00", "valid_00")

print("Train dataset:", dataset_k[0])
print("Val dataset:", dataset_k[1])

# train_loader, val_loader = get_loader(batch_size, data_dir, json_list, fold, roi)

config_train = SimpleNamespace(
    roi=roi,
    batch_size=batch_size,
    sw_batch_size=sw_batch_size,
    fold=fold,
    infer_overlap=infer_overlap,
    max_epochs=max_epochs,
    val_every=val_every,
    lr=lr,
    weight_decay=weight_decay,
    feature_size=feature_size,
    GT="N-ROI + F-ROI",  # modifica para eliminar edema "Edema + Infiltration"
    patch_with= source_k, # label - image
    network="original",
    use_v2=use_v2,
    dataset=dataset_k
)

#############################
### Inicializar WandB
#############################
# Cargar la clave API desde una variable de entorno
logging.info("Logging in WandB")
api_key = os.environ.get("WANDB_API_KEY")
# Iniciar sesión en W&B
wandb.login(key=api_key)

# create a wandb run
run = wandb.init(project="Swin_UPENN_10cases", job_type="train", config=config_train) # Swin_UPENN_106cases - Swin_UPENN_29_casos_pruebas

# we pass the config back from W&B
config_train = wandb.config


directory = "./Dataset"
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)


### Setup average meter, fold reader, checkpoint saver
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


def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val


def save_checkpoint(model, epoch, filename="model.pt", best_acc=0, dir_add=root_dir):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    filename = os.path.join(dir_add, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


###############################
#### Compose functions
###############################
train_transform = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "label"]),
        # ConvertToMultiChannelBasedOnN_Froi(keys="label"),
        ConvertToMultiChannelBasedOnAnotatedInfiltration(keys="label"),
        transforms.CropForegroundd(
            keys=["image", "label"],
            source_key=source_k,
            k_divisible=[roi[0], roi[1], roi[2]],
        ),
        transforms.RandSpatialCropd(
            keys=["image", "label"],
            roi_size=[roi[0], roi[1], roi[2]],
            random_size=False,
        ),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        
    ]
)
val_transform = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "label"]),
        # ConvertToMultiChannelBasedOnN_Froi(keys="label"),
        ConvertToMultiChannelBasedOnAnotatedInfiltration(keys="label"),
        transforms.RandSpatialCropd(
            keys=["image", "label"],
            roi_size=[-1, -1, -1], #[240, 240, 155],
            random_size=False,
        ),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
)

# Create Swin transformer
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SwinUNETR(
    img_size=roi,
    in_channels=11,  # 10 / 11
    out_channels=2,  # modificar con edema
    feature_size=feature_size,  # default 48
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=0.0,
    use_checkpoint=True, # default True para horrar memoria
    use_v2=use_v2,
)#.to(device)

##############################
### Traer modelo desde WandB #
##############################
# run = wandb.init()
# artifact = run.use_artifact(
#     "mlops-team89/Swin_UPENN_106cases/mjkearkn_best_model:v0", type="model"
# )
# artifact_dir = artifact.download()
# print(artifact_dir)
# model_path = os.path.join(artifact_dir, "model.pt")
# model_path = "./artifacts/mjkearkn_best_model:v0"
# model.load_state_dict(torch.load(os.path.join(model_path, "model.pt"))["state_dict"])


# mlops-team89/Swin_UPENN/93rp3g83_best_model:v0  -> cerebro_nroi+froi
# mlops-team89/Swin_UPENN/sq1r37ci_best_model:v0  -> cerebro_nroi+froi+edema

# run = wandb.init()
# artifact = run.use_artifact(
#     "mlops-team89/Swin_UPENN_106cases/8fhm3ha5_best_model:v0", type="model"
# )
# artifact_dir = artifact.download()
# print(artifact_dir)
# model_path = os.path.join(artifact_dir, "model.pt")
# model_path = os.path.join("./trained_models", "model.pt")

############################
# Load the model localmente
#############################
model_path = "artifacts/7y5x1mkj_best_model:v0/model.pt" #'Dataset/model_dataset_330_30_96x96x96_48f_v02.pt' # 5mm - mjkearkn_best_model-v0 / 10mm - ip0bojmx_best_model-v0

# Load the model on CPU
loaded_model = torch.load(model_path, map_location=torch.device(device))["state_dict"]
# model.load_state_dict(torch.load(model_path)["state_dict"])

# Load the state dictionary into the model
model.load_state_dict(loaded_model)

# Move the model to the desired device (e.g., GPU) if available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

###########################
# Optimiser function loss #
###########################
torch.backends.cudnn.benchmark = True
dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)
post_sigmoid = Activations(sigmoid=True)
post_pred = AsDiscrete(argmax=False, threshold=0.5)
dice_acc = DiceMetric(
    include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True
)
model_inferer = partial(
    sliding_window_inference,
    roi_size=[roi[0], roi[1], roi[2]],
    sw_batch_size=sw_batch_size,
    predictor=model,
    overlap=infer_overlap,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)


# Define Train and Validation Epoch
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
        run_loss.update(loss.item(), n=batch_size)
        print(
            "Epoch {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
            "loss: {:.4f}".format(run_loss.avg),
            "time {:.2f}s".format(time.time() - start_time),
        )
        start_time = time.time()
    return run_loss.avg


def val_epoch(
    model,
    loader,
    epoch,
    acc_func,
    model_inferer=None,
    post_sigmoid=None,
    post_pred=None,
):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"].to(device), batch_data["label"].to(
                device
            )
            logits = model_inferer(data)
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [
                post_pred(post_sigmoid(val_pred_tensor))
                for val_pred_tensor in val_outputs_list
            ]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            dice_tc = run_acc.avg[0]
            dice_wt = run_acc.avg[1]
            # dice_et = run_acc.avg[2] # comentar sin edema
            print(
                "Val {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
                ", dice_tc:",
                dice_tc,
                ", dice_wt:",
                dice_wt,
                # ", dice_et:", # comentar sin edema
                # dice_et,
                ", time {:.2f}s".format(time.time() - start_time),
            )
            # # wandb
            # wandb.log(
            #     {
            #         "val_dice_nroi": dice_tc,
            #         "val_dice_froi": dice_wt,
            #         "val_dice_edema": dice_et,
            #     }
            # )
            start_time = time.time()

    return run_acc.avg


# Define training
def trainer(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    scheduler,
    model_inferer=None,
    start_epoch=0,
    post_sigmoid=None,
    post_pred=None,
):
    val_acc_max = 0.0
    dices_tc = []
    dices_wt = []
    dices_et = []
    dices_avg = []
    loss_epochs = []
    trains_epoch = []
    for epoch in range(start_epoch, max_epochs):
        print(time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            epoch=epoch,
            loss_func=loss_func,
        )
        print(
            "Final training  {}/{}".format(epoch, max_epochs - 1),
            "loss: {:.4f}".format(train_loss),
            "time {:.2f}s".format(time.time() - epoch_time),
        )
        # wandb
        print("train_loss", train_loss.item(), type(train_loss.item()))
        print(
            "lr", optimizer.param_groups[0]["lr"], type(optimizer.param_groups[0]["lr"])
        )

        wandb.log(
            {
                "loss": train_loss.item(),
                "lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            }
        )

        if (epoch + 1) % val_every == 0 or epoch == 0:
            loss_epochs.append(train_loss)
            trains_epoch.append(int(epoch))
            epoch_time = time.time()
            val_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
            )
            dice_tc = val_acc[0]
            dice_wt = val_acc[1]
            # dice_et = val_acc[2] # comentar sin edema
            val_avg_acc = np.mean(val_acc)
            print(
                "Final validation stats {}/{}".format(epoch, max_epochs - 1),
                ", dice_tc:",
                dice_tc,
                ", dice_wt:",
                dice_wt,
                # ", dice_et:", # comentar sin edema
                # dice_et,
                ", Dice_Avg:",
                val_avg_acc,
                ", time {:.2f}s".format(time.time() - epoch_time),
            )
            # wandb
            wandb.log(
                {
                    "val_dice_nroi": dice_tc,
                    "val_dice_froi": dice_wt,
                    # "val_dice_edema": dice_et, # comentar sin edema
                    "val_dice_avg": val_avg_acc,
                }
            )
            start_time = time.time()
            dices_tc.append(dice_tc)
            dices_wt.append(dice_wt)
            # dices_et.append(dice_et) # comentar sin edema
            dices_avg.append(val_avg_acc)
            # Guardar last model
            save_checkpoint(
                    model,
                    epoch,
                    filename="model_last.pt",
                    best_acc=val_avg_acc,
                )
            # Guardar best model
            if val_avg_acc > val_acc_max:
                print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                val_acc_max = val_avg_acc
                save_checkpoint(
                    model,
                    epoch,
                    best_acc=val_acc_max,
                )

            scheduler.step()
    print("Training Finished !, Best Accuracy: ", val_acc_max)
    # Guardar artefacto en W&B
    artifact_name = f"{wandb.run.id}_best_model"
    at = wandb.Artifact(artifact_name, type="model")
    at.add_file(os.path.join(directory, "model.pt"))
    wandb.log_artifact(at, aliases=["final"])

    return (
        val_acc_max,
        dices_tc,
        dices_wt,
        dices_et,
        dices_avg,
        loss_epochs,
        trains_epoch,
    )


####################################
# Load DATASET and training modelo #
####################################
def main(config_train):
    dataset_path = "./Dataset/Dataset_10_1_casos/"

    train_set = CustomDataset(
        dataset_path, section=dataset_k[0], transform=train_transform
    )  # t_transform
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    im_t = train_set[0]
    # (im_t["image"].shape)
    print(im_t["image"].shape)

    val_set = CustomDataset(
        dataset_path, section=dataset_k[1], transform=val_transform
    )  # v_transform
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    im_v = val_set[0]
    # (im_t["image"].shape)
    print(im_v["label"].shape)

    ##########################################################
    # Comenzar entrenamiento
    ##########################################################
    start_epoch = 0

    (
        val_acc_max,
        dices_tc,
        dices_wt,
        dices_et,
        dices_avg,
        loss_epochs,
        trains_epoch,
    ) = trainer(
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
    print(f"train completed, best average dice: {val_acc_max:.4f} ")

    # plt.figure("train", (12, 6))
    # plt.subplot(1, 2, 1)
    # plt.title("Epoch Average Loss")
    # plt.xlabel("epoch")
    # plt.plot(trains_epoch, loss_epochs, color="red")
    # plt.subplot(1, 2, 2)
    # plt.title("Val Mean Dice")
    # plt.xlabel("epoch")
    # plt.plot(trains_epoch, dices_avg, color="green")
    # plt.show()
    # plt.figure("train", (18, 6))
    # plt.subplot(1, 3, 1)
    # plt.title("Val Mean Dice TC")
    # plt.xlabel("epoch")
    # plt.plot(trains_epoch, dices_tc, color="blue")
    # plt.subplot(1, 3, 2)
    # plt.title("Val Mean Dice WT")
    # plt.xlabel("epoch")
    # plt.plot(trains_epoch, dices_wt, color="brown")
    # # plt.subplot(1, 3, 3)
    # # plt.title("Val Mean Dice ET")
    # # plt.xlabel("epoch")
    # # plt.plot(trains_epoch, dices_et, color="purple")
    # plt.show()

    # finish W&B run


if directory is None:
    shutil.rmtree(root_dir)

if __name__ == "__main__":
    main(config_train)
