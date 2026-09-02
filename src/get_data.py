import os
import glob
from monai.data import Dataset

def get_upenn_dicts(root_dir, split="train", pipeline=1):
    """
    Rastrea el dataset UPenn-GBM y devuelve una lista de diccionarios.
    Orden estricto de 11 canales: 
    DSC (rCBV, PH, PSR), DTI (AD, FA, RD, TR), Estructurales (FLAIR, T1, T1GD, T2).
    """
    dicts = []
    split_dir = os.path.join(root_dir, split)
    struc_dir = os.path.join(split_dir, "images", "images_structural")
    
    if not os.path.exists(struc_dir):
        return dicts

    cases = sorted(os.listdir(struc_dir))
    for case in cases:
        # El nombre del caso es algo como "UPENN-GBM-00055_11"
        
        # 1. Construir las 11 rutas en ORDEN ESTRICTO
        image_paths = [
            # DSC (0, 1, 2)
            os.path.join(split_dir, "images", "images_DSC", case, f"{case}_DSC_ap-rCBV.nii.gz"),
            os.path.join(split_dir, "images", "images_DSC", case, f"{case}_DSC_PH.nii.gz"),
            os.path.join(split_dir, "images", "images_DSC", case, f"{case}_DSC_PSR.nii.gz"),
            # DTI (3, 4, 5, 6)
            os.path.join(split_dir, "images", "images_DTI", case, f"{case}_DTI_AD.nii.gz"),
            os.path.join(split_dir, "images", "images_DTI", case, f"{case}_DTI_FA.nii.gz"),
            os.path.join(split_dir, "images", "images_DTI", case, f"{case}_DTI_RD.nii.gz"),
            os.path.join(split_dir, "images", "images_DTI", case, f"{case}_DTI_TR.nii.gz"),
            # Estructurales (7, 8, 9, 10)
            os.path.join(split_dir, "images", "images_structural", case, f"{case}_FLAIR.nii.gz"),
            os.path.join(split_dir, "images", "images_structural", case, f"{case}_T1.nii.gz"),
            os.path.join(split_dir, "images", "images_structural", case, f"{case}_T1GD.nii.gz"),
            os.path.join(split_dir, "images", "images_structural", case, f"{case}_T2.nii.gz"),
        ]
        
        # 2. Asignar el label correcto según el Pipeline
        if pipeline == 1:
            # Para P1 (Tumor Core y Edema Total)
            label_path = os.path.join(split_dir, "labels", f"{case}_automated_approx_segm.nii.gz")
            if not os.path.exists(label_path): # Respaldo por si se llama diferente
                label_path = os.path.join(split_dir, "labels", f"{case}_segm.nii.gz")
        else:
            # --- LÍNEA CORREGIDA PARA P2 ---
            # Ahora busca la máscara que nosotros generamos, que contiene toda la topología
            label_path = os.path.join(split_dir, "labels", f"{case}_tumorMask2_approx_segm.nii.gz")
            
        # 3. Validar existencia y añadir
        if all(os.path.exists(p) for p in image_paths) and os.path.exists(label_path):
            dicts.append({
                "image": image_paths, 
                "label": label_path, 
                "source": "UPENN"
            })
            
    return dicts


def get_muglioma_dicts(root_dir, split="train", pipeline=1):
    """
    Rastrea el dataset MU-Glioma Post y devuelve una lista de diccionarios.
    Solo 4 canales estructurales ordenados para coincidir con el final del tensor UPenn:
    FLAIR (t2f), T1 (t1n), T1GD (t1c), T2 (t2w).
    """
    dicts = []
    split_dir = os.path.join(root_dir, split)
    
    if not os.path.exists(split_dir):
        return dicts
        
    patients = glob.glob(os.path.join(split_dir, "PatientID_*"))
    for pat in patients:
        pat_id = os.path.basename(pat)
        timepoints = glob.glob(os.path.join(pat, "Timepoint_*"))
        
        for tp in timepoints:
            tp_id = os.path.basename(tp)
            
            # 1. Construir las 4 rutas en el ORDEN DE UPENN (índices 7, 8, 9, 10)
            image_paths = [
                os.path.join(tp, f"{pat_id}_{tp_id}_brain_t2f.nii.gz"), # FLAIR
                os.path.join(tp, f"{pat_id}_{tp_id}_brain_t1n.nii.gz"), # T1
                os.path.join(tp, f"{pat_id}_{tp_id}_brain_t1c.nii.gz"), # T1GD
                os.path.join(tp, f"{pat_id}_{tp_id}_brain_t2w.nii.gz"), # T2
            ]
            
            # 2. Asignar el label correcto según el Pipeline
            label_name = f"{pat_id}_{tp_id}_tumorMask_{pipeline}.nii.gz"
            label_path = os.path.join(tp, label_name)
            
            # 3. Validar existencia y añadir
            if all(os.path.exists(p) for p in image_paths) and os.path.exists(label_path):
                dicts.append({
                    "image": image_paths, 
                    "label": label_path, 
                    "source": "MUGLIOMA"
                })
                
    return dicts

def get_ucsd_dicts(root_dir, split="train", pipeline=1):
    """
    Rastrea el dataset UCSD-PTGBM y devuelve una lista de diccionarios.
    Carga los 11 canales densos y 2 máscaras de GT (BraTS + NECT).
    """
    dicts = []
    # La carpeta split podría ser "UCSD-PTGBM-BraTS-2024-test-set" o la de entrenamiento
    split_dir = os.path.join(root_dir, split) 
    
    if not os.path.exists(split_dir):
        return dicts

    cases = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    
    for case in cases:
        base_path = os.path.join(split_dir, case, f"{case}_")
        
        # 1. Los 11 canales mapeados al orden UPenn
        image_paths = [
            f"{base_path}CBV_LC.nii.gz",       # 0: DSC rCBV (Equivalente)
            f"{base_path}CBF_svd.nii.gz",      # 1: DSC PH (Sustituto)
            f"{base_path}MTT_svd.nii.gz",      # 2: DSC PSR (Sustituto)
            f"{base_path}RSI_Cell.nii.gz",     # 3: DTI AD (Sustituto de celularidad)
            f"{base_path}RSI_Free.nii.gz",     # 4: DTI FA (Sustituto de agua libre)
            f"{base_path}RSI_Hindered.nii.gz", # 5: DTI RD (Sustituto de obstaculización)
            f"{base_path}ADC_vendor.nii.gz",   # 6: DTI TR (Equivalente matemático)
            f"{base_path}FLAIR.nii.gz",        # 7: Estructural
            f"{base_path}T1pre.nii.gz",        # 8: Estructural
            f"{base_path}T1post.nii.gz",       # 9: Estructural
            f"{base_path}T2.nii.gz"            # 10: Estructural
        ]
        
        # 2. Cargamos las dos máscaras maestras
        label_brats = f"{base_path}BraTS_tumor_seg.nii.gz"
        label_nect = f"{base_path}non_enhancing_cellular_tumor_seg.nii.gz"
        
        # 3. Validar existencia y empaquetar
        if all(os.path.exists(p) for p in image_paths) and os.path.exists(label_brats) and os.path.exists(label_nect):
            dicts.append({
                "image": image_paths, 
                "label": [label_brats, label_nect], # MONAI lo cargará como shape (2, H, W, D)
                "source": "UCSD"
            })
            
    return dicts
class UnifiedDataset(Dataset):
    """
    Clase unificada que hereda de monai.data.Dataset.
    Puede cargar datos de UPenn, de MU-Glioma, o combinados, dependiendo de los 
    directorios proporcionados. 
    """
    def __init__(self, upenn_dir=None, muglioma_dir=None, section="train", pipeline=1, transform=None):
        data_dicts = []
        
        # Cargar UPenn si se provee la ruta
        if upenn_dir and os.path.exists(upenn_dir):
            upenn_dicts = get_upenn_dicts(upenn_dir, section, pipeline)
            data_dicts.extend(upenn_dicts)
            print(f"[{section.upper()}] Cargados {len(upenn_dicts)} casos de UPenn-GBM (Pipeline {pipeline})")
            
        # Cargar MU-Glioma si se provee la ruta
        if muglioma_dir and os.path.exists(muglioma_dir):
            muglioma_dicts = get_muglioma_dicts(muglioma_dir, section, pipeline)
            data_dicts.extend(muglioma_dicts)
            print(f"[{section.upper()}] Cargados {len(muglioma_dicts)} casos de MU-Glioma Post (Pipeline {pipeline})")
            
        if not data_dicts:
            print(f"⚠️ ADVERTENCIA: No se cargaron datos para la sección '{section}'. Verifica las rutas.")
            
        # Al pasar la lista de diccionarios al constructor padre, MONAI se encarga de todo el __getitem__
        super().__init__(data=data_dicts, transform=transform)
