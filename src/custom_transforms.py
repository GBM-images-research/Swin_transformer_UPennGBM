import torch
import numpy as np
from scipy import ndimage
from monai.transforms import MapTransform

#######################################################
# 1. TRANSFORMACIONES DE ESTANDARIZACIÓN (INPUTS)
#######################################################

class ImputeMissingChannelsd(MapTransform):
    """
    Rellena con ceros los canales funcionales si solo se reciben 4 estructurales.
    Asume que si entran 4 canales, corresponden a las estructurales (FLAIR, T1, T1GD, T2)
    y deben ocupar los índices 7, 8, 9 y 10 para coincidir con UPenn-GBM.
    """
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            # Si el tensor tiene 4 canales (MU-Glioma Post o Inferencia Clínica)
            if img.shape[0] == 4:
                # Crear tensor de 11 canales lleno de ceros
                if isinstance(img, torch.Tensor):
                    padded_img = torch.zeros((11, *img.shape[1:]), dtype=img.dtype, device=img.device)
                else:
                    padded_img = np.zeros((11, *img.shape[1:]), dtype=img.dtype)
                
                # Insertar los 4 canales estructurales al final (índices 7 al 10)
                padded_img[7:11, ...] = img
                d[key] = padded_img
                
            elif img.shape[0] != 11:
                raise ValueError(f"Se esperaban 4 o 11 canales. Se recibieron: {img.shape[0]}")
        return d


class RandModalityDropoutd(MapTransform):
    """
    Transformación de regularización: Con probabilidad 'prob', apaga (pone a 0) 
    los canales 0 al 6 (DSC y DTI). Esto fuerza al modelo a generalizar usando 
    solo las imágenes estructurales.
    """
    def __init__(self, keys, prob=0.5, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.prob = prob

    def __call__(self, data):
        d = dict(data)
        # Aplicar el dropout basado en la probabilidad
        if np.random.rand() < self.prob:
            for key in self.keys:
                img = d[key]
                # Clonar para evitar modificar el tensor original in-place
                img_dropped = img.clone() if isinstance(img, torch.Tensor) else img.copy()
                
                # Poner a cero los canales de perfusión y difusión (índices 0 al 6)
                img_dropped[0:7, ...] = 0.0
                d[key] = img_dropped
        return d


#######################################################
# 2. TRANSFORMACIONES DE ETIQUETAS UNIVERSALES (LABELS)
#######################################################
class ConvertToMultiChannelPipeline1d(MapTransform):
    """
    PIPELINE 1: Modelo Base (Sólidos Anidados)
    Salida: [Canal 0: Tumor Core Sólido, Canal 1: Whole Tumor Sólido]
    """
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            if img.ndim == 4 and img.shape[0] == 1: img = img.squeeze(0)

            # 1. El núcleo sólido
            tc = (img == 1) | (img == 4) | (img == 3)
            
            # 2. La masa total sólida (Núcleo + Edema)
            wt = tc | (img == 2)
            
            result = [tc, wt]
            d[key] = torch.stack(result, dim=0).float() if isinstance(img, torch.Tensor) else np.stack(result, axis=0).astype(np.float32)
        return d


class ConvertToMultiChannelPipeline2d(MapTransform):
    """
    Conversor para PIPELINE 2 (Sólidos Anidados Extendidos)
    Salida: [Canal 0: Target Extendido Sólido, Canal 1: Whole Abnormal Area Sólida]
    """
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            if img.ndim == 4 and img.shape[0] == 1: img = img.squeeze(0)

            # 1. Identificar componentes básicos
            # En MU-Glioma P2 el núcleo ahora es 4. En UPenn el núcleo es 1 o 4.
            core_base = (img == 4) | (img == 1) | (img == 3) 
            
            # La infiltración (1 en MU-Glioma, 6 en UPenn)
            infilt = (img == 1) | (img == 6)

            # 2. Construir Sólidos Jerárquicos
            # CANAL 0: Extended Target (Núcleo Original + Infiltración) -> MASA SÓLIDA
            extended_target = core_base | infilt
            
            # CANAL 1: Whole Abnormal Area (Extended Target + Edema Puro) -> MASA SÓLIDA GLOBAL
            whole_abnormal = extended_target | (img == 2)
            
            result = [extended_target, whole_abnormal]
            d[key] = torch.stack(result, dim=0).float() if isinstance(img, torch.Tensor) else np.stack(result, axis=0).astype(np.float32)
        return d

# class ConvertToMultiChannelPipeline1d(MapTransform):
#     """
#     Conversor Universal para el PIPELINE 1 (Tumor Core vs Edema Total).
#     Compatible con UPenn-GBM y MU-Glioma Post.
#     Salida: [Canal 0: Tumor Core, Canal 1: Edema Total]
#     """
#     def __init__(self, keys, allow_missing_keys=False):
#         super().__init__(keys, allow_missing_keys)

#     def __call__(self, data):
#         d = dict(data)
#         for key in self.keys:
#             img = d[key]
#             # Si tiene dimensión de canal extra (ej: [1, H, W, D]), comprimir
#             if img.ndim == 4 and img.shape[0] == 1:
#                 img = img.squeeze(0)

#             # --- Lógica Universal de Clases ---
#             # UPenn: Necrosis=1, Enhancing=4. MU-Glioma P1: Tumor Core=1, Enhancing=3.
#             tc = (img == 1) | (img == 4) | (img == 3)
#             # Edema en ambas bases de datos es siempre 2.
#             edema = (img == 2)
            
#             result = [tc, edema]
#             d[key] = torch.stack(result, dim=0).float() if isinstance(img, torch.Tensor) else np.stack(result, axis=0).astype(np.float32)
#         return d


# class ConvertToMultiChannelPipeline2d(MapTransform):
#     """
#     Conversor Universal para el PIPELINE 2 (Infiltración vs Edema Vasogénico Puro).
#     Compatible con UPenn-GBM y MU-Glioma Post.
#     Salida: [Canal 0: Infiltración, Canal 1: Edema Puro]
#     """
#     def __init__(self, keys, allow_missing_keys=False):
#         super().__init__(keys, allow_missing_keys)

#     def __call__(self, data):
#         d = dict(data)
#         for key in self.keys:
#             img = d[key]
#             if img.ndim == 4 and img.shape[0] == 1:
#                 img = img.squeeze(0)

#             # --- Lógica Universal de Clases ---
#             # UPenn: Infiltración=6. MU-Glioma P2: Infiltración=1.
#             infilt = (img == 1) | (img == 6)
#             # Edema Puro en ambas bases de datos es 2.
#             edema_puro = (img == 2)
            
#             result = [infilt, edema_puro]
#             d[key] = torch.stack(result, dim=0).float() if isinstance(img, torch.Tensor) else np.stack(result, axis=0).astype(np.float32)
#         return d


#######################################################
# 3. UTILIDADES ADICIONALES
#######################################################

class MaskedRegiond(MapTransform):
    """
    Aplica la máscara de etiquetas a la imagen de entrada, poniendo a cero 
    todo el fondo (background). Útil si quieres que el modelo solo atienda al cerebro/tumor.
    """
    def __init__(self, keys, label_key="label", allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.label_key = label_key

    def __call__(self, data_dict):
        d = dict(data_dict)
        # Crear máscara donde la etiqueta sea diferente de 0 (Fondo)
        mask = d[self.label_key] != 0
        
        for key in self.keys:
            img = d[key]
            # Expandir la máscara para cubrir todos los canales de la imagen (ej: 11 canales)
            mask_expanded = mask.expand(img.shape[0], -1, -1, -1)
            d[key] = img * mask_expanded
            
        return d
