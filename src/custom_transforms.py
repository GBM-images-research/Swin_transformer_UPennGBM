# Loading functions
import torch
import torch.nn.parallel

import numpy as np
from scipy import ndimage

from monai.transforms import (
    MapTransform,
    Transform,
)
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor


def fill_holes_3d(mask):
    # Rellenar huecos en la máscara 3D
    filled_mask = ndimage.binary_fill_holes(mask)
    return filled_mask


def expand_mask_3d_td(
    mask, edema, distance_cm_max=0.5, distance_cm_min=0.1, voxel_size=0.1
):
    distance_pixels_max = int(distance_cm_max / voxel_size)
    distance_pixel_min = int(distance_cm_min / voxel_size)

    # Calcular la transformada de distancia
    distance_transform = ndimage.distance_transform_edt(np.logical_not(mask))

    # Crear la nueva máscara alrededor del tumor core
    # expanded_mask_distance = distance_transform >= distance_threshold
    expanded_mask = np.logical_and(
        distance_transform >= distance_pixel_min,
        distance_transform <= distance_pixels_max,
    )

    # Restar la máscara original para obtener solo la región expandida
    exterior_mask = np.logical_and(expanded_mask, np.logical_not(mask))
    # Hacer un AND con el edema para eliminar zonas externas a este
    exterior_mask = np.logical_and(exterior_mask, edema)

    return exterior_mask  # torch.from_numpy(exterior_mask)


######################
# BraTS original
################################
class ConvertToMultiChannelBasedOnBratsClassesI(Transform):
    """
    Convert labels to multi channels based on `brats18 <https://www.med.upenn.edu/sbia/brats2018/data.html>`_ classes,
    which include TC (Tumor core), WT (Whole tumor) and ET (Enhancing tumor):
    label 1 is the necrotic and non-enhancing tumor core, which should be counted under TC and WT subregion,
    label 2 is the peritumoral edema, which is counted only under WT subregion,
    label 4 is the GD-enhancing tumor, which should be counted under ET, TC, WT subregions.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        # Usar con etiquetado BraTS - automated_approx_segm
        result = [
            (img == 1) | (img == 4),
            # (img == 1) | (img == 4) | (img == 2),
            img == 2,
        ]
        
        # # Usar con etiquetado TC+Inf - Edema combined3_approx_segm
        # result = [
        #     img == 6,
        #     # (img == 1) | (img == 4) | (img == 2),
        #     img == 2,
        # ]
        
        # merge labels 1 (tumor non-enh) and 4 (tumor enh) and 2 (large edema) to WT
        # label 4 is ET
        return (
            torch.stack(result, dim=0)
            if isinstance(img, torch.Tensor)
            else np.stack(result, axis=0)
        )


class ConvertToMultiChannelBasedOnBratsClassesdI(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertToMultiChannelBasedOnBratsClasses`.
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = ConvertToMultiChannelBasedOnBratsClassesI.backend

    def __init__(self, keys, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvertToMultiChannelBasedOnBratsClassesI()

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


########################
# NROI + FROI
###########################
class ConvertToMultiChannelBasedOnBratsClassesII(Transform):
    """
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img):
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        # Máscaras básicas
        necro = img == 1  # Necrosis
        edema = img == 2  # Edema
        active = (img == 3) | (img == 4)  # Activo (3 y 4 combinados)

        # Tumor core (necro + activo)
        tumor_core = torch.logical_or(necro, active)

        # Rellenar huecos en la máscara
        filled_tumor_core = fill_holes_3d(tumor_core)

        # Configuración del tamaño de voxel (en cm)
        voxel_size_cm = 0.1

        # Generar ROI cercana (Nroi) y ROI lejana (Froi)
        N_roi = expand_mask_3d_td(
            filled_tumor_core,
            edema=edema,
            distance_cm_max=0.5,
            distance_cm_min=0.1,
            voxel_size=voxel_size_cm,
        )

        F_roi = expand_mask_3d_td(
            filled_tumor_core,
            edema=edema,
            distance_cm_max=10,
            distance_cm_min=1,
            voxel_size=voxel_size_cm,
        )
        result = [
            torch.from_numpy(N_roi),
            torch.from_numpy(F_roi),
        ]
        return (
            torch.stack(result, dim=0)
            if isinstance(img, torch.Tensor)
            else np.stack(result, axis=0)
        )  #


class ConvertToMultiChannelBasedOnN_Froi(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertToMultiChannelBasedOnBratsClasses`.
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = ConvertToMultiChannelBasedOnBratsClassesII.backend

    def __init__(self, keys, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvertToMultiChannelBasedOnBratsClassesII()

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


###############################################
# Dataset manual Edema infiltrado + Edema puro
#################################################
class ConvertToMultiChannelBasedOnAnotatedInfiltration(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 2 is edema pure
    label 6 is infiltrated edema
    The possible classes are IE (infiltrated edema), PE (pure edema)
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # label 1 N_ROI
            result.append(d[key] == 6)
            # label 2 F_ROI
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d


class masked_1(MapTransform):
    # def __init__(self, keys):
    #     super().__init__(keys)

    def __call__(self, data_dict):

        B = data_dict["label"] == 2
        B = B.unsqueeze(0).expand(11, -1, -1, -1)
        data_dict["image"] = data_dict["image"] * B
        return data_dict


class ConvertToMultiChannelBasedOnBratsClassesdCustom(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is necrosis
    label 2 is edema
    label 3 is activo
    The possible classes are N (necrosis), E (edema)
    and TA (active).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # label 1 necro
            result.append(d[key] == 1)
            # label 2 is ET
            result.append(d[key] == 2)
            # merge labels 3, 4 and 3 to construct activo
            result.append(torch.logical_or(d[key] == 3, d[key] == 4))

            d[key] = torch.stack(result, dim=0)
        return d


##################
# Con infiltracion
##########################
class ConvertToMultiChannel_with_infiltration1(MapTransform):
    """
    Convert labels to Nroi + Froi + Edema:
    label 1 is necrosis
    label 2 is edema
    label 3 is activo
    The possible classes are Nroi (ROI cercana), Froi(ROI lejana), Edema

    """

    # backend = [TransformBackends.TORCH, TransformBackends.NUMPY]
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            img = d[key]
            # label 1 necro
            necro = d[key] == 1
            # result.append(necro)

            # label 2 is Edema
            edema = d[key] == 2
            # result.append(edema)

            # merge labels 3, 4 and 3 to construct activo
            active = torch.logical_or(d[key] == 3, d[key] == 4)
            # result.append(active)

            # Determinar las ROI cercana y lejana al Tumor Core
            tumor_core_mask = np.logical_or(necro, active)

            # Rellenar los huecos en la máscara
            filled_tumor_core = fill_holes_3d(tumor_core_mask)
            # result.append(torch.from_numpy(filled_tumor_core))

            # Definir el tamaño de voxel en centímetros (ajusta según tus datos)
            voxel_size_cm = 0.1

            # Expandir la máscara de 1 cm alrededor del tumor core (N_ROI)
            N_roi = expand_mask_3d_td(
                filled_tumor_core,
                edema=edema,
                distance_cm_max=0.5,  # 0.5
                distance_cm_min=0.1,  # 0.1
                voxel_size=voxel_size_cm,
            )
            result.append(N_roi)

            F_roi = expand_mask_3d_td(
                filled_tumor_core,
                edema=edema,
                distance_cm_max=10,  # 10
                distance_cm_min=1,  # 1
                voxel_size=voxel_size_cm,
            )
            result.append(F_roi)
            # result.append(edema)  # comentar para eliminar edema de GT

            d[key] = (
                torch.stack(result, dim=0)
                if isinstance(img, torch.Tensor)
                else np.stack(result, axis=0)
            )
        return d


#######################################################
## # Dataset manual Edema infiltrado + Edema puro
########################################################
class ConvertToMultiChannelBasedOnAnotatedInfiltrationII(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 2 is edema pure
    label 6 is infiltrated edema
    The possible classes are IE (infiltrated edema), PE (pure edema)
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # label 1 N_ROI
            result.append(d[key] == 6)
            # label 2 F_ROI
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d


#######################################################
# Enmascarar region
#######################################################
class masked(MapTransform):
    # def __init__(self, keys):
    #     super().__init__(keys)

    def __call__(self, data_dict):
        # Tomar B donde data_dict["label"] sea diferente de 0
        B = data_dict["label"] != 0
        # B = data_dict["label"]==2
        B = B.unsqueeze(0).expand(11, -1, -1, -1)
        data_dict["image"] = data_dict["image"] * B
        return data_dict
