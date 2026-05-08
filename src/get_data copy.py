import os
from glob import glob
from typing import List, Dict
import torch

import os
from monai.data import Dataset, DataLoader
from monai.transforms import LoadImaged, Compose
import json

from monai.transforms import (
    Compose,
    Randomizable,
    RandomizableTrait,
    Transform,
    LoadImaged,
    apply_transform,
    convert_to_contiguous,
    reset_ops_id,
)


# todas las subcarpetas finales dentro data_path
def get_final_subfolders(directory):
    final_subfolders = []
    for root, dirs, files in os.walk(directory):
        if not dirs:  # Si no hay subcarpetas, es una carpeta final
            final_subfolders.append(root)
    return final_subfolders


def get_file_data_label(
    data_path: str, label_path: str, modalities: List[str] = []
) -> List[Dict]:
    """Get the list of patient cases path as
    List[Dict]->[{'image':[paths of images], 'label':[paths of labels]}
    Arg:
         data_path(str): path directory, it has a folder by patient with MRIs
         label_path(str): path directory to labels a .nni.gz by patient
         modalities(List[str]): List of modalities to include example ['T1.', 'T1GD.', ...].
         if you want select all just setup modalities = [] or omit it
     return:
         dict_files(List[Dict]):  [{'image':[paths of images], 'label':[paths of labels]}]
    """
    list_data_ = sorted(os.listdir(data_path))
    list_lab_ = sorted(os.listdir(label_path))

    list_data_files = []
    if modalities == []:
        for folder in list_data_:
            data_mri = sorted(glob(os.path.join(data_path, folder, "*.nii.gz")))
            list_data_files.append(data_mri)
    else:
        files = [f"*{mri}*nii.gz" for mri in modalities]
        for folder in list_data_:
            data_mri = []
            for file in files:
                data_mri += glob(os.path.join(data_path, folder, file))
            list_data_files.append(data_mri)

    list_labels_files = []
    for file in list_lab_:
        label_mri = os.path.join(label_path, file)
        list_labels_files.append(label_mri)

    dict_files = []
    for i in range(len(list_labels_files)):
        diccionario = {"image": list_data_files[i], "label": list_labels_files[i]}
        dict_files.append(diccionario)

    return dict_files


##


def get_file_data_label2(
    data_path: str, split: str, modalities: List[str] = []
) -> List[Dict]:
    """Get the list of patient cases path as
    List[Dict]->[{'image':[paths of images], 'label':[paths of labels]}
    Arg:
         data_path(str): path directory, it has a folder by patient with MRIs
         label_path(str): path directory to labels a .nni.gz by patient
         split(str): Dataset split ("train" or "valid")
         modalities(List[str]): List of modalities to include example ['DSC', 'DTI', 'structural'].
         if you want to select all just set modalities = [] or omit it
     return:
         dict_files(List[Dict]):  [{'image':[paths of images], 'label':[paths of labels]}]
    """
    # obtener todas las subcarpetas dentro de data_path
    list_data_ = get_final_subfolders(os.path.join(data_path, split, "images"))

    # list_data_ = sorted(
    #     os.listdir(os.path.join(data_path, split, "images"))
    # )  # Assuming "train" or "valid" is the dataset split
    print(list_data_)

    list_data_files = []

    for modality in modalities:
        for folder in os.listdir(
            os.path.join(data_path, split, "images", f"images_{modality}")
        ):
            data_mri = sorted(
                glob(
                    os.path.join(
                        data_path,
                        split,
                        "images",
                        f"images_{modality}",
                        folder,
                        "*.nii.gz",
                    )
                )
            )
            list_data_files.append(data_mri)
    print(list_data_files)

    list_labels_files = []
    label_path = data_path

    for folder in list_data_:
        label_mri = os.path.join(label_path, split, "labels", f"{folder}.nii.gz")
        list_labels_files.append(label_mri)

    dict_files = []
    for i in range(len(list_labels_files)):
        diccionario = {"image": list_data_files[i], "label": list_labels_files[i]}
        dict_files.append(diccionario)

    return dict_files


###
def sort_image_list(lista_archivos, patrones):
    # Crear una función de clave que devuelve el índice del patrón en desired_order
    def obtener_orden(archivo):
        for i, patron in enumerate(patrones):
            if patron in archivo:
                return i
        return len(patrones)  # Si no se encuentra, lo coloca al final

    # Identificar patrones que no están en la lista de archivos
    patrones_no_encontrados = [
        patron
        for patron in patrones
        if not any(patron in archivo for archivo in lista_archivos)
    ]

    # Imprimir alerta si hay patrones no encontrados
    if patrones_no_encontrados:
        print(
            "¡Alerta! Los siguientes patrones no se encontraron en la lista de archivos:"
        )
        for patron in patrones_no_encontrados:
            print(f" - {patron}")

    # Ordenar la lista usando la función de clave
    lista_ordenada = sorted(lista_archivos, key=obtener_orden)
    return lista_ordenada


class CustomDataset(Dataset):
    def __init__(self, root_dir, section="train", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.section = section
        self.image_files, self.label_files = self._load_files()

    def __len__(self):
        return len(self.image_files)

    def _transform(self, index: int):
        """
        Fetch single data item from `self.data`.
        """
        image = self.image_files[index]
        label = self.label_files[index]
        # print(image)
        # print(label)

        if self.transform is not None:
            data = apply_transform(
                self.transform,
                data={"image": image, "label": label},
            )
            # label = apply_transform(self.transform, label)

        return data["image"], data["label"]

    # def __getitem__(self, index):
    #     if self.transform:
    #         image, label = self._transform(index=index)
    #     return {"image": image, "label": label}
    
    def __getitem__(self, index):
        if self.transform:
            image, label = self._transform(index=index)
        else:
            # Devolver paths de archivos sin transformar
            image = self.image_files[index]  # Lista de 11 modalidades
            label = self.label_files[index]  # Path de la etiqueta
        return {"image": image, "label": label}

    def _load_files(self):
        image_files, label_files = [], []
        section_path = os.path.join(self.root_dir, self.section)

        modalities = ["images_DSC", "images_DTI", "images_structural"]
        # Lista con el orden estricto de las modalidades
        modality_order = {
            "images_DSC": ["DSC_ap-rCBV", "DSC_PH", "DSC_PSR"], # "DSC_ap-rCBV", "DSC_PH", "DSC_PSR"
            "images_DTI": ["DTI_AD", "DTI_FA", "DTI_RD", "DTI_TR"],
            "images_structural": ["FLAIR", "T1.", "T1GD", "T2"],
        }

        for modality in modalities:
            modality_files = []
            modality_path = os.path.join(section_path, "images", modality)

            for n, case_folder in enumerate(os.listdir(modality_path)):
                case_path = os.path.join(modality_path, case_folder)
                # print(os.listdir(case_path), type(os.listdir(case_path)))

                # Obtener los archivos de imágenes para cada caso y modalidad /for file in sorted(os.listdir(case_path))
                lista = [
                    os.path.join(case_path, file)
                    for file in os.listdir(case_path)
                    if file.endswith(".nii.gz")
                    and not file.endswith("segmentation.nii.gz")
                    # and not file.endswith("DSC_PH.nii.gz")
                ]
                case_files = {n: sort_image_list(lista, modality_order[modality])}

                
                modality_files.append(case_files)


                # Obtener el archivo de etiqueta correspondiente
                label_file = os.path.join(
                    section_path,
                    "labels",
                    f"{case_folder}_segm.nii.gz",
                )
                if not os.path.exists(label_file):
                    label_file = os.path.join(
                        section_path,
                        "labels",
                        f"{case_folder}_combined2_approx_segm.nii.gz"
                    ) # automated_approx_segm.nii.gz / combined_approx_segm.nii.gz /combined2_approx_segm.nii.gz
                   
                    # combined2_approx_segm.nii.gz -> infiltracion + vasogenico (ConvertToMultiChannelBasedOnAnotatedInfiltration(keys="label"))
                    # combined3_approx_segm.nii.gz -> (TC-infiltracion) + vasogenico
                    # automated_approx_segm.nii.gz -> TC + edema

                # Verificar si el caso ya ha sido procesado
                if label_file not in label_files:
                    label_files.append(label_file)

            image_files.append(modality_files)

        # Lista de listas resultante

        converted_list = [[] for _ in range(len(image_files[0]))]

        for l in image_files:
            for key, values in enumerate(l):
                converted_list[key] += values[key]

        print(f"Found {len(converted_list)} images and {len(label_files)} labels.")
        # print(f"Image files: {converted_list[318]}")
        # print(f"Label files: {label_files[318]}")
        return converted_list, label_files

## Custom datset con recurrencia
class CustomDatasetRec(Dataset):
    def __init__(self, root_dir, section="train", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.section = section
        self.image_files, self.label_files, self.recurrence_files = self._load_files()

    def __len__(self):
        return len(self.image_files)

    # def _transform(self, index: int):
    #     image = self.image_files[index]  # Lista de 11 modalidades
    #     label = self.label_files[index]  # Path de la etiqueta
    #     recurrence = self.recurrence_files[index]  # Path de la imagen de recurrencia o None

    #     # Validar paths
    #     if not all(isinstance(p, str) and os.path.exists(p) for p in image):
    #         raise ValueError(f"Error en caso {index}: Algunos paths de imagen no son válidos o no existen: {image}")
    #     if not isinstance(label, str) or not os.path.exists(label):
    #         raise ValueError(f"Error en caso {index}: Label path no es válido o no existe: {label}")

    #     # Preparar datos para transformaciones
    #     data = {"image": image, "label": label, "recurrence":recurrence}
                
    #     try:
    #         if self.transform is not None:
    #             data = apply_transform(self.transform, data=data)
    #             # Ensure recurrence has correct shape
    #     except Exception as e:
    #         raise RuntimeError(f"Error aplicando transformaciones en caso {index}: {e}")        
    #     return data["image"], data["label"], data["recurrence"]
    def _transform(self, index: int):
        image = self.image_files[index]  # Lista de 11 modalidades
        label = self.label_files[index]  # Path de la etiqueta
        recurrence = self.recurrence_files[index]  # Path de la imagen de recurrencia o None

        # print(f"\nTransformando caso {index}:")
        # print(f"  Image paths: {image}")
        # print(f"  Label path: {label}")
        # print(f"  Recurrence path: {recurrence}")

        if not all(isinstance(p, str) and os.path.exists(p) for p in image):
            raise ValueError(f"Error en caso {index}: Algunos paths de imagen no son válidos o no existen: {image}")
        if not isinstance(label, str) or not os.path.exists(label):
            raise ValueError(f"Error en caso {index}: Label path no es válido o no existe: {label}")

        data = {"image": image, "label": label, "recurrence":recurrence}
        
        try:
            data = apply_transform(self.transform, data=data, log_stats=True)
            print(f"After transforms - image shape: {data['image'].shape}, label shape: {data['label'].shape}, recurrence shape: {data['recurrence'].shape}")
        except Exception as e:
            raise RuntimeError(f"Error aplicando transformaciones en caso {index}: {e}")

        if "recurrence" not in data:

            data["recurrence"] = torch.zeros(1, 128, 128, 128)  # [C, H, W, D]

        return data["image"], data["label"], data["recurrence"]

    def __getitem__(self, index):
        if self.transform:
            image, label, recurrence = self._transform(index=index)
        else:
            image = self.image_files[index]
            label = self.label_files[index]
            recurrence = self.recurrence_files[index]
        return {"image": image, "label": label, "recurrence": recurrence, }

    def _load_files(self):
        image_files, label_files, recurrence_files = [], [], []
        section_path = os.path.join(self.root_dir, self.section)

        modalities = ["images_DSC", "images_DTI", "images_structural"]
        modality_order = {
            "images_DSC": ["DSC_ap-rCBV", "DSC_PSR"], # "DSC_ap-rCBV", "DSC_PH", "DSC_PSR"
            "images_DTI": ["DTI_AD", "DTI_FA", "DTI_RD", "DTI_TR"],
            "images_structural": ["FLAIR", "T1.", "T1GD", "T2"],
        }

        for modality in modalities:
            modality_files = []
            modality_path = os.path.join(section_path, "images", modality)

            for n, case_folder in enumerate(sorted(os.listdir(modality_path))):
                case_path = os.path.join(modality_path, case_folder)
                lista = [
                    os.path.join(case_path, file)
                    for file in os.listdir(case_path)
                    if file.endswith(".nii.gz") and not file.endswith("segmentation.nii.gz") and not file.endswith("DSC_PH.nii.gz")
                ]
                case_files = {n: sort_image_list(lista, modality_order[modality])}
                modality_files.append(case_files)

                # Extraer el ID base del caso (por ejemplo, UPENN-GBM-00307)
                case_id = case_folder.split("_")[0]  # Toma UPENN-GBM-00307 de UPENN-GBM-00307_11
                # print(f"Procesando case_folder: {case_folder}, case_id: {case_id}")  # Depuración

                label_file = os.path.join(section_path, "labels", f"{case_folder}_segm.nii.gz")
                if not os.path.exists(label_file):
                    label_file = os.path.join(
                        section_path, "labels", f"{case_folder}_combined2_approx_segm.nii.gz"
                    )
                    # combined2_approx_segm.nii.gz -> infiltracion + vasogenico
                    # combined3_approx_segm.nii.gz -> (TC-infiltracion) + vasogenico
                    # automated_approx_segm.nii.gz -> TC + edema

                # Obtener el archivo de recurrencia correspondiente
                recurrence_path = os.path.join(section_path, "recurrence")
                recurrence_file = os.path.join(recurrence_path, f"{case_id}_21_T1GD_flo_reg.nii.gz")
                # print(f"Generando recurrence path para {case_id}: {recurrence_file}")  # Depuración
                if not os.path.exists(recurrence_file):
                    print(f"Advertencia: No se encontró recurrence file: {recurrence_file}")
                    recurrence_file = None
                    

                if label_file not in label_files and os.path.exists(label_file):
                    label_files.append(label_file)
                    recurrence_files.append(recurrence_file)

            image_files.append(modality_files)

        converted_list = [[] for _ in range(len(image_files[0]))]
        for l in image_files:
            for key, values in enumerate(l):
                converted_list[key] += values[key]

        print(f"Found {len(converted_list)} images, {len(label_files)} labels, "
              f"and {len(recurrence_files)} recurrence files.")
        return converted_list, label_files, recurrence_files

### Datset para Segmentaci'on de N, Edema y Activo ###
class CustomDatasetSeg(Dataset):
    def __init__(self, root_dir, section="train", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.section = section
        self.image_files, self.label_files = self._load_files()

    def __len__(self):
        return len(self.image_files)

    def _transform(self, index: int):
        """
        Fetch single data item from `self.data`.
        """
        image = self.image_files[index]
        label = self.label_files[index]

        if self.transform is not None:
            data = apply_transform(
                self.transform,
                data={"image": image, "label": label},
                log_stats=True,
            )
            # label = apply_transform(self.transform, label)

        return data["image"], data["label"]

    def __getitem__(self, index):
        # image_path = self.image_files[index]
        # label_path = self.label_files[index]
        # image, label = self._load_data(image_path, label_path)
        if self.transform:
            image, label = self._transform(index=index)
            # print(image.shape, label.shape)
        return {"image": image, "label": label}

    def _load_files(self):
        image_files, label_files = [], []
        section_path = os.path.join(self.root_dir, self.section)

        modalities = ["images_structural"]

        for modality in modalities:
            modality_files = []
            modality_path = os.path.join(section_path, "images", modality)

            for n, case_folder in enumerate(os.listdir(modality_path)):
                case_path = os.path.join(modality_path, case_folder)

                # Obtener los archivos de imágenes para cada caso y modalidad
                case_files = {
                    n: [
                        os.path.join(case_path, file)
                        for file in os.listdir(case_path)
                        if file.endswith(".nii.gz")
                    ]
                }

                modality_files.append(case_files)

                # Obtener el archivo de etiqueta correspondiente
                label_file = os.path.join(
                    section_path,
                    "labels",
                    f"{case_folder}_segm.nii.gz",
                )
                if not os.path.exists(label_file):
                    label_file = os.path.join(
                        section_path,
                        "labels",
                        f"{case_folder}_automated_approx_segm.nii.gz",
                    )

                # _automated_approx_segm / _segm

                # Verificar si el caso ya ha sido procesado
                if label_file not in label_files:
                    label_files.append(label_file)

            image_files.append(modality_files)

        # Lista de listas resultante

        converted_list = [[] for _ in range(len(image_files[0]))]

        for l in image_files:
            for key, values in enumerate(l):
                converted_list[key] += values[key]

        print(f"Found {len(converted_list)} images and {len(label_files)} labels.")
        # print(f"Image files: {converted_list}")
        # print(f"Label files: {label_files}")
        return converted_list, label_files


#####################
### Combinar lebels y guardar###
####################################
import nibabel as nib
import numpy as np


# Función para guardar el volumen combinado
def save_img(I_img, savename, header=None, affine=None):
    if header is None or affine is None:
        affine = np.diag([1, 1, 1, 1])
        new_img = nib.Nifti1Image(I_img, affine)
    else:
        new_img = nib.Nifti1Image(I_img, affine, header=header)

    nib.save(new_img, savename)


# Función para combinar los volúmenes
def combine_labels(path_label1, path_label2, output_path):
    # Leer los volúmenes de segmentación
    label1_img = nib.load(path_label1)
    label2_img = nib.load(path_label2)

    # Obtener los datos de los volúmenes
    label1_data = label1_img.get_fdata()
    label2_data = label2_img.get_fdata()

    # Imprimir los valores que toma el volumen
    print(np.unique(label1_data))
    valores = np.unique(label2_data)
    print(valores)

    # Crear un nuevo volumen inicializado en cero
    combined_data = np.zeros_like(label1_data)

    # Conservar solo los voxeles con valor 2 en label1 o valor 6 en label2
    combined_data[(label1_data == 2.0)] = 2.0
    # Si el valor es 6 en label2, y se intercepta con el valor 4 en label1, se conserva el valor 6
    combined_data[(label1_data == 2.0) & (label2_data == valores[2])] = 6.0
    # combined_data[(label2_data == valores[1])] = 6.0

    # Guardar el nuevo volumen combinado
    save_img(
        combined_data, output_path, header=label1_img.header, affine=label1_img.affine
    )

def combine_labels_recurrence(path_label1, path_label2, output_path):
    # Leer los volúmenes de segmentación
    label1_img = nib.load(path_label1)
    label2_img = nib.load(path_label2)

    # Obtener los datos de los volúmenes
    label1_data = label1_img.get_fdata() # Seg Image base
    label2_data = label2_img.get_fdata() # Seg Image follow registrada

    # Imprimir los valores que toma el volumen
    print(np.unique(label1_data))
    valores = np.unique(label2_data)
    print(valores)

    # Crear un nuevo volumen inicializado en cero
    combined_data = np.zeros_like(label1_data)

    #### Infiltracion (6) + edema no infiltrado (2) ###############
    # # Conservar solo los voxeles con valor 2 en label1 o valor 6 en label2
    # combined_data[(label1_data == 2.0)] = 2.0

    #  # Obtener los valores de interés: posición 1 y todas las posiciones >= 3
    # valores_interes = np.concatenate(([valores[1]], valores[3:]))  # Unión de posición 1 y >= 3

    # # Si el valor de label2 está en valores_interes y label1 es 2.0, asignar 6.0
    # for v in valores_interes:
    #     combined_data[(label1_data == 2.0) & (label2_data == v)] = 6.0
    ################################################################################
    #### TC + Infiltracion (6) - Edema (2)
    # valores_interes = np.concatenate(([valores[1]], valores[2])) 
    combined_data[(label2_data == valores[1])] = 2.0
    if len(valores)>2:
        combined_data[(label2_data == valores[2])] = 6.0
    combined_data[(label1_data == 1.0) | (label1_data == 4)] = 6.0

    # Guardar el nuevo volumen combinado
    save_img(
        combined_data, output_path, header=label1_img.header, affine=label1_img.affine
    )


# ####################
# # Create loader
# ###########################
# from monai import data

# def load_files(root_dir, fold):
#         image_files, label_files = [], []
#         section_path = os.path.join(root_dir, fold)

#         modalities = ["images_structural"]

#         for modality in modalities:
#             modality_files = []
#             modality_path = os.path.join(section_path, "images", modality)

#             for n, case_folder in enumerate(os.listdir(modality_path)):
#                 case_path = os.path.join(modality_path, case_folder)

#                 # Obtener los archivos de imágenes para cada caso y modalidad
#                 case_files = {
#                     n: [
#                         os.path.join(case_path, file)
#                         for file in os.listdir(case_path)
#                         if file.endswith(".nii.gz")
#                     ]
#                 }

#                 modality_files.append(case_files)

#                 # Obtener el archivo de etiqueta correspondiente
#                 label_file = os.path.join(
#                     section_path,
#                     "labels",
#                     f"{case_folder}_segm.nii.gz",
#                 )
#                 if not os.path.exists(label_file):
#                     label_file = os.path.join(
#                         section_path,
#                         "labels",
#                         f"{case_folder}_automated_approx_segm.nii.gz",
#                     )

#                 # _automated_approx_segm / _segm

#                 # Verificar si el caso ya ha sido procesado
#                 if label_file not in label_files:
#                     label_files.append(label_file)

#             image_files.append(modality_files)
#         # Lista de listas resultante

#         converted_list = [[] for _ in range(len(image_files[0]))]

#         for l in image_files:
#             for key, values in enumerate(l):
#                 converted_list[key] += values[key]

#         print(f"Found {len(converted_list)} images and {len(label_files)} labels.")
#         return converted_list, label_files

# def custom_get_loader(data_dir, set_transform, batch_size=1, fold="train", workers=8):
#     data_dir = data_dir
#     set_files = load_files(data_dir, fold=fold)

#     set_ds = data.Dataset(data=set_files, transform=set_transform)

#     set_loader = data.DataLoader(
#         set_ds,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=workers,
#         pin_memory=True,
#     )
#     return set_ds, set_loader
