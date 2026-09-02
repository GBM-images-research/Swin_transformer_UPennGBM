import os
import shutil
import pandas as pd

# =====================================================================
# 1. CONFIGURACIÓN DE RUTAS
# =====================================================================
CSV_PATH = 'Dataset/MU_glioma/MU-Glioma-Post_ClinicalData-July2025_CSV.csv'
TARGET_DIR = 'Dataset/MU_glioma' 

print("="*60)
print("INICIANDO SEGMENTACIÓN TEMPORAL (ESTRICTAMENTE NO VISTOS)")
print("="*60)

# =====================================================================
# 2. AUDITORÍA DE PACIENTES NO VISTOS (El candado de seguridad)
# =====================================================================
def get_patients_from_dir(directory):
    if not os.path.exists(directory):
        print(f"Advertencia: No se encontró la ruta {directory}")
        return set()
    # Retorna solo las subcarpetas (que corresponden a los Patient_IDs)
    return {d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))}

val_dir = os.path.join(TARGET_DIR, 'val')
test_dir = os.path.join(TARGET_DIR, 'test')

val_patients = get_patients_from_dir(val_dir)
test_patients = get_patients_from_dir(test_dir)

# Unimos ambos sets para formar nuestro grupo de evaluación
unseen_patients = val_patients.union(test_patients)

if not unseen_patients:
    raise ValueError("No se encontraron pacientes en las carpetas val/test. Verifica TARGET_DIR.")

print(f"Pacientes detectados en Val  : {len(val_patients)}")
print(f"Pacientes detectados en Test : {len(test_patients)}")
print(f"Total de pacientes seguros (Unseen) : {len(unseen_patients)}")

# =====================================================================
# 3. CARGA DEL CSV Y CÁLCULO DE INTERVALOS
# =====================================================================
df = pd.read_csv(CSV_PATH, sep=';')
df.columns = [col.strip() for col in df.columns]
df_gbm = df[df['Primary Diagnosis'] == 'GBM'].copy() 

timepoint_cols = [
    'Number of Days from Diagnosis to 1st MRI (Timepoint_1)',
    'Number of Days from Diagnosis to 2nd MRI (Timepoint_2)',
    'Number of Days from Diagnosis to 3rd MRI (Timepoint_3)',
    'Number of Days from Diagnosis to 4th MRI (Timepoint_4)',
    'Number of Days from Diagnosis to 5th MRI (Timepoint_5)',
    'Number of Days from Diagnosis to 6th MRI (Timepoint_6)'
]

for col in timepoint_cols:
    df_gbm[col] = pd.to_numeric(df_gbm[col], errors='coerce')

all_pairs = []
for _, row in df_gbm.iterrows():
    # FILTRO DE SEGURIDAD MÁXIMA: Si el paciente no está en val/test, se ignora.
    if row['Patient_ID'] not in unseen_patients:
        continue
        
    for i in range(len(timepoint_cols) - 1):
        t_start_col = timepoint_cols[i]
        t_end_col = timepoint_cols[i+1]
        
        val_start = row[t_start_col]
        val_end = row[t_end_col]
        
        if pd.notna(val_start) and pd.notna(val_end):
            delta_t = int(val_end - val_start)
            # Solo guardamos pares con salto temporal válido (0 < Delta <= 120)
            if 0 < delta_t <= 120:
                all_pairs.append({
                    'Patient_ID': row['Patient_ID'],
                    'Pair_Type': f"Timepoint_{i+1} -> Timepoint_{i+2}",
                    'Day_T': int(val_start),
                    'Day_T_plus_1': int(val_end),
                    'Delta_T_Days': delta_t
                })

df_unseen = pd.DataFrame(all_pairs)

# =====================================================================
# 4. ABLACIÓN TEMPORAL (Grupos de 60, 90 y 120 días)
# =====================================================================
df_test60 = df_unseen[df_unseen['Delta_T_Days'] <= 60].copy()
df_test90 = df_unseen[(df_unseen['Delta_T_Days'] > 60) & (df_unseen['Delta_T_Days'] <= 90)].copy()
df_test120 = df_unseen[(df_unseen['Delta_T_Days'] > 90) & (df_unseen['Delta_T_Days'] <= 120)].copy()

print("\n--- RESUMEN DE LA PARTICIÓN TEMPORAL ---")
print(f"Total de pares válidos extraídos : {len(df_unseen)}")
print(f"  -> Grupo test60  (0 - 60 días)      : {len(df_test60)} pares")
print(f"  -> Grupo test90  (61 - 90 días)     : {len(df_test90)} pares")
print(f"  -> Grupo test120 (91 - 120 días)    : {len(df_test120)} pares")

# =====================================================================
# 5. FUNCIONES DE EXTRACCIÓN Y COPIA FÍSICA (ADAPTADA A TU ENTORNO)
# =====================================================================
def extract_needed_timepoints(df_split):
    patient_tps = {}
    for _, row in df_split.iterrows():
        pid = row['Patient_ID']
        pair_string = row['Pair_Type']
        
        t_str, t_plus_1_str = pair_string.split(' -> ')
        t_idx = int(t_str.split('_')[1])
        t_plus_1_idx = int(t_plus_1_str.split('_')[1])
        
        if pid not in patient_tps:
            patient_tps[pid] = set()
            
        patient_tps[pid].add(t_idx)
        patient_tps[pid].add(t_plus_1_idx)
        
    return patient_tps

def build_dataset_split(split_name, df_split, target_dir):
    if df_split.empty:
        print(f"\nSaltando {split_name.upper()} (No hay casos en este rango).")
        return

    print(f"\nProcesando el conjunto de {split_name.upper()}...")
    patient_tps = extract_needed_timepoints(df_split)
    
    split_target_dir = os.path.join(target_dir, split_name)
    os.makedirs(split_target_dir, exist_ok=True)
    
    archivos_copiados = 0
    
    for pid, timepoints in patient_tps.items():
        for tp in timepoints:
            tp_folder_name = f"Timepoint_{tp}"
            target_tp_dir = os.path.join(split_target_dir, pid, tp_folder_name)
            
            # Búsqueda dinámica: intentamos leer desde 'val' o desde 'test'
            source_tp_dir = None
            for folder_origen in ['val', 'test']:
                ruta_posible = os.path.join(target_dir, folder_origen, pid, tp_folder_name)
                if os.path.exists(ruta_posible):
                    source_tp_dir = ruta_posible
                    break
            
            # Si encontramos el timepoint en cualquiera de las dos carpetas, copiamos
            if source_tp_dir is not None:
                os.makedirs(target_tp_dir, exist_ok=True)
                for file_name in os.listdir(source_tp_dir):
                    if file_name.endswith('.nii.gz'):
                        src_file = os.path.join(source_tp_dir, file_name)
                        dst_file = os.path.join(target_tp_dir, file_name)
                        shutil.copy2(src_file, dst_file)
                        archivos_copiados += 1
            else:
                print(f"  [ADVERTENCIA] No se localizó {tp_folder_name} para el paciente {pid} en val ni en test.")
                
    print(f"✓ {split_name.upper()}: Se copiaron {archivos_copiados} archivos NIfTI.")

# =====================================================================
# 6. EJECUCIÓN FÍSICA (GENERACIÓN DE CARPETAS TEMPORALES)
# =====================================================================
# Genera test60, test90 y test120 directamente en Dataset/MU_glioma/
build_dataset_split('test60', df_test60, TARGET_DIR)
build_dataset_split('test90', df_test90, TARGET_DIR)
build_dataset_split('test120', df_test120, TARGET_DIR)

print("\n" + "="*60)
print(f"¡Estructuración temporal completada con éxito en {TARGET_DIR}!")
print("="*60)