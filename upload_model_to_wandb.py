import wandb

#   # Guardar artefacto en W&B
#     artifact_name = f"{wandb.run.id}_best_model"
#     at = wandb.Artifact(artifact_name, type="model")
#     at.add_file(os.path.join(directory, "model.pt"))
#     wandb.log_artifact(at, aliases=["final"])

# Configura el proyecto y la corrida
project_name = "Swin_UPENN_10casos_pruebas"
run_name = "splendid-dream-5"

# Inicializa WandB en el proyecto y la corrida especificada
wandb.init(project=project_name, name=run_name, resume="allow")

# Define el nombre del archivo y la metadata si es necesario
artifact_name = "best_checkpoint"
artifact_type = "model"
artifact_description = "Best model checkpoint for the Swin_UPENN_10casos_pruebas project"

# Crea un artefacto en WandB
artifact = wandb.Artifact(
    artifact_name,
    type=artifact_type,
    description=artifact_description,
)

# Adjunta el archivo del modelo
artifact.add_file("Dataset/model.pt")

# Sube el artefacto
wandb.log_artifact(artifact)

# Finaliza la corrida
wandb.finish()
