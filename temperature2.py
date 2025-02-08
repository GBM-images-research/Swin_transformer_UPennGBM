# import torch
# import torch.nn.functional as F
# from torch.optim import LBFGS
# import numpy as np
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from functools import partial

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def infer(data, model):
        # model.eval()
        model_inferer = partial(
                sliding_window_inference,
                roi_size=[128, 128, 128],
                sw_batch_size=2,
                predictor=model,
                overlap=0.6,
            )

        with torch.no_grad():
                logits = model_inferer(data)

        return logits

# class TemperatureScaling(torch.nn.Module):
#     def __init__(self):
#         super(TemperatureScaling, self).__init__()
#         self.temperature = torch.nn.Parameter(torch.ones(1) * 1.5)

#     def forward(self, logits):
#         return logits / self.temperature.to(device)

# def find_optimal_temperature(model, val_loader, criterion):
#     # Set model to evaluation mode
#     model.eval()
#     model.to(device)
    
#     # Initialize temperature scaling model
#     temp_model = TemperatureScaling().to(device)
#     optimizer = LBFGS([temp_model.temperature], lr=0.1, history_size=100, 
#                     max_iter=20, 
#                     line_search_fn="strong_wolfe")
#     c=0
#     call_count = 0 
#     def _eval():
#         nonlocal call_count
#         call_count += 1
#         optimizer.zero_grad()
#         nll = 0
        
#         for data in tqdm(val_loader):
#                 images = data["image"].to(device)
#                 labels = data["label"].to(device)
#                 logits = infer(images, model)
                                
#                 # Apply temperature scaling
#                 scaled_logits = temp_model(logits)
                
#                 # Calculate loss F.sigmoid(
#                 loss = criterion(torch.sigmoid(scaled_logits), labels)
#                 # loss.backward()  # Backward pass to compute gradients
#                 nll += loss
#         nll.backward()
#         # nll /= len(val_loader)
#         # print(nll.item(), temp_model.temperature.item())
#         print(f"Iteración: {call_count}, Temperatura: {temp_model.temperature.item()}, Loss: {nll}")
#         # nll.backward()
#         return nll
#     for i in range(10):
#         optimizer.step(_eval)
#     print(f"Total closure calls: {call_count}")

#     return temp_model.temperature.item()

import torch
import torch.nn as nn
import torch.optim as optim


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 5.6)  # Valor inicial de temperatura

    def forward(self, logits):
        return logits / self.temperature

def find_optimal_temperature(model, val_loader, criterion, device="cuda"):
    model.eval()
    model.to(device)
    temp_scaler = TemperatureScaler().to(device)
    
    optimizer = optim.LBFGS([temp_scaler.temperature], lr=0.05, max_iter=100)
    call_count=0

    def closure():
        nonlocal call_count
        call_count += 1
        optimizer.zero_grad()
        loss = 0.0
        num_batches = len(val_loader)
        # with torch.no_grad():
        for data in tqdm(val_loader):
                images, labels = data["image"].to(device), data["label"].to(device)
                logits = infer(images, model) # Obtener logits de la red
                scaled_logits = temp_scaler(logits)  # Aplicar temperatura
                loss += criterion(scaled_logits, labels)  # Calcular pérdida
        loss = loss / num_batches
        loss.backward()
        print(f"Iteración: {call_count}, Temperatura: {temp_scaler.temperature.item()}, Loss: {loss}")
        return loss

    optimizer.step(closure)

    return temp_scaler.temperature.item()