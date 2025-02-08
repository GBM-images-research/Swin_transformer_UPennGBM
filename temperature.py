import torch
from torch import nn, optim
from torch.nn import functional as F
from monai.inferers import sliding_window_inference
from functools import partial
import gc


class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        temperature = self.temperature.view(-1, 1, 1, 1, 1).expand_as(logits)
        return logits / temperature

    def infer(self, data):
        model_inferer = partial(
            sliding_window_inference,
            roi_size=[64, 64, 64],
            sw_batch_size=1,
            predictor=self.model,
            overlap=0.5,
        )
        return model_inferer(data)

    def set_temperature(self, valid_loader):
        self.cuda()
        nll_criterion = nn.BCEWithLogitsLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        total_loss = 0.0
        total_ece = 0.0
        total_samples = 0

        with torch.no_grad():
            for idx, batch_data in enumerate(valid_loader):
                input, label = batch_data["image"], batch_data["label"]
                input = input.cuda()
                label = label.cuda()
                logits = self.model(input)
                                
                scaled_logits = self.temperature_scale(logits)
                
                loss = nll_criterion(scaled_logits, label).item()
                ece = ece_criterion(scaled_logits, label).item()

                total_loss += loss * input.size(0)
                total_ece += ece * input.size(0)
                total_samples += input.size(0)

                # Desacoplar los tensores del grafo antes de liberar memoria
                logits = logits.detach()
                scaled_logits = scaled_logits.detach()
                del logits, scaled_logits, input, label
                gc.collect()
                torch.cuda.empty_cache()
                print(f"🔹 Calculated logits case {idx}")

        print('Before temperature - NLL: %.3f, ECE: %.3f' % (total_loss / total_samples, total_ece / total_samples))
        
        self.temperature.requires_grad_(True)
        # optimizer = optim.LBFGS([self.temperature], lr=0.01) # , max_iter=50
        optimizer = optim.Adam([self.temperature], lr=0.01)  # 🔹 Cambiado LBFGS → Adam

        # def eval():
        #     optimizer.zero_grad()
        #     loss = 0.0
        #     with torch.no_grad():
        #         for idx, batch_data in enumerate(valid_loader):
        #             input, label = batch_data["image"], batch_data["label"]
        #             input = input.cuda()
        #             label = label.cuda()
        #             logits = self.infer(input)
        #             scaled_logits = self.temperature_scale(logits)
        #             loss += nll_criterion(scaled_logits, label)

        #             # Desacoplar los tensores del grafo antes de liberar memoria
        #             logits = logits.detach()
        #             scaled_logits = scaled_logits.detach()
        #             del logits, scaled_logits, input, label
        #             torch.cuda.empty_cache()

        #     loss.backward()
        #     return loss

        # optimizer.step(eval)
        total_loss_after = 0.0
        total_ece_after = 0.0
        total_samples_after = 0
        for iteration in range(20):  # 🔹 50 iteraciones
            optimizer.zero_grad()

            loss = 0.0
            ece  = 0.0
            # with torch.no_grad():  # 🔹 Evita el tracking de gradientes innecesario
            for idx, batch_data in enumerate(valid_loader):
                    input, label = batch_data["image"], batch_data["label"]
                    input = input.cuda()
                    label = label.cuda()
                    logits = self.model(input)
                    scaled_logits = self.temperature_scale(logits)

                    loss += nll_criterion(scaled_logits, label)
                    ece += ece_criterion(scaled_logits, label)
                    # total_loss_after += loss * input.size(0)
                    # total_ece_after += ece * input.size(0)
                    # total_samples_after += input.size(0)

                    # # Desacoplar los tensores del grafo antes de liberar memoria
                    # logits = logits.detach()
                    # scaled_logits = scaled_logits.detach()

                    # del logits, scaled_logits, input, label
                    # torch.cuda.empty_cache()
            
            loss.backward()
            optimizer.step()

            logits = logits.detach()
            scaled_logits = scaled_logits.detach()
            del logits, scaled_logits, input, label
            gc.collect()
            torch.cuda.empty_cache()

            # 🔹 Mostrar información en cada iteración
            print(f"🟢 Iteración {iteration+1}/20 - Loss: {loss.item() / total_samples:.5f} -NCC: {ece.item() / total_samples:.5f}- Temp: {self.temperature.item():.3f}")
        
        avg_loss = loss.item() / total_samples
        avg_ece = ece.item() / total_samples

        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (avg_loss, avg_ece))

        return self

class _ECELoss(nn.Module):
    def __init__(self, n_bins=15):
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        probabilities = torch.sigmoid(logits)
        confidences, _ = torch.max(probabilities, 1)
        labels = labels.max(dim=1)[1]  # Obtener las etiquetas de clase en lugar de tensor binario
        predictions = (probabilities >= 0.5).long()  # Convertir probabilidades en predicciones binarias
        # accuracies = predictions.eq(labels)
        accuracies = predictions.argmax(dim=1).eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            # in_bin_expanded = in_bin.unsqueeze(1).expand_as(accuracies)  # Expandir la máscara para que coincida con accuracies
            in_bin_expanded = in_bin.expand_as(accuracies)
            # print("logits shape:", logits.shape)
            # print("labels shape:", labels.shape)
            # print("in_bin_expanded shape:", in_bin_expanded.shape)
            # Aplanar las dimensiones
            accuracies_flat = accuracies.reshape(-1)
            in_bin_flat = in_bin_expanded.reshape(-1)
            confidences_flat = confidences.reshape(-1)

            if in_bin_flat.float().mean().item() > 0:
                accuracy_in_bin = accuracies_flat[in_bin_flat].float().mean()
                avg_confidence_in_bin = confidences_flat[in_bin_flat].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * in_bin_flat.float().mean()

        return ece