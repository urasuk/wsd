#!/usr/bin/env python3
# !pip install datasets evaluate transformers[sentencepiece]

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# targets_df_train_new = Датасет, де леми НЕ огорнуті <ti></ti>
DATASET_NAMES = [
    "targets_df_train_bert_together.jsonl",
    "targets_df_test_bert_together.jsonl",
]
# PS: targets_df_train_new - в цьому датасеті є баг: коли я брав токени леми наприклад 'річка' з input_ids,
#     то, якщо в речені було слово 'порічка' то брало і б частину 'поРІЧКА'
FILE_PATH_TRAIN_DF = f"/Users/yurayano/PycharmProjects/wsd/models/data_train_dedup/{DATASET_NAMES[0]}"
FILE_PATH_VALID_DF = f"/Users/yurayano/PycharmProjects/wsd/models/data_train_dedup/{DATASET_NAMES[1]}"
FILE_PATH_TO_SAVE_MODEL = "./weights/model_bert_no_tags_together.pth"

data_files = {
    "train": FILE_PATH_TRAIN_DF,
    "valid": FILE_PATH_VALID_DF,
}

# TRAINING PARAMS 
EPOCH_NUM = 10           # PS: ⛔️⛔️⛔️ Можливо варто збільшити...
LEARNING_RATE = 0.00005  # PS: ⛔️⛔️⛔️ Можливо варто збільшити...
BATCH_SIZE = 64          # PS: ⛔️⛔️⛔️  Варто збільшити
PLOT_LOSS_AND_ACC = True
EVALUTION_FREQ = 100  # batches
ACCURACY_TRESHOLD = 0.5 
PROJ_TARGET_LEN = 64     # PS: ⛔️⛔️⛔️ Можливо варто збільшити довжину векторів на виході з self.fc


checkpoint = "bert-base-multilingual-cased"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)


class SiameseNNBatch(nn.Module):
    def __init__(self, checkpoint, proj_target_len=PROJ_TARGET_LEN):
        super(SiameseNNBatch, self).__init__()
        self.bert = AutoModel.from_pretrained(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.fc = nn.Sequential(nn.Linear(self.bert.config.hidden_size, proj_target_len))
        # nn.ReLU(),
        # nn.Linear(64, 1),
        # nn.Sigmoid())

    def forward(self, batch):
        input_ids = batch["input_ids"]
        token_type_ids = batch["token_type_ids"]
        attention_mask = batch["attention_mask"]

        # Mask that tells indeces of tokens of target word in a sentence:
        # [0, 0, 0, 1, 1, 0, 0] -> [3, 4]
        target_1_mask = batch["target_1_mask"]
        target_2_mask = batch["target_2_mask"]

        # bert_outputs = self.bert(
        #     input_ids=input_ids,
        #     token_type_ids=token_type_ids,
        #     attention_mask=attention_mask,
        # )
        bert_outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True
        )

        # Get last hidden state from Bert output
        hidden_state = bert_outputs.last_hidden_state                   # (batch_size, seq_len, hidden_size)
        hidden_states = bert_outputs['hidden_states'][-4:]  # List of tensors, last 4 hidden states


        # Compute average hidden state of target word in each sentence
        sum_target_1 = (target_1_mask.unsqueeze(-1) * hidden_state).sum(dim=1)  # (batch_size, hidden_size)
        sum_target_2 = (target_2_mask.unsqueeze(-1) * hidden_state).sum(dim=1)  # (batch_size, hidden_size)
        avg_target_1 = sum_target_1 / target_1_mask.sum(dim=1).unsqueeze(-1)    # (batch_size, hidden_size)
        avg_target_2 = sum_target_2 / target_2_mask.sum(dim=1).unsqueeze(-1)    # (batch_size, hidden_size)

        # Pass through the fully connected layers
        proj_target_1 = self.fc(avg_target_1)
        proj_target_2 = self.fc(avg_target_2)

        # Обчисліть cosine similarity
        cosine_sim = F.cosine_similarity(proj_target_1, proj_target_2, dim=-1)
        return cosine_sim
    

def evaluate_model(model, device, eval_dataloader, criterion):
    if eval_dataloader is None:
        return None, None

    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch, batch_labels in eval_dataloader:
            # Move batch data and labels to GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_labels = {k: v.to(device) for k, v in batch_labels.items()}

            outputs = model(batch)
            loss = criterion(outputs, batch_labels["labels"])
            running_loss += loss.item()

            # Calculate the number of correct predictions for accuracy
            correct_predictions += (
                (outputs > ACCURACY_TRESHOLD).long() == batch_labels["labels"]
            ).sum().item()
            total_samples += len(batch_labels["labels"])

    # Calculate average loss and accuracy
    eval_loss = running_loss / len(eval_dataloader)
    eval_accuracy = correct_predictions / total_samples

    return eval_loss, eval_accuracy

def has_none_values(batch):
    return any((value is None) or torch.isnan(value) or (value > 1) or (value < -1) for value in batch)

def train(model, train_dataloader, device, optimizer, criterion, num_epochs, eval_dataloader=None):

    # Lists to store loss and accuracy values
    train_losses = []
    train_accuracies = []

    eval_losses = []
    eval_accuracies = []

    batches_since_eval = 0
    batches_since_epoch = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Iterate over the training dataset
        for batch, batch_labels in tqdm(
            train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"
        ):
            # Move batch data and labels to GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_labels = {k: v.to(device) for k, v in batch_labels.items()}

            optimizer.zero_grad()
            outputs = model(batch)

            if has_none_values(outputs):
                print("Error: Batch contains None values")

            loss = criterion(
                outputs, batch_labels["labels"]
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate the number of correct predictions for accuracy
            correct_predictions += (
                (outputs > ACCURACY_TRESHOLD).float() == batch_labels["labels"]
            ).sum().item()
            total_samples += len(batch_labels["labels"])

            # Evaluate the model every EVALUTION_FREQ batches
            batches_since_eval += 1
            batches_since_epoch += 1
            if eval_dataloader is not None and batches_since_eval >= EVALUTION_FREQ:
                epoch_loss = running_loss / batches_since_epoch
                epoch_accuracy = correct_predictions / total_samples
                eval_loss, eval_accuracy = evaluate_model(model, device, eval_dataloader, criterion)
                print(f"\nEpoch {epoch+1}, Train_Loss: {epoch_loss:.2f}, Train_Accuracy: {epoch_accuracy:.2f}, "
                      f"Eval_Loss: {eval_loss:.2f}, Eval_Accuracy: {eval_accuracy:.2f}\n")
                batches_since_eval = 0

        # Calculate average loss and accuracy for this epoch
        epoch_loss = running_loss / len(train_dataloader)
        epoch_accuracy = correct_predictions / total_samples

        # Append loss and accuracy values to lists
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        # Compute the evaluation loss and accuracy
        eval_loss, eval_accuracy = evaluate_model(model, device, eval_dataloader, criterion)

        # Append loss and accuracy values to lists
        eval_losses.append(eval_loss)
        eval_accuracies.append(eval_accuracy)

        # Print the average loss and accuracy for this epoch
        print(f"\nEpoch {epoch+1}, Train_Loss: {epoch_loss:.2f}, Train_Accuracy: {epoch_accuracy:.2f}, "
              f"Eval_Loss: {eval_loss:.2f}, Eval_Accuracy: {eval_accuracy:.2f} \n")

    return model, train_losses, train_accuracies, eval_losses, eval_accuracies


def my_collate_fn(features, return_tensors="pt"):
    def select_columns(xs, columns):
        result = []
        for x in xs:
            result.append({k: x[k] for k in columns})
        return result

    def make_mask(batch, feature):
        mask = torch.zeros_like(batch['input_ids'])
        for i, xs in enumerate(features):
            # mask[i][xs[feature]] = 1
            mask[i][xs[feature].flatten()] = 1
        return mask

    inputs = select_columns(features, ['input_ids', 'attention_mask', 'token_type_ids'])
    batch_inputs = data_collator(inputs)
    batch_inputs['target_1_mask'] = make_mask(batch_inputs, 'sent1_target_tokens_indexes')
    batch_inputs['target_2_mask'] = make_mask(batch_inputs, 'sent2_target_tokens_indexes')

    # Виділення міток з функції
    labels = [float(x['label']) for x in features]

    # Додавання міток до окремого словника даних
    batch_labels = {'labels': torch.tensor(labels)}

    # Повернення окремих пакунків даних для бачів та міток
    return batch_inputs, batch_labels


def plot_loss_accuracy(losses, accuracies, df_name="Train", title="Loss and Accuracy"):
    # Plot loss and accuracy
    plt.figure(figsize=(10, 5))
    plt.suptitle(title)  # Title for the entire figure

    plt.subplot(1, 2, 1)
    plt.plot(losses, label=f"{df_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{df_name} Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label=f"{df_name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{df_name} Accuracy")
    plt.legend()
    plt.show()


def main():
    os.makedirs(os.path.dirname(FILE_PATH_TO_SAVE_MODEL), exist_ok=True)

    # Load the dataset
    no_tags_datasets = load_dataset(
        "json",
        data_files=data_files,
    )

    no_tags_datasets.set_format("torch")

    train_dataloader = DataLoader(
        no_tags_datasets["train"],
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=my_collate_fn,
    )
    eval_dataloader = DataLoader(
        no_tags_datasets["valid"],
        batch_size=BATCH_SIZE, 
        collate_fn=my_collate_fn
    )

    model = SiameseNNBatch(checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move model to GPU
    model.to(device)

    # Define your loss function
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()

    # Define your optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training ...
    model, train_losses, train_accuracies, eval_losses, eval_accuracies = train(
        model,
        train_dataloader,
        device,
        optimizer,
        criterion,
        num_epochs=EPOCH_NUM,
        eval_dataloader=eval_dataloader
    )
    
    # Save model and additional training information
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "eval_losses": eval_losses,
            "eval_accuracies": eval_accuracies,
        },
        FILE_PATH_TO_SAVE_MODEL,
    )

    if PLOT_LOSS_AND_ACC:
        plot_loss_accuracy(train_losses, train_accuracies, "Train", "Bert (no tags) (together)")
        plot_loss_accuracy(eval_losses, eval_accuracies, "Eval", "Bert (no tags) (together)")


if __name__ == "__main__":
    main()