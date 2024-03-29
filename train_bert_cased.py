#!/usr/bin/env python3
# !pip install datasets evaluate transformers[sentencepiece]

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from evaluate import evaluate_model 


FILE_PATH_TRAIN_DF = "./data/drive/targets_df_train.jsonl"
FILE_PATH_TEST_DF = "./data/drive/targets_df_test.jsonl"
data_files = {
    "train": FILE_PATH_TRAIN_DF, 
    "test":  FILE_PATH_TEST_DF,
}
FILE_PATH_TO_SAVE_MODEL = './weights/my_model.pth'
EPOCH_NUM = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64
PLOT_LOSS_AND_ACC = True

checkpoint = "bert-base-multilingual-cased"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True
)

#  ❌ бажано щоб ви переглянули структуру нейронки, і в разі чого можливо змінили self.fc 
#     або логіку в forward
class SiameseNNBatch(nn.Module):
    def __init__(self, checkpoint):
        super(SiameseNNBatch, self).__init__()
        self.bert = AutoModel.from_pretrained(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.fc = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 64))
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

        bert_outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        # Get last hidden state from Bert output
        hidden_state = bert_outputs.last_hidden_state

        # Get indeces of tokens of target word in sentence 1
        target_token_ids_sen_1 = torch.where(target_1_mask == 1)[0]
        # Get indeces of tokens of target word in sentence 2
        target_token_ids_sen_2 = torch.where(target_2_mask == 1)[0]

        # Extract hidden vectors of target's tokens for each sentence
        target_hidden_vectors_sen_1 = hidden_state[:, target_token_ids_sen_1, :]
        target_hidden_vectors_sen_2 = hidden_state[:, target_token_ids_sen_2, :]

        # Avarage those hidden vectors for each sentence
        average_target_sen_1 = torch.mean(target_hidden_vectors_sen_1, dim=1)
        average_target_sen_2 = torch.mean(target_hidden_vectors_sen_2, dim=1)

        # Pass through the fully connected layers
        decreased_target_sen_1 = self.fc(average_target_sen_1)
        decreased_target_sen_2 = self.fc(average_target_sen_2)

        # Нормалізуйте вектори (потрібно для cosine_similarity)
        normalized_sen_1 = F.normalize(decreased_target_sen_1, p=2, dim=-1)
        normalized_sen_2 = F.normalize(decreased_target_sen_2, p=2, dim=-1)

        # Обчисліть cosine similarity
        cosine_sim = F.cosine_similarity(normalized_sen_1, normalized_sen_2, dim=-1)
        return cosine_sim


def my_collate_fn(features, return_tensors="pt"):
    def select_columns(xs, columns):
        result = []
        for x in xs:
            result.append({k: x[k] for k in columns})
        return result

    def make_mask(batch, feature):
        mask = torch.zeros_like(batch['input_ids'])
        for i, xs in enumerate(features):
            mask[i][xs[feature]] = 1
        return mask

    inputs = select_columns(features, ['input_ids', 'attention_mask', 'token_type_ids'])
    # labels = select_columns(features, ['sent1_target_tokens_indexes', 'sent2_target_tokens_indexes'])
    batch_inputs = data_collator(inputs)
    batch_inputs['target_1_mask'] = make_mask(batch_inputs, 'sent1_target_tokens_indexes')
    batch_inputs['target_2_mask'] = make_mask(batch_inputs, 'sent2_target_tokens_indexes')

    # Виділення міток з функції
    labels = [float(x['label']) for x in features]

    # Додавання міток до окремого словника даних
    batch_labels = {'labels': torch.tensor(labels)}

    # Повернення окремих пакунків даних для бачів та міток
    return batch_inputs, batch_labels


#  ❌ бажано щоб ви переглянули тренування, чи цикл правильно написаний
def train(model, train_dataloader, device, optimizer, criterion, num_epochs):
    # Lists to store loss and accuracy values
    train_losses = []
    train_accuracies = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Iterate over the training dataset
        for batch, batch_labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        # for batch, batch_labels in train_dataloader:
            # Move batch data and labels to GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_labels = {k: v.to(device) for k, v in batch_labels.items()}

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch)
            # print(outputs)

            # Calculate the loss
            loss = criterion(outputs, batch_labels["labels"])  # Assuming you have labels in your batch

            # Backward pass
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Update the running loss
            running_loss += loss.item()

            # Calculate the number of correct predictions for accuracy
            # TODO: Outputs are float numbers that will never equal label (0 or 1) exactly.
            correct_predictions += (outputs == batch_labels["labels"]).sum().item()
            total_samples += len(batch_labels)

        # Calculate average loss and accuracy for this epoch
        epoch_loss = running_loss / len(train_dataloader)
        epoch_accuracy = correct_predictions / total_samples

        # Append loss and accuracy values to lists
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        # Print the average loss and accuracy for this epoch
        print(f"Epoch {epoch+1}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}")

    return model, train_losses, train_accuracies


def plot_loss_accuracy(train_losses, train_accuracies):
    # Plot loss and accuracy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.show()

def main():
    os.makedir(os.path.dirname(FILE_PATH_TO_SAVE_MODEL), exist_ok=True)

    # Load the dataset
    targets_datasets = load_dataset("json", data_files=data_files, )

    batch_size = BATCH_SIZE # 64 ⛔️ colab: OutOfMemoryError: CUDA out of memory.

    train_dataloader = DataLoader(
        targets_datasets["train"], shuffle=True, batch_size=batch_size, collate_fn=my_collate_fn,
    )
    eval_dataloader = DataLoader(
        targets_datasets["test"], batch_size=batch_size, collate_fn=my_collate_fn
    )

    model = SiameseNNBatch(checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move model to GPU
    model.to(device)

    # Define your loss function
    criterion = nn.MSELoss()

    # Define your optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training ...
    model, train_losses, train_accuracies = train(model, train_dataloader, device, optimizer, criterion, num_epochs=EPOCH_NUM)
    
    # Save model and additional training information
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'train_accuracies': train_accuracies
    }, FILE_PATH_TO_SAVE_MODEL)

    # SAVE MODEL
    # torch.save(model.state_dict(), FILE_PATH_TO_SAVE_MODEL)

    if PLOT_LOSS_AND_ACC:
        plot_loss_accuracy(train_losses, train_accuracies)

    # Evaluate the model on test data
    eval_loss, eval_accuracy = evaluate_model(model, eval_dataloader, device, criterion)
    
    print("Evaluation Loss:", eval_loss)
    print("Evaluation Accuracy:", eval_accuracy)

if __name__ == "__main__":
    main()


