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
import random
from datasets import Dataset
import torch.optim.lr_scheduler as lr_scheduler_torch
from torch.utils.data import WeightedRandomSampler

DATASET_NAMES = [
    "targets_df_train_bert_together.jsonl",
    "targets_df_test_bert_together.jsonl"
]

FILE_PATH_TRAIN_DF = f"./data/input_dfs/{DATASET_NAMES[0]}"
FILE_PATH_TEST_DF = f"./data/input_dfs/{DATASET_NAMES[1]}"

# FILE_PATH_TRAIN_DF = f"/Users/yurayano/PycharmProjects/wsd/models/data_train_dedup/{DATASET_NAMES[0]}"
# FILE_PATH_TEST_DF = f"/Users/yurayano/PycharmProjects/wsd/models/data_train_dedup/{DATASET_NAMES[1]}"


FILE_PATH_TO_SAVE_MODEL = "./weights/model_bert_no_tags_together_contras.pth"
CHECKPOINT_MODEL_PATH = "./weights/model_bert_no_tags_together_contras_checkpoint.pth" 


data_files = {
    "train": FILE_PATH_TRAIN_DF,
    "test": FILE_PATH_TEST_DF,
}

# TRAINING PARAMS 
EPOCH_NUM = 10           
BATCH_SIZE = 64  
EVALUTION_FREQ = 400  # eval on valid data every 400 batches
WARMUP_BATCH_STEPS = 2000 # Number of steps for warmup
PLOT_LOSS_AND_ACC = True
MIN_LR = 5e-9
INITIAL_LR = 5e-3 # for contrastive loss!
DIST_METRIC = 'euclid' # euclid + contrastive loss
PATIENCE = 3
FACTOR = 0.2

ACCURACY_TRESHOLD = 0.5 
ACCURACY_TRESHOLD_MARGIN = 1.0


PROJ_TARGET_LEN = 64 # not used

checkpoint = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=ACCURACY_TRESHOLD_MARGIN):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss
    
def find_euclid_dist(x0, x1):
    diff = x0 - x1
    dist_sq = torch.sum(torch.pow(diff, 2), 1)
    return torch.sqrt(dist_sq)


class SiameseNNBatch(nn.Module):
    def __init__(self, checkpoint, proj_target_len=PROJ_TARGET_LEN, only_head=False, simple_head=1):
        # only_head:
        # -1 = all
        # 1 = head
        # n = last 'n' bert layers + head    (n > 1)
        
        super(SiameseNNBatch, self).__init__()
        self.only_head = only_head
    
        self.bert = AutoModel.from_pretrained(checkpoint)
             
        if simple_head == 1:
            self.head = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, proj_target_len),
            ) 
        elif simple_head == 0:
            self.head = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, 512),
                nn.LeakyReLU(),
#                 nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.LeakyReLU(),
#                 nn.Dropout(0.1),
                nn.Linear(256, proj_target_len),
            )
        else:
            self.head = None

    def parameters(self, recurse: bool = True):
        if self.only_head == 1:
            self.freeze_first_n(12)
            print("Only HEAD")
            return self.head.parameters(recurse=recurse)
        elif self.only_head == -1:
            print("All BERT layers + HEAD")
            return super(SiameseNNBatch, self).parameters(recurse)
        else:
            first_n_to_freeze = self.only_head
            self.freeze_first_n(first_n_to_freeze)
#             modules = [*self.bert.encoder.layer[first_n_to_freeze:], self.head]
#             return (module.parameters() for module in modules)
            return super(SiameseNNBatch, self).parameters(recurse)

        
    def freeze_first_n(self, n):
        modules = [self.bert.embeddings, *self.bert.encoder.layer[:n]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        
    def forward(self, batch):
        input_ids = batch["input_ids"]
        token_type_ids = batch["token_type_ids"]
        attention_mask = batch["attention_mask"]

        # Mask that tells indeces of tokens of target word in a sentence:
        # [0, 0, 0, 1, 1, 0, 0] -> [3, 4]
        target_1_mask = batch["target_1_mask"]
        target_2_mask = batch["target_2_mask"]

        bert_outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        # Get last hidden state from Bert output
        hidden_state = bert_outputs.last_hidden_state                   # (batch_size, seq_len, hidden_size)

        # Compute average hidden state of target word in each sentence
        sum_target_1 = (target_1_mask.unsqueeze(-1) * hidden_state).sum(dim=1)  # (batch_size, hidden_size)
        sum_target_2 = (target_2_mask.unsqueeze(-1) * hidden_state).sum(dim=1)  # (batch_size, hidden_size)
        avg_target_1 = sum_target_1 / target_1_mask.sum(dim=1).unsqueeze(-1)    # (batch_size, hidden_size)
        avg_target_2 = sum_target_2 / target_2_mask.sum(dim=1).unsqueeze(-1)    # (batch_size, hidden_size)

        # Pass through the fully connected layers
        if self.head:
            proj_target_1 = self.head(avg_target_1)
            proj_target_2 = self.head(avg_target_2)
        else:
            proj_target_1 = avg_target_1
            proj_target_2 = avg_target_2
            
        return proj_target_1, proj_target_2


def calc_correct_predictions(outputs, batch_labels, dist='cosine'):
    proj_target_1, proj_target_2 = outputs
    #  Calculate the number of correct predictions for accuracy
    if dist == 'cosine':
        cosine_sim = F.cosine_similarity(proj_target_1, proj_target_2, dim=-1)
        correct_predictions_i = (
            (cosine_sim > ACCURACY_TRESHOLD).float() == batch_labels["labels"])\
        .sum().item() 
    elif dist == 'euclid':
        euclid_dists = find_euclid_dist(proj_target_1, proj_target_2)
        # (MY)
        correct_predictions_i = (
        (euclid_dists < ACCURACY_TRESHOLD_MARGIN+1).float() == batch_labels["labels"]
        ).sum().item()
    return correct_predictions_i


class EarlyStopping:
    def __init__(self, patience=20, verbose=True,verbose_save=False, delta=0, path='checkpoint.pth'):
        """
        Args:
            patience (int): How long to wait after last time the validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
            path (str): Path for the checkpoint to be saved to.
                        Default: 'checkpoint.pth'
        """
        self.patience = patience
        self.verbose = verbose
        self.verbose_save = verbose_save
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.test_loss_min = float('inf')

    def __call__(self, test_loss, model, data):
        score = -test_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(test_loss, model, data)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"THERE WAS ALREADY {self.patience} iterations with NO test_loss DROP!")
        else:
            self.best_score = score
            self.save_checkpoint(test_loss, model, data)
            self.counter = 0

    def save_checkpoint(self, test_loss, model, data):
        '''Saves model when test loss decrease.'''
        if self.verbose_save:
            print(f'Test loss decreased ({self.test_loss_min:.6f} --> {test_loss:.6f}).  Saving model ...')
        # Save model
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "train_losses": data['train_losses'],
                "train_accuracies": data['train_accuracies'],
                "test_losses": data['test_losses'],
                "test_accuracies": data['test_accuracies'],
                "inter_valid_losses":data['inter_valid_losses'],
                "inter_valid_accuracies": data['inter_valid_accuracies'],
            },
            self.path,
        )
        self.test_loss_min = test_loss


def train(model, train_dataloader, device, optimizer, criterion, num_epochs,
          patience,
          factor, # 0.1
          warmup_batch_steps, # бажано train_size/batch_size*N (first N% of train data)
          min_lr,
          initial_lr,        # target_lr == initial_lr (основний lr)
          dist_metric,
          path_early_model_save,
          eval_freq = EVALUTION_FREQ,
          use_reduce_lr_sched=True,
          use_warmup_lr_sched=False,
          eval_dataloader=None,
          test_dataloader=None
         ): 

    # Lists to store loss and accuracy values
    train_losses = []
    train_accuracies = []

    test_losses = []
    test_accuracies = []
    
    inter_valid_losses = []
    inter_valid_accuracies = []

    batches_since_eval = 0
    
    # Create learning rate scheduler with warmup
    if use_warmup_lr_sched:
        warmup_scheduler = lr_scheduler_torch.LambdaLR(optimizer,
                                    lambda current_batch_step: min_lr + (initial_lr - min_lr)\
                                    * current_batch_step / warmup_batch_steps)

    if use_reduce_lr_sched:
        reduce_lr_scheduler = lr_scheduler_torch.ReduceLROnPlateau(optimizer,
                                                            mode='min',
                                                            factor=factor, # new_lr = lr * factor
                                                            patience=patience,
                                                            min_lr=min_lr)

    is_warmup_finished = False   
        
    #     Якщо valid_loss майже дорівнює 0.25, і різниця між новим valid_loss та попереднім велика (більше порогу), то швидкість навчання буде зменшуватися
    #     Параметр threshold визначає поріг для визначення, наскільки велика повинна бути зміна метрики, щоб спрацювало зменшення швидкості навчання.
    #     За замовчуванням threshold встановлено на 0.0001, що означає, що зміни метрики менші за цей поріг будуть вважатися незначними, і швидкість навчання не буде змінюватися.
        
    early_stopping = EarlyStopping(patience=3, path=path_early_model_save)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        batches_since_epoch = 0

        valid_losses_per_current_epoch = []
        valid_acc_per_current_epoch = []
         
        # Iterate over the training dataset
        for batch, batch_labels in tqdm(
            train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"
        ):
            optimizer.zero_grad()
            
            # Move batch data and labels to GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_labels = {k: v.to(device) for k, v in batch_labels.items()}
            
            outputs = model(batch)
            
            proj_target_1, proj_target_2 = outputs
            
            # contras + cosine
            if dist_metric == 'cosine': 
                cosine_sim = F.cosine_similarity(proj_target_1, proj_target_2, dim=-1)
                loss = criterion(cosine_sim, batch_labels["labels"])
            # ContrastiveLoss + euclid
            else:  
                loss = criterion(proj_target_1, proj_target_2, batch_labels["labels"])
                
            loss.backward()
            
            optimizer.step()
        
            # Update learning rate based on scheduler
            # ✅ Ostap: "Вони мають ходити парою (optimizer.step & lr_sched.step)" 
            # 🔥DOCS: Note that step should be called after validate()
            # 🤔 переношу вниз, щоб робити крок після обрахунку val_loss,
            # 🤔 тобто передаю val_loss в .step()
#             reduce_lr_scheduler.step()

            running_loss += loss.item()
        
            correct_predictions += calc_correct_predictions(outputs, batch_labels, dist=dist_metric)
            
            total_samples += len(batch_labels["labels"])

                
            # Evaluate the model every EVALUTION_FREQ batches
            batches_since_eval += 1
            batches_since_epoch += 1
            if eval_dataloader is not None and batches_since_eval >= eval_freq:
                epoch_loss = running_loss / batches_since_epoch
                epoch_accuracy = correct_predictions / total_samples
                eval_loss, eval_accuracy = evaluate_model(model,
                                                          device,
                                                          eval_dataloader,
                                                          criterion,
                                                          dist_metric=dist_metric)
                
                # 🤔 переніс сюди
                if use_reduce_lr_sched and is_warmup_finished:
                    reduce_lr_scheduler.step(eval_loss)
#                 current_lr = reduce_lr_scheduler.get_last_lr()
#                 reduce_lr_scheduler.print_lr(is_verbose, group, lr, epoch=None)
                
                print(f"\nEpoch [{epoch+1}], Train_Loss: {epoch_loss:.5f}, Train_Accuracy: {epoch_accuracy:.2f}, "
                      f"Valid_Loss: {eval_loss:.5f}, Valid_Accuracy: {eval_accuracy:.2f}\n")
                   
                batches_since_eval = 0

                # inter_valid_losses.append(eval_loss)
                valid_losses_per_current_epoch.append(eval_loss)

                # inter_valid_accuracies.append(eval_accuracy)
                valid_acc_per_current_epoch.append(eval_loss)

    
            # Perform linear warmup
            if not is_warmup_finished and (batches_since_epoch < warmup_batch_steps):
                warmup_scheduler.step()
            else:
                is_warmup_finished = True
                
        # Calculate average loss and accuracy for this epoch
        epoch_loss = running_loss / len(train_dataloader)
        epoch_accuracy = correct_predictions / total_samples

        # Append loss and accuracy values to lists
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        # Compute the evaluation loss and accuracy
        eval_loss, eval_accuracy = evaluate_model(model,
                                                  device,
                                                  test_dataloader,
                                                  criterion,
                                                  dist_metric=dist_metric)

        # Append loss and accuracy values to lists
        test_losses.append(eval_loss)
        test_accuracies.append(eval_accuracy)

        inter_valid_losses.append(valid_losses_per_current_epoch)
        inter_valid_accuracies.append(valid_acc_per_current_epoch)

        # Print the average loss and accuracy for this epoch
        print(f"\nEpoch {epoch+1}, Train_Loss: {epoch_loss:.5f}, Train_Accuracy: {epoch_accuracy:.2f}, "
              f"Test_Loss: {eval_loss:.5f}, Test_Accuracy: {eval_accuracy:.2f} \n")

        # based on TEST loss !!!!
        data =  {
                "train_losses": train_losses,
                "train_accuracies": train_accuracies,
                "test_losses": test_losses,
                "test_accuracies": test_accuracies,
                "inter_valid_losses":inter_valid_losses,
                "inter_valid_accuracies": inter_valid_accuracies,
        }
        early_stopping(eval_loss, model, data)



    return model, train_losses, train_accuracies, test_losses, test_accuracies, inter_valid_losses, inter_valid_accuracies



def evaluate_model(model, device, eval_dataloader, criterion, dist_metric='cosine'):
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
            proj_target_1, proj_target_2 = outputs
            
            # contras + cosine
            if dist_metric == 'cosine': 
                cosine_sim = F.cosine_similarity(proj_target_1, proj_target_2, dim=-1)
                loss = criterion(cosine_sim, batch_labels["labels"])
            # ContrastiveLoss + euclid
            else:  
                loss = criterion(proj_target_1,proj_target_2, batch_labels["labels"]) 
                
            running_loss += loss.item()

            # Calculate the number of correct predictions for accuracy
            correct_predictions += calc_correct_predictions(outputs, batch_labels, dist=dist_metric)

            total_samples += len(batch_labels["labels"])

    # Calculate average loss and accuracy
    eval_loss = running_loss / len(eval_dataloader)
    eval_accuracy = correct_predictions / total_samples

    return eval_loss, eval_accuracy



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


def get_balanced_sample2(df_input, samples_num=10000):
    df_label_1 = []
    df_label_0 = []

    ones_count = 0
    zeros_count = 0
    
    rest_ex = []
    
    df = df_input.shuffle(seed=42)  # You can set a seed for reproducibility
    
    for i, ex in (enumerate(df)):
        if ones_count+zeros_count != samples_num:          
            if ex["label"].item() == 1 and ones_count<int(samples_num/2):
                df_label_1.append(ex)
                ones_count+=1
                continue
            elif ex["label"].item() == 0 and zeros_count<int(samples_num/2):
                df_label_0.append(ex)
                zeros_count+=1
                continue
        rest_ex.append(ex)

    # Merge the lists
    merged_examples = df_label_0 + df_label_1

    # Shuffle the merged list
    random.shuffle(merged_examples)

    # Convert lists to dictionaries
    dict_labels = {key: [ex[key] for ex in merged_examples] for key in merged_examples[0].keys()}

    # Convert dictionaries to datasets
    df_sample_balanced = Dataset.from_dict(dict_labels)

    # Convert lists to dictionaries 2
    dict_labels_rest = {key: [ex[key] for ex in rest_ex] for key in rest_ex[0].keys()}

    # Convert dictionaries to datasets 2
    df_sample_balanced_rest = Dataset.from_dict(dict_labels_rest)

    if(df_sample_balanced.filter(lambda example: example['label'] == 0).num_rows != int(samples_num/2)):
        print("ERROR")

    df_sample_balanced.set_format("torch")
    df_sample_balanced_rest.set_format("torch")
    return df_sample_balanced, df_sample_balanced_rest



def main():
    no_tags_datasets = load_dataset(
        "json",
        data_files=data_files,
    )

    no_tags_datasets.set_format("torch")

    eval_size = int(no_tags_datasets["test"].num_rows*0.355)+1 # = 25048 |70557 - 25048 = 45509|
    df_valid_sample_balanced, df_test_sample_balanced = get_balanced_sample2(no_tags_datasets["test"],
                                                                              eval_size)


    train_dataloader = DataLoader(
        no_tags_datasets["train"],
            #  df_train_sample_balanced,
        #     no_tags_datasets["train10k"],
            shuffle=True,
            batch_size=BATCH_SIZE,
            collate_fn=my_collate_fn,
    )
    eval_dataloader = DataLoader(
        #     no_tags_datasets["valid"],
            df_valid_sample_balanced,
        #     shuffled_no_tags_datasets_val,
            batch_size=BATCH_SIZE, 
            collate_fn=my_collate_fn
    )
    test_dataloader = DataLoader(
        #     no_tags_datasets["valid"],
            df_test_sample_balanced,
        #     shuffled_no_tags_datasets_test,
            batch_size=BATCH_SIZE, 
            collate_fn=my_collate_fn
    )

    model = SiameseNNBatch(checkpoint, simple_head=-1, only_head=-1)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Кількість параметрів: {num_params}")  # 178394944

    # for name, param in model.named_parameters():
    #     print(name)
    #     print(param.requires_grad)  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move model to GPU
    model.to(device)

    # Define your loss function
    criterion = ContrastiveLoss()

    # Define your optimizer
    optimizer = optim.Adam(model.parameters())

    # Training ...
    model, train_losses, train_accuracies, test_losses, test_accuracies, inter_valid_losses, inter_valid_accuracies = train(
        model,
        train_dataloader,
        device,
        optimizer,
        criterion,
        eval_dataloader=eval_dataloader,
        test_dataloader=test_dataloader,
        use_reduce_lr_sched=True,
        use_warmup_lr_sched=True,
        num_epochs=EPOCH_NUM, 
        warmup_batch_steps = WARMUP_BATCH_STEPS,
        min_lr = MIN_LR,
        initial_lr = INITIAL_LR,  # Target learning rate (learning rate after warmup),
        dist_metric = DIST_METRIC,
        patience=PATIENCE,
        factor=FACTOR,
        eval_freq = EVALUTION_FREQ,
        path_early_model_save=CHECKPOINT_MODEL_PATH
    )

    # if PLOT_LOSS_AND_ACC:
    #     plot_loss_accuracy(train_losses, train_accuracies, "Train", "Bert (no tags) (together)")
    #     plot_loss_accuracy(test_losses, test_accuracies, "Test", "Bert (no tags) (together)")
        # plot_loss_accuracy(inter_valid_losses, inter_valid_accuracies, "Valid inter", "Bert (no tags) (together)")

     # Save model and additional training information
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "test_losses": test_losses,
            "test_accuracies": test_accuracies,
            "inter_valid_losses":inter_valid_losses,
            "inter_valid_accuracies":inter_valid_accuracies,
        },
        FILE_PATH_TO_SAVE_MODEL,
    )

if __name__ == "__main__":
    main()