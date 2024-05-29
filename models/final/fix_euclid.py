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
import matplotlib.pyplot as plt
import numpy as np

DATASET_NAMES = [
    "targets_df_train_bert_together.jsonl",
    "targets_df_test_bert_together.jsonl",
    "df_train_10k_specific.jsonl"
]

FILE_PATH_TRAIN_DF = f"/Users/yurayano/PycharmProjects/wsd/models/data_train_dedup/{DATASET_NAMES[0]}"
FILE_PATH_TEST_DF = f"/Users/yurayano/PycharmProjects/wsd/models/data_train_dedup/{DATASET_NAMES[1]}"
FILE_PATH_TRAIN_10K_DF = f"/Users/yurayano/PycharmProjects/wsd/models/data_train_dedup/{DATASET_NAMES[2]}"


# FILE_PATH_TRAIN_DF = "/kaggle/input/bert-together/targets_df_train_bert_together.jsonl"
# FILE_PATH_TEST_DF = "/kaggle/input/bert-together/targets_df_test_bert_together.jsonl"
# FILE_PATH_TRAIN_10K_DF = "/kaggle/input/bert-together/df_train_10k_specific.jsonl"

FILE_PATH_TO_SAVE_MODEL = "/kaggle/working"

data_files = {
    "train": FILE_PATH_TRAIN_DF,
    "test": FILE_PATH_TEST_DF,
    "train10k": FILE_PATH_TRAIN_10K_DF,
}

# TRAINING PARAMS
EPOCH_NUM = 5           # PS: ⛔️⛔️⛔️ Можливо варто збільшити...
# LEARNING_RATE = 0.00005  # PS: ⛔️⛔️⛔️ Можливо варто збільшити...
BATCH_SIZE = 64          # PS: ⛔️⛔️⛔️  Варто збільшити
PLOT_LOSS_AND_ACC = True
# EVALUTION_FREQ = 100  # batches
EVALUTION_FREQ = 100  # batches


ACCURACY_TRESHOLD = 0.5
ACCURACY_TRESHOLD_MARGIN = 1.0

PROJ_TARGET_LEN = 64     # PS: ⛔️⛔️⛔️ Можливо варто збільшити довжину векторів на виході з self.fc

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
    def __init__(self, checkpoint, proj_target_len=PROJ_TARGET_LEN, only_head=False, simple_head=1, use_lora=False):
        # only_head:
        # -1 = all
        # 1 = head
        # n = last 'n' bert layers + head    (n > 1)

        super(SiameseNNBatch, self).__init__()
        self.only_head = only_head

        self.bert = AutoModel.from_pretrained(checkpoint)

        #         if use_lora:
        #             # Applying LoRA to the BERT layers
        #             lora_config = LoraConfig(
        #                 r=1, lora_alpha=1, lora_dropout=0.1, target_modules=["query", "value"]
        #             )
        #             self.bert = apply_lora(self.bert, lora_config)

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
        hidden_state = bert_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        # Compute average hidden state of target word in each sentence
        sum_target_1 = (target_1_mask.unsqueeze(-1) * hidden_state).sum(dim=1)  # (batch_size, hidden_size)
        sum_target_2 = (target_2_mask.unsqueeze(-1) * hidden_state).sum(dim=1)  # (batch_size, hidden_size)
        avg_target_1 = sum_target_1 / target_1_mask.sum(dim=1).unsqueeze(-1)  # (batch_size, hidden_size)
        avg_target_2 = sum_target_2 / target_2_mask.sum(dim=1).unsqueeze(-1)  # (batch_size, hidden_size)

        # Pass through the fully connected layers
        if self.head:
            proj_target_1 = self.head(avg_target_1)
            proj_target_2 = self.head(avg_target_2)
        else:
            proj_target_1 = avg_target_1
            proj_target_2 = avg_target_2

        return proj_target_1, proj_target_2


def calculate_statistics(tensor):
    statistics = {
        'min': torch.min(tensor).item(),
        'max': torch.max(tensor).item(),
        'std': torch.std(tensor).item(),
        'mean': torch.mean(tensor).item(),
        'median': torch.median(tensor).item()
    }
    return statistics


def calc_correct_predictions_hyper(outputs, batch_labels, idx, dist='cosine', threshold=0.5):
    proj_target_1, proj_target_2 = outputs
    if dist == 'cosine':
        cosine_sim = F.cosine_similarity(proj_target_1, proj_target_2, dim=-1)
        correct_predictions_i = ((cosine_sim > ACCURACY_TRESHOLD).float() == batch_labels["labels"]).sum().item()
    elif dist == 'euclid':
        euclid_dists = find_euclid_dist(proj_target_1, proj_target_2)
#         print(euclid_dists)
#         stats = calculate_statistics(euclid_dists)
#         mean_threshold = stats['mean']
        median_threshold = torch.median(euclid_dists).item()
        correct_predictions_i = ((euclid_dists < median_threshold).float() == batch_labels["labels"]).sum().item()
        num_positives = torch.sum(batch_labels["labels"] == 1).item()
        total = len(batch_labels["labels"])
        print(f"{num_positives} {total-num_positives}")
        if idx == 0:
            print(f"---- Num of correct predictions: {correct_predictions_i} -----")

            stats = calculate_statistics(euclid_dists)
            print(f" Num of positives: {num_positives}")
            print(f" Num of negatives: {total-num_positives}")
            print(euclid_dists)
            print(stats)
            plot_sorted_distances(euclid_dists, batch_labels["labels"])
            print(" -----------------------------------------")
    return correct_predictions_i


def plot_sorted_distances(euclid_dists, batch_labels):
    """
    Функція будує стовпчастий графік з відсортованими евклідовими відстанями та відповідними бінарними мітками.

    Параметри:
    euclid_dists (torch.Tensor): Вектор евклідових відстаней.
    batch_labels (torch.Tensor): Відповідні бінарні мітки (0 або 1).

    Повертає:
    Нічого.
    """
    # Переміщення тензорів на CPU та конвертація в numpy
    euclid_dists = euclid_dists.cpu().numpy()
    batch_labels = batch_labels.cpu().numpy()

    # Сортування відстаней та міток
    sorted_indices = np.argsort(euclid_dists)
    sorted_dists = euclid_dists[sorted_indices]
    sorted_labels = batch_labels[sorted_indices]

    # Підготовка кольорів для кожного стовпчика
    colors = ['green' if label == 1 else 'red' for label in sorted_labels]

    # Побудова стовпчастого графіка
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(sorted_dists)), sorted_dists, color=colors)
    plt.xlabel('Index')
    plt.ylabel('Euclidean Distance')
    plt.title('Sorted Euclidean Distances with Binary Labels')
    plt.show()

# Приклад використання
# euclid_dists = np.random.rand(100)  # Замініть на ваші евклідові відстані
# batch_labels = np.random.randint(0, 2, 100)  # Замініть на ваші бінарні мітки

# plot_sorted_distances(euclid_dists, batch_labels)


def evaluate_model_hyper(model, device, eval_dataloader, criterion, threshold,
                         dist_metric='cosine'):
    if eval_dataloader is None:
        return None, None

    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        #         for batch, batch_labels in eval_dataloader:
        for idx, (batch, batch_labels) in enumerate(eval_dataloader):
            # Move batch data and labels to GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_labels = {k: v.to(device) for k, v in batch_labels.items()}

            outputs = model(batch)
            proj_target_1, proj_target_2 = outputs

            # MSE + cosine
            if dist_metric == 'cosine':
                cosine_sim = F.cosine_similarity(proj_target_1, proj_target_2, dim=-1)
                loss = criterion(cosine_sim, batch_labels["labels"])
            # ContrastiveLoss + euclid
            else:
                loss = criterion(proj_target_1, proj_target_2, batch_labels["labels"])

            running_loss += loss.item()

            # Calculate the number of correct predictions for accuracy
            correct_predictions += calc_correct_predictions_hyper(outputs,
                                                                  batch_labels,
                                                                  dist=dist_metric,
                                                                  #                                                                   threshold=threshold,
                                                                  idx=idx,
                                                                  )

            total_samples += len(batch_labels["labels"])

    # Calculate average loss and accuracy
    eval_loss = running_loss / len(eval_dataloader)
    eval_accuracy = correct_predictions / total_samples

    return eval_loss, eval_accuracy


def train_hyper(model, train_dataloader, device, optimizer, criterion, num_epochs,
                eval_dataloader=None,
                test_dataloader=None,
                use_reduce_lr_sched=True,
                use_warmup_lr_sched=False,
                patience=1,
                factor=0.2,  # 0.1
                warmup_batch_steps=100,  # бажано train_size/batch_size*N (first N% of train data)
                min_lr=1e-10,
                initial_lr=1e-4,  # target_lr == initial_lr (основний lr)
                dist_metric='cosine',
                eval_freq=EVALUTION_FREQ
                ):
    # Lists to store loss and accuracy values
    train_losses = []
    train_accuracies = []

    eval_losses = []
    eval_accuracies = []

    inter_eval_losses = []
    inter_eval_accuracies = []

    batches_since_eval = 0

    # Create learning rate scheduler with warmup
    if use_warmup_lr_sched:
        warmup_scheduler = lr_scheduler_torch.LambdaLR(optimizer,
                                                       lambda current_batch_step: min_lr + (initial_lr - min_lr) \
                                                                                  * current_batch_step / warmup_batch_steps)

    if use_reduce_lr_sched:
        reduce_lr_scheduler = lr_scheduler_torch.ReduceLROnPlateau(optimizer,
                                                                   mode='min',
                                                                   factor=factor,  # new_lr = lr * factor
                                                                   patience=patience,
                                                                   min_lr=min_lr)

    is_warmup_finished = False

    #     Якщо valid_loss майже дорівнює 0.25, і різниця між новим valid_loss та попереднім велика (більше порогу), то швидкість навчання буде зменшуватися
    #     Параметр threshold визначає поріг для визначення, наскільки велика повинна бути зміна метрики, щоб спрацювало зменшення швидкості навчання.
    #     За замовчуванням threshold встановлено на 0.0001, що означає, що зміни метрики менші за цей поріг будуть вважатися незначними, і швидкість навчання не буде змінюватися.

    best_thresholds = []
    best_threshold = 0
    best_threshold_accuracy = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        batches_since_epoch = 0

        # Iterate over the training dataset
        for batch, batch_labels in tqdm(
                train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        ):
            optimizer.zero_grad()

            # Move batch data and labels to GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_labels = {k: v.to(device) for k, v in batch_labels.items()}

            outputs = model(batch)

            proj_target_1, proj_target_2 = outputs

            # MSE + cosine
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

            #             correct_predictions += calc_correct_predictions(outputs, batch_labels, dist=dist_metric, threshold=best_threshold)

            total_samples += len(batch_labels["labels"])

            # Evaluate the model every EVALUTION_FREQ batches
            batches_since_eval += 1
            batches_since_epoch += 1
            if eval_dataloader is not None and batches_since_eval >= eval_freq:

                #                 best_threshold, best_threshold_accuracy = find_optimal_threshold(model,
                #                                                                                  device,
                #                                                                                  eval_dataloader,
                #                                                                                  dist_metric)

                best_threshold = -13498
                #                 print(f"best_threshold: [{best_threshold}]")
                #                 print(f"best_threshold_acc: [{best_threshold_accuracy}]")

                epoch_loss = running_loss / batches_since_epoch
                epoch_accuracy = correct_predictions / total_samples

                eval_loss, eval_accuracy = evaluate_model_hyper(model,
                                                                device,
                                                                eval_dataloader,
                                                                criterion,
                                                                dist_metric=dist_metric,
                                                                threshold=best_threshold)

                # 🤔 переніс сюди
                if use_reduce_lr_sched and is_warmup_finished:
                    reduce_lr_scheduler.step(eval_loss)
                #                 current_lr = reduce_lr_scheduler.get_last_lr()
                #                 reduce_lr_scheduler.print_lr(is_verbose, group, lr, epoch=None)

                print(f"\nEpoch [{epoch + 1}], Train_Loss: {epoch_loss:.5f}, Train_Accuracy: {epoch_accuracy:.2f}, "
                      f"✅Valid_Loss: {eval_loss:.5f}, Valid_Accuracy: {eval_accuracy:.2f}\n")

                batches_since_eval = 0
                inter_eval_losses.append(eval_loss)
                inter_eval_accuracies.append(eval_accuracy)

            correct_predictions += calc_correct_predictions_hyper(outputs,
                                                                  batch_labels,
                                                                  dist=dist_metric,
                                                                  threshold=best_threshold,
                                                                  idx=-1)

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
        eval_loss, eval_accuracy = evaluate_model_hyper(model,
                                                        device,
                                                        test_dataloader,
                                                        criterion,
                                                        dist_metric=dist_metric,
                                                        threshold=best_threshold)

        # Append loss and accuracy values to lists
        eval_losses.append(eval_loss)
        eval_accuracies.append(eval_accuracy)

        # Print the average loss and accuracy for this epoch
        print(f"\nEpoch {epoch + 1}, Train_Loss: {epoch_loss:.5f}, Train_Accuracy: {epoch_accuracy:.2f}, "
              f"Test_Loss: {eval_loss:.5f}, Test_Accuracy: {eval_accuracy:.2f} \n")

    return model, train_losses, train_accuracies, eval_losses, eval_accuracies, inter_eval_losses, inter_eval_accuracies


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


def get_balanced_sample(df, samples_num=10000):
    df_label_1 = []
    df_label_0 = []

    ones_count = 0
    zeros_count = 0
    for i, ex in (enumerate(df)):
        if ones_count+zeros_count == samples_num:
            break
        if ex["label"].item() == 1 and ones_count<int(samples_num/2):
            df_label_1.append(ex)
            ones_count+=1
        elif ex["label"].item() == 0 and zeros_count<int(samples_num/2):
            df_label_0.append(ex)
            zeros_count+=1

    # Merge the lists
    merged_examples = df_label_0 + df_label_1

    # Shuffle the merged list
    random.shuffle(merged_examples)

    # Convert lists to dictionaries
    dict_labels = {key: [ex[key] for ex in merged_examples] for key in merged_examples[0].keys()}

    # Convert dictionaries to datasets
    df_sample_balanced = Dataset.from_dict(dict_labels)


    if(df_sample_balanced.filter(lambda example: example['label'] == 0).num_rows != int(samples_num/2)):
        print("ERROR")

    df_sample_balanced.set_format("torch") # ?
    return df_sample_balanced


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def get_balanced_dataloaders(dfs, batch_size, collate_fn):
    final_dataloaders = []

    for i, df in enumerate(dfs):
        if i == 0:
            dataloader = DataLoader(
                df,
                shuffle=True,
                batch_size=batch_size,
                collate_fn=collate_fn,
            )
        else:
            labels = df["label"]
            class_counts = []
            for label in range(2):
                val_only_pn = df.filter(lambda example: example["label"] == label)
                class_counts.append(val_only_pn.num_rows)
            print(f"Class counts for dataset {i}: {class_counts}")

            #             class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
            #             sample_weights = labels.apply(lambda x: class_weights[x])

            weights = make_weights_for_balanced_classes(df["label"], 2)

            sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
            dataloader = DataLoader(
                df,
                shuffle=True,
                sampler=sampler,  # Use the generated sampler for balanced sampling
                batch_size=batch_size,
                collate_fn=collate_fn,
                pin_memory=True

            )

        final_dataloaders.append(dataloader)

    return final_dataloaders


def do_training(train_dataloader, eval_dataloader, test_dataloader):
    model = SiameseNNBatch(checkpoint, simple_head=-1, only_head=-1)

    # 178394944
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Кількість параметрів: {num_params}")  # Виводить: <кількість параметрів self.head>

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
    model, train_losses, train_accuracies, eval_losses, eval_accuracies, inter_eval_losses, inter_eval_accuracies = train_hyper(
        model,
        train_dataloader,
        device,
        optimizer,
        criterion,
        num_epochs=5,
        eval_dataloader=eval_dataloader,
        test_dataloader=test_dataloader,
        use_reduce_lr_sched=True,
        use_warmup_lr_sched=True,
        warmup_batch_steps=100,  # Number of steps for warmup
        min_lr=5e-9,
        initial_lr=5e-3,  # Target learning rate (learning rate after warmup),
        dist_metric='euclid',
    )

    # if PLOT_LOSS_AND_ACC:
    #     plot_loss_accuracy(train_losses, train_accuracies, "Train", "Bert (no tags) (together)")
    #     plot_loss_accuracy(eval_losses, eval_accuracies, "Eval", "Bert (no tags) (together)")
    #     plot_loss_accuracy(inter_eval_losses, inter_eval_accuracies, "Eval inter", "Bert (no tags) (together)")

def main():
    no_tags_datasets = load_dataset(
        "json",
        data_files=data_files,
    )

    no_tags_datasets.set_format("torch")

    df_valid_sample_balanced = get_balanced_sample(no_tags_datasets["test"].select(range(0, 20000)), 1000)
    df_test_sample_balanced = get_balanced_sample(no_tags_datasets["test"].select(range(20000,
                                                                                        len(no_tags_datasets["test"]))),3000)
    train_dataloader = DataLoader(
        #     no_tags_datasets["train"],
        #     df_train_sample_balanced,
        no_tags_datasets["train10k"],
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=my_collate_fn,
    )
    eval_dataloader = DataLoader(
        #     no_tags_datasets["valid"],
        df_valid_sample_balanced,
        shuffle=True,
        #     shuffled_no_tags_datasets_val,
        batch_size=BATCH_SIZE,
        collate_fn=my_collate_fn
    )
    test_dataloader = DataLoader(
        #     no_tags_datasets["valid"],
        df_test_sample_balanced,
        shuffle=True,
        #     shuffled_no_tags_datasets_test,
        batch_size=BATCH_SIZE,
        collate_fn=my_collate_fn
    )

    b_ex = None
    for i, batch in enumerate(eval_dataloader):
        if i > 3:
            break
        b_ex = batch

    do_training(train_dataloader, eval_dataloader, test_dataloader)


if __name__ == "__main__":
    main()