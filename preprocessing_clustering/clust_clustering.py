import json
import stats as stats_data
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering
import numpy as np

import seaborn as sns
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm




from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy  # Import hierarchy module

input_file_path = "/Users/yurayano/PycharmProjects/wsd/data/final_results/clustering_sum14_merged.json"
output_file_path = "/Users/yurayano/PycharmProjects/wsd/data/final_results/clustering_sum14_clustered.json"


model = SentenceTransformer("all-MiniLM-L6-v2")

# Two lists of sentences
# sentences1 = [
#     "The cat sits outside",
#     "A man is playing guitar",
#     "The new movie is awesome",
# ]

sentences1 = [
    "Настійне прагнення до чого-небудь, сильна внутрішня потреба робити щось, бути десь Велика зацікавленість чим-небудь",
    "Посилена, непереборна схильність до кого-небудь, настійна потреба спілкуватися з певною особою",
    "Напівсвідоме, інстинктивне прагнення до чого-небудь",
    "Те саме, що поїзд (Ряд з'єднаних між собою вагонів, що рухаються рейковими коліями з допомогою локомотива; потяг (див. потяг2).)  1",
    # "Напівсвідоме, інстинктивне прагнення до чого-небудь",
]

sentences2 = [
    "Те саме, що поїзд (Ряд з'єднаних між собою вагонів, що рухаються рейковими коліями з допомогою локомотива; потяг (див. потяг2).)  1",
    "Настійне прагнення до чого-небудь, сильна внутрішня потреба робити щось, бути десь Велика зацікавленість чим-небудь",
    "Посилена, непереборна схильність до кого-небудь, настійна потреба спілкуватися з певною особою",
    "Напівсвідоме, інстинктивне прагнення до чого-небудь",
    # "Настійне прагнення до чого-небудь, сильна внутрішня потреба робити щось, бути десь Велика зацікавленість чим-небудь",

]

def cluster_glosses(data):
    for ob in data:
        pass


def cluster_sentences(sentences, threshold):
    # Encode sentences into embeddings
    embeddings = model.encode(sentences, convert_to_tensor=True)
    
    # Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings, embeddings).cpu().numpy()  # Convert to NumPy array

    # Create Agglomerative Clustering model
    clustering_model = AgglomerativeClustering(n_clusters=None,
                                               metric='cosine',
                                               distance_threshold=threshold,
                                               linkage='average')
    
    # Fit the model
    clustering_model.fit(cosine_scores)
    
    # Get cluster labels
    cluster_labels = clustering_model.labels_
    
    # Create dictionary to store sentences for each cluster
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(sentences[i])
    
    return clusters, cosine_scores


def is_omonim(entity):
    return entity["omonim_count"] != 0


def split_data(data):
    data_omonims = []
    data_polysemous = []
    for ob in data:
        deepcopied_ob = copy.deepcopy(ob)
        if is_omonim(ob):
            data_omonims.append(deepcopied_ob)
        else:
          data_polysemous.append(deepcopied_ob)
    return data_omonims, data_polysemous


def cluster_synsets_omon(cosine_similarities, k):
    # Create AgglomerativeClustering model with desired number of clusters (k=2)
    clustering_model = AgglomerativeClustering(n_clusters=k, linkage='average')

    # Fit the model to the cosine similarities matrix
    clustering_model.fit(cosine_similarities.cpu().numpy())

    # Get cluster labels for each synset
    cluster_labels = clustering_model.labels_

    return cluster_labels


def plot_elbow_method(Ks, inertias):
    plt.plot(Ks, inertias, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method using Inertia')
    plt.show()


def find_otimal_k_with_elbow(gloss_embeddings_np, n_glosses):
    wcss_inertias_list = []
    for k in range(1, n_glosses + 1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(gloss_embeddings_np)
        wcss_iter = kmeans.inertia_
        wcss_inertias_list.append(wcss_iter)

    n_clusters_range = range(1, n_glosses + 1)

    # plot_elbow_method(n_clusters_range, wcss_inertias_list)
    # Compute the rate of decrease in WCSS

    rate_of_decrease = np.diff(wcss_inertias_list) / np.diff(range(1, n_glosses + 1))

    # Find the optimal value of k where the rate of decrease slows down significantly
    return wcss_inertias_list, np.argmin(rate_of_decrease) + 1  # Add 1 because of zero-based indexing


def remove_non_ukrainian_symbols(input_string):
    allowed_symbols = set("-' ")  # hyphen are allowed ТА АПОСТРОФ 🔥

    ukrainian_letters = set("АБВГҐДЕЄЖЗИIІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯабвгґдеєжзиiіїйклмнопрстуфхцчшщьюя")

    result_string = ''.join(char for char in input_string if char in ukrainian_letters or char in allowed_symbols)

    return result_string


def print_sentences_per_cluster(sentences, cluster_labels):
    for k in range(max(cluster_labels) + 1):
        sentences_in_cluster = [sentences[i] for i, label in enumerate(cluster_labels) if label == k]
        print(f"\nCluster {k + 1}:")
        for sentence in sentences_in_cluster:
            print(sentence)


def find_optimal_k_with_silhouette(gloss_embeddings_np, n_glosses=10):
    silhouette_scores = []
    for k in range(2, n_glosses):      # [2; n-1]
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(gloss_embeddings_np)
        silhouette_scores.append(silhouette_score(gloss_embeddings_np, kmeans.labels_))

    return silhouette_scores, silhouette_scores.index(max(silhouette_scores)) + 1
    # return silhouette_scores, 1

    # Add 1 to account for starting from k=2


def calc_n_gloss_count(data_polysemous):
    polysemous_gloss_count = {}
    for ob in data_polysemous:
        synset_count = len(ob["synsets"])
        if synset_count in polysemous_gloss_count:
            polysemous_gloss_count[synset_count] += 1
        else:
            polysemous_gloss_count[synset_count] = 1


def get_omonim_groups(omonim):
    omonim_groups = {}
    for synset in omonim["synsets"]:
        if synset["index"] in omonim_groups:
            omonim_groups[synset["index"]].append(synset["gloss"])
        else:
            omonim_groups[synset["index"]] = [synset["gloss"]]
    return omonim_groups


# def euclid_dist(x, y):
#     return np.linalg.norm(x - y, axis=1)

def euclid_dist(x, y):
    """
    Calculates the pairwise Euclidean distances between vectors in two arrays.

    Args:
        x: First NumPy array of vectors.
        y: Second NumPy array of vectors.

    Returns:
        A 2D NumPy array containing the Euclidean distances between each vector in x and each vector in y.
    """

    # Check if vectors are 1D or 2D
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=1)  # Expand 1D vectors to 2D
    if len(y.shape) == 1:
        y = np.expand_dims(y, axis=1)  # Expand 1D vectors to 2D

    # Calculate pairwise distances using nested loops
    distances = np.zeros((x.shape[0], y.shape[0]))
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            distances[i, j] = np.linalg.norm(x[i] - y[j])

    return distances

def sim_between_groups(omonim_groups):
    groups_gloss_list = []
    for index, gloss_list in omonim_groups.items():
        gloss_list = [remove_non_ukrainian_symbols(gloss) for gloss in gloss_list]
        groups_gloss_list.append(gloss_list)

    group_embedings = []
    for i in range(len(omonim_groups)):
        group_i_embeddings = model.encode(groups_gloss_list[i], convert_to_tensor=True) 
        group_embedings.append(group_i_embeddings)

    omonim_cos_sim_between_groups = []
    omonim_euclid_dist_between_groups = []

    group_cenroids = []
    for i in range(len(omonim_groups)):
        group_i_embeddings = group_embedings[i]
        centroid = np.mean(group_i_embeddings, axis=0)
        group_cenroids.append(centroid)
     
    # Calculate cosine similarities & euclidean distance between all pairs of homonymous groups
    for i in range(len(omonim_groups)):
        group_i_embeddings = group_embedings[i]
        group_i_embeddings_np = group_i_embeddings.cpu().numpy()

        for j in range(i + 1, len(omonim_groups)):
            group_j_embeddings = group_embedings[j]
            group_j_embeddings_np = group_j_embeddings.cpu().numpy()

             # не знаю як доречніше шукати відстань між групами 
             # (n глосів в групі А; m глосів в групі B; шукаю середнє усіх n*m відстаней)
            cosine_similarity = torch.mean(util.cos_sim(group_i_embeddings_np, group_j_embeddings_np))
            euclidean_distance = torch.mean(torch.FloatTensor(euclid_dist(group_i_embeddings_np, group_j_embeddings_np)))

            omonim_cos_sim_between_groups.append(cosine_similarity)
            omonim_euclid_dist_between_groups.append(euclidean_distance)

    omonim_mean_cos_sims_between_groups = torch.mean(torch.FloatTensor(omonim_cos_sim_between_groups))
    omonim_median_cos_sims_between_groups = torch.median(torch.FloatTensor(omonim_cos_sim_between_groups))
    omonim_max_cos_sims_between_groups = torch.max(torch.FloatTensor(omonim_cos_sim_between_groups))
    omonim_min_cos_sims_between_groups = torch.min(torch.FloatTensor(omonim_cos_sim_between_groups))

    omonim_mean_euclid_dist_between_groups = torch.mean(torch.FloatTensor(omonim_euclid_dist_between_groups))
    omonim_median_euclid_dist_between_groups = torch.median(torch.FloatTensor(omonim_euclid_dist_between_groups))
    omonim_max_euclid_dist_between_groups = torch.max(torch.FloatTensor(omonim_euclid_dist_between_groups))
    omonim_min_euclid_dist_between_groups = torch.min(torch.FloatTensor(omonim_euclid_dist_between_groups))

    # if omonim_mean_cos_sims_between_groups > 0.98:
    #     print("llk")

    return [
        [
            omonim_mean_cos_sims_between_groups,
            omonim_median_cos_sims_between_groups,
            omonim_max_cos_sims_between_groups,
            omonim_min_cos_sims_between_groups
        ],
        [
            omonim_mean_euclid_dist_between_groups,
            omonim_median_euclid_dist_between_groups,
            omonim_max_euclid_dist_between_groups,
            omonim_min_euclid_dist_between_groups
        ]
    ]


def inspect_omonim_similarities(data_omonims):
    omonims_mean_cos_sims_in_group = []
    omonims_median_cos_sims_in_group = []

    omonims_mean_euclid_in_group = []
    omonims_median_euclid_in_group = []

    omonim_distances_between_groups = []

    iter = 0
    for omonim in tqdm(data_omonims): # дойшов до iter 12
        iter+=1

        omonim_groups = get_omonim_groups(omonim)

        omonim_distances_between_groups.append(sim_between_groups(omonim_groups))

        for index, gloss_list in omonim_groups.items():
            num_pairs = 0
            omonim_group_cos_sims = []
            omonim_group_euclid = []

            n_glosses = len(gloss_list)
            if n_glosses <= 1:
                continue

            gloss_list = [remove_non_ukrainian_symbols(gloss) for gloss in gloss_list]

            gloss_embeddings = model.encode(gloss_list, convert_to_tensor=True)
            gloss_embeddings_np = gloss_embeddings.cpu().numpy()

            # [0][0].item()
            for i in range(len(gloss_embeddings_np)):
                for j in range(i + 1, len(gloss_embeddings_np)):
                    cosine_similarity = torch.mean(util.cos_sim(gloss_embeddings_np[i],
                                                                gloss_embeddings_np[j]))
                    euclidean_distance = torch.mean(torch.FloatTensor(
                        euclid_dist(np.expand_dims(gloss_embeddings_np[i], axis=0),
                                    np.expand_dims(gloss_embeddings_np[j], axis=0))))
                    
                    omonim_group_cos_sims.append(cosine_similarity)
                    omonim_group_euclid.append(euclidean_distance)
                    num_pairs += 1

            if num_pairs > 0:
                omonims_mean_cos_sims_in_group.append(torch.mean(torch.FloatTensor(omonim_group_cos_sims)))
                omonims_median_cos_sims_in_group.append(torch.median(torch.FloatTensor(omonim_group_cos_sims)))

                omonims_mean_euclid_in_group.append(torch.mean(torch.FloatTensor(omonim_group_euclid)))
                omonims_median_euclid_in_group.append(torch.median(torch.FloatTensor(omonim_group_euclid)))

    print(f"[IN GROUP - COS]    [MEAN]: {torch.mean(torch.FloatTensor(omonims_mean_cos_sims_in_group))}")
    print(f"[IN GROUP - COS]    [MEDIAN]: {torch.median(torch.FloatTensor(omonims_median_cos_sims_in_group))}")
    print(f"[IN GROUP - EUCLID] [MEAN]: {torch.mean(torch.FloatTensor(omonims_mean_euclid_in_group))}")
    print(f"[IN GROUP - EUCLID] [MEDIAN]: { torch.median(torch.FloatTensor(omonims_median_euclid_in_group))}")


    mean_cos_sims = []
    mean_euclid_dists = []
    for entity in omonim_distances_between_groups:
        # Extract specific metric values
        mean_cos_sims.append(entity[0])
        mean_euclid_dists.append(entity[1])

    between_cos_sims = np.mean(np.array(mean_cos_sims), axis=0)
    between_euclid_dists = np.mean(np.array(mean_euclid_dists), axis=0)


    print(f"[BETWEEN GROUPS - COS]    [MEAN]: {between_cos_sims[0]}")
    print(f"[BETWEEN GROUPS - COS]    [MEDIAN]: {between_cos_sims[1]}")
    print(f"[BETWEEN GROUPS - EUCLID] [MEAN]: {between_euclid_dists[0]}")
    print(f"[BETWEEN GROUPS - EUCLID] [MEDIAN]: {between_euclid_dists[1]}")


    sorted_cos_in_group = np.sort(omonims_median_cos_sims_in_group)
    sorted_euc_in_group = np.sort(omonims_median_euclid_in_group)
    sorted_cos_out_group = np.sort(np.array(mean_cos_sims)[:, 1])
    sorted_euc_out_group = np.sort(np.array(mean_euclid_dists)[:, 1])

    # sns.displot(data=sorted_cos_in_group, bins=15) # kde=True

    # # Створення словника для зберігання масивів
    data = {
        "sorted_cos_in_group": list(map(np.float64, sorted_cos_in_group)),
        "sorted_euc_in_group": list(map(np.float64, sorted_euc_in_group)),
        "sorted_cos_out_group": list(map(np.float64, sorted_cos_out_group)),
        "sorted_euc_out_group": list(map(np.float64, sorted_euc_out_group)),
    }

    # Збереження даних у файл JSON
    file_path = "/Users/yurayano/PycharmProjects/wsd/data/final_results/data_clust.json"
    with open(file_path, "w") as json_file:
        json.dump(data, json_file)


def main():
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        data = [json.loads(line) for line in input_file]

    data_omonims, data_polysemous = split_data(data)

    inspect_omonim_similarities(data_omonims)

    # threshold_calculated = 0.6607006788253784
    #
    # n_omonims_2_glosses = 0
    # for polysem in tqdm(data_polysemous):
    #     gloss_list = [remove_non_ukrainian_symbols(s["gloss"]) for s in polysem["synsets"]]
    #     n_glosses = len(gloss_list)
    #     if n_glosses < 3:
    #         n_omonims_2_glosses += 1
    #         continue
    #
    #     cluster_sentences(gloss_list, threshold_calculated)


    # k_omonim_list = []
    # n_omonims_2_glosses = 0
    # for iter, omonim in enumerate(data_polysemous[66:68]):
    #     # print(omonim["lemma"])
    #     gloss_list = [synset["gloss"] for synset in omonim["synsets"]]
    #
    #     gloss_list = [remove_non_ukrainian_symbols(gloss) for gloss in gloss_list]
    #     # print(gloss_list)
    #
    #     n_glosses = len(gloss_list)
    #
    #     if n_glosses < 4:
    #         n_omonims_2_glosses += 1
    #         continue
    #
    #     gloss_embeddings = model.encode(gloss_list, convert_to_tensor=True)
    #
    #     gloss_embeddings_np = gloss_embeddings.cpu().numpy()
    #
    #     wcss_inertias_list, optimal_k_elb = find_otimal_k_with_elbow(gloss_embeddings_np, n_glosses)
    #     silhouette_scores, optimal_k_sil = find_optimal_k_with_silhouette(gloss_embeddings_np, n_glosses)
    #
    #     if optimal_k_elb > 1:
    #         print(optimal_k_elb)
    #
    #     if optimal_k_sil > 1:
    #         print(optimal_k_sil)
    #
    #     # Perform k-means clustering with optimal k
    #     kmeans = KMeans(n_clusters=optimal_k_elb, init='k-means++', random_state=42)
    #     kmeans.fit(gloss_embeddings_np)
    #     cluster_labels = kmeans.labels_
    #     # Print sentences from each cluster
    #     print_sentences_per_cluster(gloss_list, cluster_labels)
    #
    #     # Perform k-means clustering with optimal k
    #     kmeans = KMeans(n_clusters=optimal_k_sil, init='k-means++', random_state=42)
    #     kmeans.fit(gloss_embeddings_np)
    #     cluster_labels = kmeans.labels_
    #     # Print sentences from each cluster
    #     print_sentences_per_cluster(gloss_list, cluster_labels)

        #     cosine_similarities = util.cos_sim(gloss_embeddings[0].unsqueeze(0),
        #                                        gloss_embeddings[1].unsqueeze(0))
        #     k_omonim_list.append(cosine_similarities[0][0].item())
        #     continue
        #
        # cosine_similarities = util.cos_sim(gloss_embeddings, gloss_embeddings)
        # cluster_labels = cluster_synsets_omon(cosine_similarities, k=omonim["omonim_count"] + 1)
        #
        # # Print the clustering results
        # for i, synset in enumerate(omonim['synsets']):
        #     print(f"{omonim['lemma']} | Gloss: {synset['gloss']} | Cluster: |👁️ {cluster_labels[i]} |✅ {synset['index']} |")



    # output_data = cluster_glosses(data)

    # with open(output_file_path, 'w', encoding='utf-8') as output_file:
    #     for entry in output_data:
    #         json.dump(entry, output_file, ensure_ascii=False)
    #         output_file.write('\n')

    #! Compute embedding for both lists
    # embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    # embeddings2 = model.encode(sentences2, convert_to_tensor=True)
    #! Compute cosine-similarities
    # cosine_scores = util.cos_sim(embeddings1, embeddings2)

    # Output the pairs with their score
    # for i in range(len(sentences1)):
    #     print("{} \t\t {} \t\t Score: {:.4f}".format(
    #         sentences1[i], sentences2[i], cosine_scores[i][i]
    #     ))

    threshold = 0.5  # Define your threshold here

    # Cluster sentences
    clusters, cosine_scores = cluster_sentences(sentences1, threshold)

    # Output the clusters
    for label, cluster in clusters.items():
        print("Cluster {}:".format(label))
        for sentence in cluster:
            print("\t{}".format(sentence))




if __name__ == '__main__':
    main()