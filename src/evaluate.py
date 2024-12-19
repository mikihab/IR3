import pandas as pd


def load_results(output_file):
    """Load the output results from a CSV file."""
    return pd.read_csv(output_file)


def load_ground_truth(ground_truth_file):
    """Load the ground truth data from a CSV file."""
    return pd.read_csv(ground_truth_file)


def compute_average_precision(retrieved, relevant):
    """Compute the average precision for a single query."""
    if len(relevant) == 0:
        return 0.0

    ap = 0.0
    relevant_count = 0

    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            relevant_count += 1
            ap += relevant_count / (i + 1)  # Precision at this rank

    return ap / len(relevant)  # Normalize by the number of relevant documents


def compute_mean_average_precision(results, ground_truth, k):
    """Compute MAP@K."""
    average_precisions = []

    for query_id in results["Query number"].unique():
        # Get relevant documents for this query
        relevant = set(
            ground_truth[ground_truth["Query_number"] == query_id]["doc_number"]
        )

        # Get top K retrieved documents for this query
        retrieved = (
            results[results["Query number"] == query_id]["doc_number"].head(k).tolist()
        )

        ap = compute_average_precision(retrieved, relevant)
        average_precisions.append(ap)

    return (
        sum(average_precisions) / len(average_precisions) if average_precisions else 0.0
    )


def compute_average_recall(retrieved, relevant):
    """Compute the average recall for a single query."""
    if len(relevant) == 0:
        return 0.0

    retrieved_set = set(retrieved)
    relevant_set = set(relevant)

    # Recall = TP / (TP + FN)
    tp = len(retrieved_set.intersection(relevant_set))  # True positives
    recall = tp / len(relevant_set)

    return recall


def compute_mean_average_recall(results, ground_truth, k):
    """Compute MAR@K."""
    average_recalls = []

    for query_id in results["Query number"].unique():
        # Get relevant documents for this query
        relevant = set(
            ground_truth[ground_truth["Query_number"] == query_id]["doc_number"]
        )

        # Get top K retrieved documents for this query
        retrieved = (
            results[results["Query number"] == query_id]["doc_number"].head(k).tolist()
        )

        recall = compute_average_recall(retrieved, relevant)
        average_recalls.append(recall)

    return sum(average_recalls) / len(average_recalls) if average_recalls else 0.0


def evaluate(output_file, ground_truth_file, k_values):
    """Evaluate the results and print MAP@K and MAR@K."""
    results = load_results(output_file)
    ground_truth = load_ground_truth(ground_truth_file)

    for k in k_values:
        map_k = compute_mean_average_precision(results, ground_truth, k)
        mar_k = compute_mean_average_recall(results, ground_truth, k)

        print(f"MAP@{k}: {map_k:.4f}")
        print(f"MAR@{k}: {mar_k:.4f}")


if __name__ == "__main__":
    output_file = "ranked_results.csv"  # Your output results file
    ground_truth_file = "dev_query_results.csv"  # Your ground truth file
    k_values = [1, 3, 5, 10]

    evaluate(output_file, ground_truth_file, k_values)
