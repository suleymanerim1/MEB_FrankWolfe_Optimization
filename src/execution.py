import numpy as np
from src.logger import logging
import os
from src.FrankWolfeVariants import awayStep_FW, blendedPairwise_FW, one_plus_eps_MEB_approximation
from src.plotting import plot_graphs, plot_points_circle
from src.utils import print_on_console, create_save_dict_full


def execute_algorithm(method, A, config, incremented_path, test_data=None):
    show_graphs = config.get('show_graphs')
    maxIter = eval(config.get('maxiter'))
    epsilon = eval(config.get('epsilon'))
    perform_test = config.get('perform_test')

    print("\n*****************")
    title = method.upper()
    print(title)
    print("*****************")

    logging.info("\n" + method + " algorithm started!")
    out_dict = {}
    if method == "asfw":
        line_search_strategy = config.get('line_search_asfw')
        out_dict = awayStep_FW(A, epsilon, line_search_strategy, maxIter)
    elif method == "bpfw":
        line_search_strategy = config.get('line_search_bpfw')
        out_dict = blendedPairwise_FW(A, epsilon, line_search_strategy, maxIter)
    elif method == "appfw":
        out_dict = one_plus_eps_MEB_approximation(A, epsilon, maxIter)

    # Print results:
    print_on_console(out_dict)

    # Create dict to save results on YAML
    train_dict = create_save_dict_full(out_dict)
    # Plot graphs for awayStep_FW
    graph_path = os.path.join(incremented_path, method + "_graphs")
    plot_graphs(title, show_graphs, graph_path, out_dict)
    if A.shape[0] == 2:
        plot_points_circle(A, out_dict.get("radius"), out_dict.get("center"), title, graph_path, test_data, show_graphs)

    test_dict = None
    # test_data, center, radius
    if perform_test:
        logging.info(method + " Test")
        test_dict = test_algorithm(test_data, out_dict.get("center"), out_dict.get("radius"))

    return train_dict, test_dict


# Algorithm Testing
def test_algorithm(test_data, center, radius):
    # test_data = (test_X, test_Y)
    # 0: good, 1: anomaly
    logging.info("\n----Testing Started------")

    true_negative = 0
    true_positive = 0
    false_negative = 0
    false_positive = 0
    test_X, test_Y = test_data
    number_of_test_points = test_X.shape[1]

    for point_idx in range(number_of_test_points):

        point = test_X[:, point_idx]
        y = test_Y[point_idx]
        assert y in [0, 1], "Variable must be either 0 or 1"
        dist = calculate_euc_distance(point, center)

        # logging.info(f"y:{y},dist>radius: {dist > radius},  distance: {dist}, radius {radius}")
        if y == 1:  # Then point is anomaly
            if dist > radius:  # then point is out of circle and it is anomaly
                true_positive += 1
                # logging.info(f"true positive ++, {true_positive}")
            else:  # Then point is inside circle but it is anomaly
                false_negative += 1
                # logging.info(f"false negative ++, {false_negative}")
        else:  # Then point is good
            if dist > radius:  # Then point is out of circle but it is good
                false_positive += 1
                # logging.info(f"false_positive ++, {false_positive}")
            else:  # Then point is in circle and it is good
                true_negative += 1
                # logging.info(f"true_negative ++, {true_negative}")

    logging.info(f"total points: {test_X.shape[1]}")
    logging.info(f"true positive: {true_positive}")
    logging.info(f"false negative: {false_negative}")
    logging.info(f"false positive: {false_positive}")
    logging.info(f"true_negative: {true_negative}")

    out_dict = {
        "tp": true_positive,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative
    }

    return create_test_save_dict(out_dict)


def create_test_save_dict(out_dict):
    TP = out_dict.get("tp")
    FP = out_dict.get("fp")
    TN = out_dict.get("tn")
    FN = out_dict.get("fn")

    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)

    if precision == 0 or recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    save_dict = {
        "true_positive": TP,
        "true_negative": TN,
        "false_positive": FP,
        "false_negative": FN,
        "precision": round(100*precision, 3),
        "recall": round(100*recall, 3),
        "f1_score": round(100*f1, 3)
    }

    return save_dict


def calculate_euc_distance(point1, point2):
    return np.linalg.norm(point1 - point2)
