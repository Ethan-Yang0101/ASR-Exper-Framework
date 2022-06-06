import matplotlib.pyplot as plt
import numpy as np


def make_reliability_diagrams(confidences, predictions, labels, n_bins=10, adaptive=False):
    """
    confidences - a list (size n)
    predictions - a list (size n)
    predictions - a list (size n)
    adaptive: whether use adaptive ECE
    """
    confidences = np.array(confidences)
    predictions = np.array(predictions)
    labels = np.array(labels)
    accuracies = predictions == labels

    # Reliability diagram
    bins = np.linspace(0, 1, n_bins + 1)
    width = 1.0 / n_bins
    bin_centers = np.linspace(0, 1.0 - width, n_bins) + width / 2
    if not adaptive:
        bin_indices = [(confidences >= bin_lower) * (confidences < bin_upper)
                       for bin_lower, bin_upper in zip(bins[:-1], bins[1:])]
        bin_corrects = np.array([np.mean(accuracies[bin_index])
                                 for bin_index in bin_indices])
        bin_scores = np.array([np.mean(confidences[bin_index])
                               for bin_index in bin_indices])
        bin_corrects = np.nan_to_num(bin_corrects)
        bin_scores = np.nan_to_num(bin_scores)
    else:
        n_classes = len(np.unique(labels))
        avg_corrects, avg_scores = np.zeros(n_bins), np.zeros(n_bins)
        for class_num in range(n_classes):
            conf = confidences[labels == class_num]
            pred = predictions[labels == class_num]
            lab = labels[labels == class_num]
            acc = pred == lab
            sort_indices = np.argsort(conf)
            bin_size = len(sort_indices) // n_bins
            bin_indices = [sort_indices[n * bin_size: (n+1) * bin_size]
                           for n in range(0, n_bins-1)]
            bin_indices.append(sort_indices[(n_bins-1) * bin_size:])
            bin_corrects = np.array([np.mean([acc[index] for index in bin_index])
                                     for bin_index in bin_indices])
            bin_scores = np.array([np.mean([conf[index] for index in bin_index])
                                   for bin_index in bin_indices])
            bin_corrects = np.nan_to_num(bin_corrects)
            bin_scores = np.nan_to_num(bin_scores)
            avg_corrects += bin_corrects
            avg_scores += bin_scores
        bin_corrects = avg_corrects / n_classes
        bin_scores = avg_scores / n_classes

    plt.figure(0, figsize=(8, 8))
    gap = bin_scores - bin_corrects

    gap[bin_corrects == 1.0] = 0.01
    bin_corrects[bin_corrects == 1.0] = 0.15

    confs = plt.bar(bin_centers, bin_corrects, color=[
                    0, 0, 1], width=width, ec='black')
    gaps = plt.bar(bin_centers, gap, bottom=bin_corrects, color=[
                   1, 0.7, 0.7], alpha=0.5, width=width, hatch='//', edgecolor='r')

    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.legend([confs, gaps], ['Accuracy', 'Gap'],
               loc='lower right', fontsize='x-large')

    if not adaptive:
        ece = _calculate_ece(
            confidences, predictions, labels, n_bins=n_bins)
    else:
        n_classes = len(np.unique(labels))
        n_ece = 0.0
        for class_num in range(n_classes):
            conf = confidences[labels == class_num]
            pred = predictions[labels == class_num]
            lab = labels[labels == class_num]
            ece = _calculate_adaptive_ece(conf, pred, lab, n_bins=n_bins)
            n_ece += ece
        ece = n_ece / n_classes

    if adaptive:
        label_str = "ACE: "
    else:
        label_str = "ECE: "
    # Clean up
    bbox_props = dict(boxstyle="square", fc="lightgrey", ec="gray", lw=1.5)
    plt.text(0.85, 0.18, label_str + "{:.4f}".format(
        ece), ha="center", va="center", size=20, weight='normal', bbox=bbox_props)

    plt.title("Reliability Diagram", size=22)
    plt.ylabel("Accuracy",  size=18)
    plt.xlabel("Confidence",  size=18)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig('reliability_diagram.png')
    plt.show()
    return ece


def _calculate_ece(confidences, predictions, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    accuracies = predictions == labels

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) * (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin -
                          accuracy_in_bin) * prop_in_bin
    return ece


def _calculate_adaptive_ece(confidences, predictions, labels, n_bins=10):
    accuracies = predictions == labels
    sort_indices = np.argsort(confidences)
    bin_size = len(sort_indices) // n_bins
    bin_indices = [sort_indices[n * bin_size: (n+1) * bin_size]
                   for n in range(0, n_bins-1)]
    bin_indices.append(sort_indices[(n_bins-1) * bin_size:])

    ece = 0.0
    for indices in bin_indices:
        accuracy_in_bin = np.mean([accuracies[index] for index in indices])
        avg_confidence_in_bin = np.mean(
            [confidences[index] for index in indices])
        prop_in_bin = len(indices) / len(accuracies)
        if prop_in_bin > 0:
            ece += np.abs(avg_confidence_in_bin -
                          accuracy_in_bin) * prop_in_bin
    return ece


def make_stop_histogram(gt_length, pred_length, n_bins=10):
    gt_length = np.array(gt_length)
    pred_length = np.array(pred_length)
    gap = pred_length - gt_length
    max_length = max(max(gt_length), max(pred_length))
    bins = np.linspace(0, max_length, n_bins + 1)
    width = max_length / n_bins
    bin_centers = np.linspace(0, max_length - width, n_bins) + width / 2
    bin_indices = [(gt_length >= bin_lower) * (gt_length < bin_upper)
                   for bin_lower, bin_upper in zip(bins[:-1], bins[1:])]
    bin_gts = np.array([np.mean(gt_length[bin_index])
                        for bin_index in bin_indices])
    bin_preds = np.array([np.mean(pred_length[bin_index])
                          for bin_index in bin_indices])
    bin_gts = np.nan_to_num(bin_gts)
    bin_preds = np.nan_to_num(bin_preds)

    plt.figure(0, figsize=(8, 8))
    gap = bin_preds - bin_gts
    gts = plt.bar(bin_centers, bin_gts, color=[
        0, 0, 1], width=width, ec='black')
    gaps = plt.bar(bin_centers, gap, bottom=bin_gts, color=[
                   1, 0.7, 0.7], alpha=0.5, width=width, hatch='//', edgecolor='r')
    plt.legend([gts, gaps], ['GT Length', 'Gap'],
               loc='lower right', fontsize='x-large')

    max_val = max(max(bin_gts), max(bin_preds))
    mae = np.mean(np.abs(pred_length - gt_length))
    bbox_props = dict(boxstyle="square", fc="lightgrey", ec="gray", lw=1.5)
    plt.text(0.875 * max_length, 0.18 * max_val, "MAE: {:.4f}".format(
        mae), ha="center", va="center", size=20, weight='normal', bbox=bbox_props)

    plt.title("Length Histogram", size=22)
    plt.ylabel("Mean Binwise Length", size=18)
    plt.xlabel("Length Range", size=18)
    plt.savefig('length_histogram.png')
    plt.show()
