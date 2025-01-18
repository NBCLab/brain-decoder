import numpy as np


def mix_match(similarity):
    accuracies = []
    for row_index in range(len(similarity)):
        current_row_accumulator = 0
        for col_index in range(len(similarity[row_index])):
            if col_index == row_index:
                continue
            else:
                if similarity[row_index][row_index] > similarity[row_index][col_index]:
                    current_row_accumulator += 1

        accuracies.append(current_row_accumulator / (len(similarity[row_index]) - 1))

    return np.mean(accuracies)


def recall_n(y_pred, y_truth, n_first=10, thresh=0.95, reduce_mean=False):
    assert (y_pred.ndim in (1, 2)) and (
        y_truth.ndim in (1, 2)
    ), "arrays should be of dimension 1 or 2"
    assert y_pred.shape == y_truth.shape, "both arrays should have the same shape"

    if y_pred.ndim == 1:
        # recall@n for a single sample
        targets = np.where(y_truth >= thresh)[0]
        pred_n_first = np.argsort(y_pred)[::-1][:n_first]

        if len(targets) > 0:
            ratio_in_n = len(np.intersect1d(targets, pred_n_first)) / len(targets)
        else:
            ratio_in_n = np.nan

        return ratio_in_n
    else:
        # recall@n for a dataset (mean of recall@n for all samples)
        result = np.zeros(len(y_pred))
        for i, (sample_y_pred, sample_y_truth) in enumerate(zip(y_pred, y_truth)):
            result[i] = recall_n(sample_y_pred, sample_y_truth, n_first, thresh)
        if reduce_mean:
            return np.nanmean(result)

        return result
