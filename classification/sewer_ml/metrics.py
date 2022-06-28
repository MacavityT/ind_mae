import numpy as np
import torch
import pickle
import util.misc as misc
import torch.distributed as dist


def average_precision(scores, target, max_k=None):

    assert (scores.shape == target.shape
            ), "The input and targets do not have the same shape"
    assert (scores.ndim == 1
            ), "The input has dimension {}, but expected it to be 1D".format(
                scores.shape)

    # sort examples
    indices = np.argsort(scores, axis=0)[::-1]

    total_cases = np.sum(target)

    if max_k == None:
        max_k = len(indices)

    # Computes prec@i
    pos_count = 0.0
    total_count = 0.0
    precision_at_i = 0.0

    for i in range(max_k):
        label = target[indices[i]]
        total_count += 1
        if label == 1:
            pos_count += 1
            precision_at_i += pos_count / total_count
        if pos_count == total_cases:
            break

    if pos_count > 0:
        precision_at_i /= pos_count
    else:
        precision_at_i = 0
    return precision_at_i


def micro_f1(Ng, Np, Nc):
    mF1 = (2 * np.sum(Nc)) / (np.sum(Np) + np.sum(Ng))

    return mF1


def macro_f1(Ng, Np, Nc):
    n_class = len(Ng)
    precision_k = Nc / Np
    recall_k = Nc / Ng
    F1_k = (2 * precision_k * recall_k) / (precision_k + recall_k)

    F1_k[np.isnan(F1_k)] = 0

    MF1 = np.sum(F1_k) / n_class

    return precision_k, recall_k, F1_k, MF1


def overall_metrics(Ng, Np, Nc):
    OP = np.sum(Nc) / np.sum(Np)
    OR = np.sum(Nc) / np.sum(Ng)
    OF1 = (2 * OP * OR) / (OP + OR)

    return OP, OR, OF1


def per_class_metrics(Ng, Np, Nc):
    n_class = len(Ng)
    CP = np.sum(Nc / Np) / n_class
    CR = np.sum(Nc / Ng) / n_class
    CF1 = (2 * CP * CR) / (CP + CR)

    return CP, CR, CF1


def mean_average_precision(ap):
    return np.mean(ap)


def exact_match_accuracy(scores, targets, threshold=0.5):
    n_examples, n_class = scores.shape

    binary_mat = np.equal(targets, (scores >= threshold))
    row_sums = binary_mat.sum(axis=1)

    perfect_match = np.zeros(row_sums.shape)
    perfect_match[row_sums == n_class] = 1

    EMAcc = np.sum(perfect_match) / n_examples

    return EMAcc


def class_weighted_f2(Ng, Np, Nc, weights, threshold=0.5):
    n_class = len(Ng)
    precision_k = Nc / Np
    recall_k = Nc / Ng
    F2_k = (5 * precision_k * recall_k) / (4 * precision_k + recall_k)

    F2_k[np.isnan(F2_k)] = 0

    ciwF2 = F2_k * weights
    ciwF2 = np.sum(ciwF2) / np.sum(weights)

    return ciwF2, F2_k


def evaluation(scores, targets, threshold=0.5, weights=None):
    scores = torch.vstack(scores).cpu().numpy()
    targets = torch.vstack(targets).cpu().numpy()

    assert (
        scores.shape == targets.shape
    ), "The input and targets do not have the same size: Input: {} - Targets: {}".format(
        scores.shape, targets.shape)

    _, n_class = scores.shape

    if weights is None:
        LabelWeightDict = {
            "RB": 1.00,
            "OB": 0.5518,
            "PF": 0.2896,
            "DE": 0.1622,
            "FS": 0.6419,
            "IS": 0.1847,
            "RO": 0.3559,
            "IN": 0.3131,
            "AF": 0.0811,
            "BE": 0.2275,
            "FO": 0.2477,
            "GR": 0.0901,
            "PH": 0.4167,
            "PB": 0.4167,
            "OS": 0.9009,
            "OP": 0.3829,
            "OK": 0.4396
        }
        Labels = list(LabelWeightDict.keys())
        weights = list(LabelWeightDict.values())

    # Arrays to hold binary classification information, size n_class +1 to also hold the implicit normal class

    # Nc = Number of Correct Predictions  -> TP
    Nc = np.zeros(n_class + 1)
    # Np = Total number of Predictions  -> TP + FP
    Np = np.zeros(n_class + 1)
    # Ng = Total number of Ground Truth occurences  -> TP + FN
    Ng = np.zeros(n_class + 1)

    # False Positives = Np - Nc
    # False Negatives = Ng - Nc
    # True Positives = Nc
    # True Negatives = n_examples - Np + (Ng - Nc)

    # Array to hold the average precision metric. only size n_class, since it is not possible to calculate for the implicit normal class
    ap = np.zeros(n_class)

    for k in range(n_class):
        tmp_scores = scores[:, k]
        tmp_targets = targets[:, k]

        # Necessary if using MultiLabelSoftMarginLoss, instead of BCEWithLogitsLoss
        tmp_targets[tmp_targets == -1] = 0

        Ng[k] = np.sum(tmp_targets == 1)
        Np[k] = np.sum(
            tmp_scores >= threshold
        )  # when >= 0 for the raw input, the sigmoid value will be >= 0.5
        Nc[k] = np.sum(tmp_targets * (tmp_scores >= threshold))

        ap[k] = average_precision(tmp_scores, tmp_targets)

    # Get values for "implict" normal class
    tmp_scores = np.sum(scores >= threshold, axis=1)
    tmp_scores[tmp_scores > 0] = 1
    tmp_scores = np.abs(tmp_scores - 1)

    tmp_targets = targets.copy()
    tmp_targets[
        targets ==
        -1] = 0  # Necessary if using MultiLabelSoftMarginLoss, instead of BCEWithLogitsLoss
    tmp_targets = np.sum(tmp_targets, axis=1)
    tmp_targets[tmp_targets > 0] = 1
    tmp_targets = np.abs(tmp_targets - 1)

    Ng[-1] = np.sum(tmp_targets == 1)
    Np[-1] = np.sum(tmp_scores >= threshold)
    Nc[-1] = np.sum(tmp_targets * (tmp_scores >= threshold))

    # If Np is 0 for any class, set to 1 to avoid division with 0
    Np[Np == 0] = 1

    # Overall Precision, Recall and F1
    OP, OR, OF1 = overall_metrics(Ng, Np, Nc)

    # Per-Class Precision, Recall and F1
    CP, CR, CF1 = per_class_metrics(Ng, Np, Nc)

    # Macro F1
    precision_k, recall_k, F1_k, MF1 = macro_f1(Ng, Np, Nc)

    # Micro F1
    mF1 = micro_f1(Ng, Np, Nc)

    # Zero-One exact match accuracy
    EMAcc = exact_match_accuracy(scores, targets)

    # Mean Average Precision (mAP)
    mAP = mean_average_precision(ap)

    (
        F2,
        F2_k,
    ) = class_weighted_f2(Ng[:-1], Np[:-1], Nc[:-1], weights)

    F2_normal = (5 * precision_k[-1] * recall_k[-1]) / (4 * precision_k[-1] +
                                                        recall_k[-1])

    new_metrics = {
        "F2": F2,
        "F2_class": list(F2_k) + [F2_normal],
        "F1_Normal": F1_k[-1],
    }

    main_metrics = {
        "OP": OP,
        "OR": OR,
        "OF1": OF1,
        "CP": CP,
        "CR": CR,
        "CF1": CF1,
        "MF1": MF1,
        "mF1": mF1,
        "EMAcc": EMAcc,
        "mAP": mAP,
    }

    auxillery_metrics = {
        "P_class": list(precision_k),
        "R_class": list(recall_k),
        "F1_class": list(F1_k),
        "AP": list(ap),
        "Np": list(Np),
        "Nc": list(Nc),
        "Ng": list(Ng),
    }

    return new_metrics, main_metrics, auxillery_metrics


def collect_results_gpu(result_part, size):
    """Collect results under gpu mode.

    On gpu mode, this function will encode results to gpu tensors and use gpu
    communication for results collection.

    Args:
        result_part (list): Result list containing result parts
            to be collected.
        size (int): Size of the results, commonly equal to length of
            the results.

    Returns:
        list: The collected results.
    """
    rank = misc.get_rank()
    world_size = misc.get_world_size()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(bytearray(pickle.dumps(result_part)),
                               dtype=torch.uint8,
                               device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_result = pickle.loads(recv[:shape[0]].cpu().numpy().tobytes())
            # When data is severely insufficient, an empty part_result
            # on a certain gpu could makes the overall outputs empty.
            if part_result:
                part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results


@torch.no_grad()
def sewer_evaluate(data_loader, model, device):
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=data_loader.dataset.class_weights.to(device,
                                                        non_blocking=True))
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    # prepare list for evaluation
    losses = []
    scores = []
    targets = []

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            target = target.type_as(images)
            output = model(images)
            score = torch.sigmoid(output)
            loss = criterion(output, target)

            metric_logger.update(loss=loss.item())

            losses.append(loss)
            scores.append(score)
            targets.append(target)

    new_metrics, main_metrics, auxillery_metrics = evaluation(scores=scores,
                                                              targets=targets)
    all_metrics = dict()
    all_metrics.update(new_metrics)
    all_metrics.update(main_metrics)
    all_metrics.update(auxillery_metrics)
    all_metrics['loss'] = metric_logger.loss.global_avg

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("* loss {losses.global_avg:.3f}".format(losses=metric_logger.loss))

    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return all_metrics
