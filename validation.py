import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import fbeta_score
from tqdm import tqdm


def validate(model, val_dl, threshold):
    model.eval()
    total_loss = 0
    true_labels, predictions = [], []
    with torch.no_grad():
        for data, target in tqdm(val_dl, desc='Validation metrics',
                                 total=len(val_dl)):
            true_labels.append(target.cpu().numpy())
            data, target = data.cuda().float(), target.cuda().float()

            pred = model(data)
            predictions.append(F.sigmoid(pred).cpu().numpy())
            total_loss += F.binary_cross_entropy_with_logits(pred,
                                                             target).item()

        avg_loss = total_loss / len(val_dl)
        predictions = np.vstack(predictions)
        true_labels = np.vstack(true_labels)
        f2_score = fbeta_score(true_labels, predictions > threshold,
                               beta=2, average='samples')
        return f2_score, avg_loss
