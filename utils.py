import os
import sys
import time
import datetime
from tqdm import tqdm
import numpy as np
import torch
import torch.cuda.amp as amp
from torch.optim import AdamW
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from main_model import indicator, Euclidean_distance

# Import necessary modules from sklearn
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

# Set CUDA environment variable
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Timestamp for filenames
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"./record_df/result_val{current_time}.txt"
filename1 = f"./record_df/result_ext{current_time}.txt"
filename2 = f"./record_df/fill_metric{current_time}.txt"

# Define initial and final weights
INITIAL_NOISE_W = 1
INITIAL_MSE_W = 1
INITIAL_CE_W = 0.1
FINAL_NOISE_W = 0.1
FINAL_MSE_W = 0.1
FINAL_CE_W = 0.5

loss_forward = []

def train(model, model1, config, train_loader, valid_loader=None, valid_epoch_interval=5, foldername=""):
    """Train the provided models with training and optionally validation data."""
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer1 = AdamW(model1.parameters(), lr=1e-4, weight_decay=1e-4)

    scaler = amp.GradScaler()


    # Define learning rate schedulers
    p1, p2 = int(0.75 * config["epochs"]), int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[p1, p2], gamma=0.1)
    lr_scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[p1, p2], gamma=0.1)

    for epoch_no in range(200):
        avg_loss, fl_loss, mse_loss = 0.0, 0.0, 0.0
        model.train()
        model1.train()

        produce_all, train_csdi, test_csdi, original_all = [], [], [], []
        score_all, status_all, test_point, label_all = [], [], [], []

        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                optimizer1.zero_grad()

                loss0, generated_data, gt_point, status, original_data, imputed_point = model(train_batch)
                loss1 = (((generated_data - original_data) * gt_point) ** 2).sum() / gt_point.sum()
                imputed_data = (generated_data * (1 - imputed_point) + original_data * imputed_point).view(-1, 45)

                data_im = (generated_data * (1 - imputed_point) + original_data * imputed_point).view(-1, 45)
                train_csdi.append(data_im)
                original_all.append(original_data)
                score_all.append(imputed_point)
                status_all.append(status)

                imputed_data = preprocess_imputed_data(imputed_data)

                x_num, x_cat = separate_data(imputed_data, 31, 40)
                output = model1(x_cat, x_num)

                loss2 = torch.nn.BCEWithLogitsLoss()(output.squeeze(1), status.float())

                total_epochs, alpha = 100, batch_no / total_epochs
                noise_w, mse_w, ce_w = calculate_weights(alpha)

                loss = ce_w * loss2 + mse_w * loss1 + noise_w * loss0

                scaler.scale(loss).backward()

                avg_loss += loss0.item()
                mse_loss += loss1.item()
                fl_loss += loss2.item()

                optimizer1.step()
                optimizer.step()
                scaler.step(optimizer1)
                scaler.step(optimizer)
                scaler.update()

                it.set_postfix(ordered_dict={
                    "avg_df_loss": avg_loss / batch_no,
                    "avg_fl_loss": fl_loss / batch_no,
                    "mse_loss": mse_loss / batch_no,
                    "epoch": epoch_no,
                }, refresh=False)

            lr_scheduler1.step()
            lr_scheduler.step()

        if valid_loader:
            validate_model(model, model1, valid_loader, epoch_no)

def preprocess_imputed_data(imputed_data):
    """Preprocess imputed data for classification."""
    imputed_data[:, 31:40] = (imputed_data[:, 31:40] > 0.5).int()
    imputed_data[:, 40] = Euclidean_distance(imputed_data[:, 40:], 5)
    return imputed_data[:, :41]

def separate_data(data, num_idx, cat_idx):
    """Separate numerical and categorical data."""
    x_num = data[:, :num_idx].to("cuda")
    x_cat = data[:, num_idx:cat_idx].type(torch.LongTensor).to("cuda")
    return x_num, x_cat

def calculate_weights(alpha):
    """Calculate the weights for combined losses."""
    noise_w = INITIAL_NOISE_W * (1 - alpha) + FINAL_NOISE_W * alpha
    mse_w = INITIAL_MSE_W * (1 - alpha) + FINAL_MSE_W * alpha
    ce_w = INITIAL_CE_W * (1 - alpha) + FINAL_CE_W * alpha
    return noise_w, mse_w, ce_w

def validate_model(model, model1, valid_loader, epoch_no):
    """Validate the model using the validation dataset."""
    model.eval()
    model1.eval()

    avg_df_valid, avg_res_valid, avg_mse_valid = 0, 0, 0

    with torch.no_grad():
        with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
            predict_all, label_all = [], []

            for batch_no, valid_batch in enumerate(it, start=1):
                loss0, produce_data, gtv_point, label, initial_data, filled_point = model(valid_batch, is_train=0)

                valid_data = (produce_data * (1 - filled_point) + initial_data * filled_point).view(-1, 45)
                loss1 = (((produce_data - initial_data) * gtv_point) ** 2).sum() / gtv_point.sum()
                avg_df_valid += loss0.item()
                avg_mse_valid += loss1.item()

                valid_data = preprocess_imputed_data(valid_data)
                x_num, x_cat = separate_data(valid_data, 31, 40)
                predict = model1(x_cat, x_num)

                loss2 = torch.nn.BCEWithLogitsLoss()(predict.squeeze(1), label.float())
                avg_res_valid += loss2.item()

                predict_all.append(torch.sigmoid(predict).cpu().detach().flatten())
                label_all.append(label)

                it.set_postfix(ordered_dict={
                    "valid_df_loss": avg_df_valid / batch_no,
                    "valid_res_loss": avg_res_valid / batch_no,
                    "mse_loss": avg_mse_valid / batch_no,
                    "epoch": epoch_no,
                }, refresh=False)

        evaluate_performance(predict_all, label_all)

def evaluate_performance(predict_all, label_all):
    """Evaluate the performance of the model."""
    predict_all = torch.cat(tuple(predict_all), 0)
    label_all = torch.cat(tuple(label_all), 0)

    fpr, tpr, _ = roc_curve(label_all.cpu().detach(), predict_all.cpu().detach())
    auc = roc_auc_score(label_all.cpu().detach(), predict_all.cpu().detach())
    print("\nAUC:", auc)

    cm = confusion_matrix(label_all.cpu().detach(), (predict_all > 0.5).int())
    print("\nConfusion Matrix:\n", cm)

    TN, FP, FN, TP = cm.ravel()
    acc, sens, spec = indicator(TN, FP, FN, TP)
    print("\nAccuracy:", acc, "\nSensitivity:", sens, "\nSpecificity:", spec)

def quantile_loss(target, forecast, q, eval_points):
    """Calculate quantile loss."""
    return 2 * torch.sum(torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q)))

def calc_denominator(target, eval_points):
    """Calculate the denominator for CRPS calculation."""
    return torch.sum(torch.abs(target * eval_points))

def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    """Calculate the continuous ranked probability score (CRPS)."""
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler
    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for q in quantiles:
        q_pred = torch.cat([torch.quantile(forecast[j:j+1], q, dim=1) for j in range(len(forecast))], 0)
        q_loss = quantile_loss(target, q_pred, q, eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def normalization(tmp_values, tmp_masks):
    """Normalize data in the range [0, 1]."""
    _, dim = tmp_values.shape
    mean = np.zeros(dim)
    std = np.zeros(dim)
    for k in range(dim):
        c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
        mean[k] = c_data.mean()
        std[k] = c_data.std()
    norm_data = (tmp_values - mean) / std * tmp_masks
    return norm_data, {'mean': mean, 'std': std}

def renormalization(norm_data, norm_parameters):
    """Renormalize data to the original range."""
    mean, std = norm_parameters['mean'], norm_parameters['std']
    renorm_data = norm_data * std + mean
    return renorm_data

def rounding(imputed_data, data_x):
    """Round imputed data for categorical variables."""
    _, dim = data_x.shape
    rounded_data = imputed_data.copy()
    for i in range(dim):
        temp = data_x[~np.isnan(data_x[:, i]), i]
        if len(np.unique(temp)) < 20:
            rounded_data[:, i] = np.round(rounded_data[:, i])
    return rounded_data

def test(model, model1, config, ext_loader, foldername=""):
    """Test the models and evaluate performance on external data."""
    avg_loss_valid = 0
    predict_all, predict_proba, label_all, data_all = [], [], [], []

    with torch.no_grad():
        model.eval()
        model1.eval()
        with tqdm(ext_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, valid_batch in enumerate(it, start=1):
                start_time = time.time()
                loss0, produce_data, gte_point, label, initial_data, filled_point = model(valid_batch, is_train=0)
                loss1 = (((produce_data - initial_data) * gte_point) ** 2).sum() / gte_point.sum()

                ext_data = (produce_data * (1 - filled_point) + initial_data * filled_point).view(-1, 45)
                data_all.append(ext_data.clone())

                ext_data = preprocess_imputed_data(ext_data)
                x_num, x_cat = separate_data(ext_data, 31, 40)
                predict = model1(x_cat, x_num)

                loss2 = torch.nn.BCEWithLogitsLoss()(predict.squeeze(1), label.float())
                avg_loss_valid += (loss0 + loss1 + loss2).item()

                predict_all.append(torch.sigmoid(predict).cpu().detach().flatten())
                predict_proba.append((predict_all[-1] > 0.5).int())
                label_all.append(label)

                end_time = time.time()
                print("Batch processing time: {:.2f} seconds".format(end_time - start_time))

        evaluate_performance(predict_all, label_all)





