import time
import torch
import math
import numpy as np
import os
from tqdm import tqdm
import torch.nn.functional as F
import Net.NET_MRI
from dataloader_MCI import load_data
import gc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix
from configp_MCI import get_args
from util import model_save
from Net import HybridLoss
from datetime import datetime


def train_epoch(epoch, model, train_data, fo, criterion):
    """Train model for one epoch"""
    model.train()
    n_batches = len(train_data)

    # Learning rate scheduling configuration
    if epoch < 30:
        LEARNING_RATE = 0.00001
    elif epoch < 40:
        LEARNING_RATE = 0.000001
    elif epoch < 50:
        LEARNING_RATE = 0.000001
    else:
        LEARNING_RATE = 0.0000001

    # Configure optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': LEARNING_RATE},
    ], lr=0.0001)

    total_loss = 0
    source_correct = 0
    total_label = 0

    # Process each batch
    for (mri_data, pet_data, label) in tqdm(train_data, total=n_batches):
        mri_data, pet_data, label = mri_data.cuda(), pet_data.cuda(), label.cuda()
        result = model(mri_data, pet_data)

        # Handle different model output types
        if type(result) is tuple:
            if len(result) == 3:
                # Handle 3-output models
                P_l, P_r, output = result
                CosineLoss = 0
                HybridLoss = criterion['HybridLoss'](P_l, P_r, output, label)
                loss = HybridLoss + CosineLoss
            elif len(result) == 5:
                # Handle 5-output models
                left, right, P_l, P_r, output = result
                cosLossLabel = -torch.ones(left.size(0), dtype=torch.long).cuda()
                CosineLoss = criterion['CosineLoss'](F.normalize(left, dim=1), F.normalize(right, dim=1),
                                                     cosLossLabel).cuda()
                HybridLoss = criterion['HybridLoss'](P_l, P_r, output, label)
                loss = HybridLoss + CosineLoss
        else:
            # Handle single-output models
            loss = torch.nn.functional.cross_entropy(result, label)
            output = result

        # Calculate accuracy
        _, preds = torch.max(output, 1)
        source_correct += preds.eq(label.data.view_as(preds)).cpu().sum()
        total_label += label.size(0)

        # Update model weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Calculate epoch metrics
    acc = source_correct / total_label
    mean_loss = total_loss / n_batches
    print(f'Epoch: [{epoch:2d}], '
          f'Loss: {mean_loss:.6f}, '
          f'LR: {LEARNING_RATE:.6f}')

    # Write to log file
    log_str = f'Epoch: {epoch} Loss: {mean_loss} train_acc: {acc}\n'
    fo.write(log_str)

    # Clean up
    del acc, n_batches
    gc.collect()

    return mean_loss, source_correct, total_label


def val_model(epoch, val_data, model, log_best):
    """Evaluate model on validation data"""
    model.eval()
    print('Evaluating model on validation data...')
    total = 0
    total_loss = 0
    true_label = []
    data_pre = []

    with torch.no_grad():
        for (mri_data, pet_data, labels) in tqdm(val_data):
            mri_data, pet_data, labels = mri_data.cuda(), pet_data.cuda(), labels.cuda()
            result = model(mri_data, pet_data)

            # Handle different model output types
            if type(result) is tuple:
                if len(result) == 3:
                    P_l, P_r, output = result
                    CosineLoss = 0
                    HybridLoss = criterion['HybridLoss'](P_l, P_r, output, labels)
                    loss = HybridLoss + CosineLoss
                elif len(result) == 5:
                    left, right, P_l, P_r, output = result
                    cosLossLabel = -torch.ones(left.size(0), dtype=torch.long).cuda()
                    CosineLoss = criterion['CosineLoss'](F.normalize(left, dim=1), F.normalize(right, dim=1),
                                                         cosLossLabel).cuda()
                    HybridLoss = criterion['HybridLoss'](P_l, P_r, output, labels)
                    loss = HybridLoss + CosineLoss
            else:
                loss = torch.nn.functional.cross_entropy(result, labels)
                output = result

            total_loss += loss.item()

            # Collect predictions
            _, predicted = torch.max(output, 1)
            true_label.extend(list(labels.cpu().flatten().numpy()))
            data_pre.extend(list(predicted.cpu().flatten().numpy()))
            total += labels.size(0)

    # Calculate metrics
    mean_loss = total_loss / len(val_data)
    TN, FP, FN, TP = confusion_matrix(true_label, data_pre).ravel()
    ACC = 100 * (TP + TN) / (TP + TN + FP + FN)
    SEN = 100 * (TP) / (TP + FN)
    SPE = 100 * (TN) / (TN + FP)
    AUC = 100 * roc_auc_score(true_label, data_pre)

    # Print metrics
    print('TP:', TP, 'FP:', FP, 'FN:', FN, 'TN:', TN)
    print('ACC: %.4f %%' % ACC)
    print('SEN: %.4f %%' % SEN)
    print('SPE: %.4f %%' % SPE)
    print('AUC: %.4f %%' % AUC)

    # Write to log file
    log_str = (f'Epoch: {epoch}\nTP: {TP} TN: {TN} FP: {FP} FN: {FN} '
               f'ACC: {ACC} SEN: {SEN} SPE: {SPE} AUC: {AUC}\n')
    log_best.write(log_str)

    # Clean up
    del true_label, data_pre, mri_data, pet_data, labels, output, TN, FP, FN, TP
    gc.collect()
    return ACC, SEN, SPE, AUC, mean_loss


def t_model(test_data, model):
    """Evaluate model on test data"""
    model.eval()
    print('Evaluating model on test data...')
    total = 0
    true_label = []
    data_pre = []

    with torch.no_grad():
        for (mri_data, pet_data, labels) in tqdm(test_data):
            mri_data, pet_data, labels = mri_data.cuda(), pet_data.cuda(), labels.cuda()
            output = model(mri_data, pet_data)

            # Handle tuple outputs
            if type(output) is tuple:
                output = output[-1]

            # Collect predictions
            _, predicted = torch.max(output, 1)
            true_label.extend(list(labels.cpu().flatten().numpy()))
            data_pre.extend(list(predicted.cpu().flatten().numpy()))
            total += labels.size(0)

    # Calculate metrics
    TN, FP, FN, TP = confusion_matrix(true_label, data_pre).ravel()
    ACC = 100 * (TP + TN) / (TP + TN + FP + FN)
    SEN = 100 * (TP) / (TP + FN)
    SPE = 100 * (TN) / (TN + FP)
    AUC = 100 * roc_auc_score(true_label, data_pre)

    # Print results
    print('\nTest results:')
    print('TP:', TP, 'FP:', FP, 'FN:', FN, 'TN:', TN)
    print('ACC: %.4f %%' % ACC)
    print('SEN: %.4f %%' % SEN)
    print('SPE: %.4f %%' % SPE)
    print('AUC: %.4f %%' % AUC)

    # Clean up
    del true_label, data_pre, mri_data, pet_data, labels, output, TN, FP, FN, TP
    gc.collect()


if __name__ == '__main__':
    # Setup experiment logging
    experience_log = "DUAL2_PET2_MRI2_MCI_r2_l2_r2l2cls2_load_dict_dataset21_" + str(datetime.now().timestamp())
    log_path = "./" + experience_log
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # Get configuration arguments
    args = get_args()
    print(vars(args))

    # Set random seeds for reproducibility
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Initialize model
    model = Net.NET_MRI.NET_MRI_1(2, (105, 125, 105), 25).cuda()

    # Load pretrained weights if available
    # model.load_state_dict(torch.load('path/to/pretrained/model.pt'))

    # Load datasets
    train_data = load_data(args, args.train_root_path_MRI, args.train_root_path_PET,
                           args.PMCI_dir, args.SMCI_dir)
    test_data = load_data(args, args.test_root_path_MRI, args.test_root_path_PET,
                          args.PMCI_dir, args.SMCI_dir)
    val_data = load_data(args, args.val_root_path_MRI, args.val_root_path_PET,
                         args.PMCI_dir, args.SMCI_dir)

    # Define loss functions
    criterion = {
        'HybridLoss': HybridLoss.HybridLoss().cuda(),
        'CosineLoss': torch.nn.CosineEmbeddingLoss().cuda()
    }

    # Initialize tracking variables
    loss_best_model_dir = None
    acc_best_model_dir = None
    train_best_loss = float('inf')
    val_best_loss = float('inf')
    train_best_acc = 0
    val_best_acc = 0
    t_SEN = 0
    t_SPE = 0
    t_AUC = 0

    # Initialize history trackers
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    test_acc_all = []

    start_time = time.time()

    # Training loop
    for epoch in range(1, args.nepoch + 1):
        # Open log files
        fo = open(log_path + "/train_log.txt", "a")
        log_best = open(log_path + '/val_log.txt', 'a')

        # Train epoch
        train_loss, train_correct, len_train = train_epoch(epoch, model, train_data, fo, criterion)

        # Save best loss model
        if train_loss < train_best_loss:
            train_best_loss = train_loss
            path_dir = log_path + '/model/bestLoss/epoch' + str(epoch) + '_Loss' + str(train_best_loss) + '.pt'
            loss_best_model_dir = path_dir
            model_save(model, path_dir)

        # Calculate training accuracy
        train_acc = 100. * train_correct / len_train
        if train_acc > train_best_acc:
            train_best_acc = train_acc

        # Print training metrics
        print(f'Current loss: {train_loss:.6f}, Best loss: {train_best_loss:.6f}')
        print(f'Train accuracy: {train_acc:.2f}% ({train_correct}/{len_train})')

        # Validate model
        ACC, SEN, SPE, AUC, val_loss = val_model(epoch, val_data, model, log_best)

        # Save best accuracy model
        if ACC >= val_best_acc:
            val_best_acc = ACC
            t_SEN = SEN
            t_SPE = SPE
            t_AUC = AUC
            path_dir = (log_path + '/model/bestACC/ACC' + str(val_best_acc) +
                        '_SEN' + str(t_SEN) + '_SPE' + str(t_SPE) +
                        '_AUC' + str(t_AUC) + '_epoch' + str(epoch) + '.pt')
            acc_best_model_dir = path_dir
            model_save(model, path_dir)

        # Save last 10 epochs
        if epoch >= (args.nepoch - 10):
            path_dir = (log_path + '/model/Last10/ACC' + str(ACC) +
                        '_SEN' + str(SEN) + '_SPE' + str(SPE) +
                        '_AUC' + str(AUC) + '_epoch' + str(epoch) + '.pt')
            model_save(model, path_dir)

        # Update best results log
        log_best.write(f'Best results:\nACC: {val_best_acc} SEN: {t_SEN} SPE: {t_SPE} AUC: {t_AUC}\n\n')

        # Print epoch summary
        print(f'Epoch {epoch} train accuracy: {train_acc:.2f}%')
        print(f'Best validation accuracy: {val_best_acc:.2f}%\n')
        fo.write(f'Train accuracy: {train_acc} Current loss: {train_loss} Best loss: {train_best_loss}\n\n')

        # Track history for plotting
        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)
        val_loss_all.append(val_loss)
        test_acc_all.append(ACC)

        # Clean up
        del train_loss, train_correct, len_train, train_acc
        gc.collect()
        fo.close()
        log_best.close()

    # Final evaluation
    print(f"Experiment: {experience_log}")

    # Evaluate best loss model
    print("\nEvaluating best loss model:")
    model.load_state_dict(torch.load(loss_best_model_dir))
    t_model(test_data, model)

    # Evaluate best accuracy model
    print("\nEvaluating best accuracy model:")
    model.load_state_dict(torch.load(acc_best_model_dir))
    t_model(test_data, model)

    # Calculate total time
    total_time = time.time() - start_time
    print(f"Training completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")

    # Save training history
    train_process = pd.DataFrame({
        "epoch": range(args.nepoch),
        "train_loss_all": train_loss_all,
        "train_acc_all": train_acc_all,
        "val_loss_all": val_loss_all,
        "test_acc_all": test_acc_all
    })
    train_process.to_csv(log_path + '/training_history.csv')

    # Plot training history
    plt.figure(figsize=(12, 4))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_process.epoch, train_process.train_loss_all, label="Train loss")
    plt.plot(train_process.epoch, train_process.val_loss_all, label="Validation loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(train_process.epoch, train_process.train_acc_all, label="Train accuracy")
    plt.plot(train_process.epoch, train_process.test_acc_all, label="Validation accuracy")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")

    # Save plot
    plt.tight_layout()
    plt.savefig(log_path + "/training_history.png")
    plt.close()