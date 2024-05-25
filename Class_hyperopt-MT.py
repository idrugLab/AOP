import sklearn
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from dataset_scoffold_random_dc import Graph_Classification_Dataset
from muti_model import EncoderModel, BertModel,CostomModel
from sklearn.metrics import r2_score, roc_auc_score, auc, precision_recall_curve, confusion_matrix
from hyperopt import fmin, tpe, hp
from utils import get_task_names
import os
import sys
import math
import csv

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# keras.backend.clear_session()
# os.environ['TF_DETERMINISTIC_OPS'] = '1'


def train_one_epoch(model,train_dataset,optimizer,loss_object, label=['Label']):
    for x, adjoin_matrix, y in train_dataset:
        with tf.GradientTape() as tape:
            seq = tf.cast(tf.math.equal(x, 0), tf.float32)
            mask = seq[:, tf.newaxis, tf.newaxis, :]
            preds = model(x, adjoin_matrix=adjoin_matrix, mask=mask, training=True)
            loss = 0
            for i in range(len(label)):
                y_label = y[:, i]
                y_pred = preds[:, i]
                validId = np.where((y_label == 0) | (y_label == 1))[0]
                if len(validId) == 0:
                    continue
                y_t = tf.gather(y_label, validId)
                y_p = tf.gather(y_pred, validId)

                loss += loss_object(y_t, y_p)
            loss = loss / (len(label))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def validate(model,val_dataset,label=['Label']):
    y_true = {}
    y_preds = {}
    for i in range(len(label)):
        y_true[i] = []
        y_preds[i] = []
    for x, adjoin_matrix, y in val_dataset:
        seq = tf.cast(tf.math.equal(x, 0), tf.float32)
        mask = seq[:, tf.newaxis, tf.newaxis, :]
        preds = model(x, mask=mask, adjoin_matrix=adjoin_matrix, training=False)
        for i in range(len(label)):
            y_label = y[:, i]
            y_pred = preds[:, i]
            y_true[i].append(y_label)
            y_preds[i].append(y_pred)
    y_tr_dict = {}
    y_pr_dict = {}
    for i in range(len(label)):
        y_tr = np.array([])
        y_pr = np.array([])
        for j in range(len(y_true[i])):
            a = np.array(y_true[i][j])
            b = np.array(y_preds[i][j])
            y_tr = np.concatenate((y_tr, a))
            y_pr = np.concatenate((y_pr, b))
        y_tr_dict[i] = y_tr
        y_pr_dict[i] = y_pr

    AUC_list = []

    for i in range(len(label)):
        y_label = y_tr_dict[i]
        y_pred = y_pr_dict[i]
        validId = np.where((y_label == 0) | (y_label == 1))[0]
        if len(validId) == 0:
            continue
        y_t = tf.gather(y_label, validId)

        y_p = tf.gather(y_pred, validId)
        if all(target == 0 for target in y_t) or all(target == 1 for target in y_t):
            AUC = float('nan')
            AUC_list.append(AUC)
            continue
        y_p = tf.sigmoid(y_p).numpy()
        AUC_new = sklearn.metrics.roc_auc_score(y_t, y_p, average=None)

        AUC_list.append(AUC_new)
    auc_new = np.nanmean(AUC_list)
    return auc_new


def test_result(model,task,test_dataset,seed,label=['Label']):
    y_true = {}
    y_preds = {}
    test_true = {}
    test_preds = {}
    for i in range(len(label)):
        y_true[i] = []
        y_preds[i] = []
    model.load_weights('classification_weights/{}_{}.h5'.format(task, seed))
    for x, adjoin_matrix, y in test_dataset:
        seq = tf.cast(tf.math.equal(x, 0), tf.float32)
        mask = seq[:, tf.newaxis, tf.newaxis, :]
        preds = model(x, mask=mask, adjoin_matrix=adjoin_matrix, training=False)
        for i in range(len(label)):
            y_label = y[:, i]
            y_pred = preds[:, i]
            y_true[i].append(y_label)
            y_preds[i].append(y_pred)
    y_tr_dict = {}
    y_pr_dict = {}
    for i in range(len(label)):
        y_tr = np.array([])
        y_pr = np.array([])
        for j in range(len(y_true[i])):
            a = np.array(y_true[i][j])
            if a.ndim == 0:
                continue
            b = np.array(y_preds[i][j])
            y_tr = np.concatenate((y_tr, a))
            y_pr = np.concatenate((y_pr, b))
        y_tr_dict[i] = y_tr
        y_pr_dict[i] = y_pr

    auc_list = []
    prauc_list = []
    for i in range(len(label)):
        y_label = y_tr_dict[i]
        y_pred = y_pr_dict[i]
        validId = np.where((y_label == 0) | (y_label == 1))[0]
        if len(validId) == 0:
            continue
        y_t = tf.gather(y_label, validId)
        test_true[i] = y_t
        y_p = tf.gather(y_pred, validId)
        if all(target == 0 for target in y_t) or all(target == 1 for target in y_t):
            AUC = float('nan')
            auc_list.append(AUC)
            continue
        y_p = tf.sigmoid(y_p).numpy()
        test_preds[i] = y_p
        AUC_new = sklearn.metrics.roc_auc_score(y_t, y_p, average=None)
        auc_list.append(AUC_new)
    return auc_list,y_tr_dict,y_pr_dict




def main(seed, args):
    label = ['Label']
    arch = {'name': 'Medium', 'path': 'medium3_weights_chem_pubchem_share'}
    pretraining = True
    trained_epoch = 20
    num_layers = 6
    d_model = 256
    addH = True
    dff = d_model * 2
    vocab_size = 18

    num_heads = args['num_heads']
    dense_dropout = args['dense_dropout']
    learning_rate = args['learning_rate']
    batch_size = args['batch_size']
    seed = seed
    np.random.seed(seed=seed)
    tf.random.set_seed(seed=seed)
    # 加载数据集
    train_dataset1, test_dataset1, val_dataset1 = Graph_Classification_Dataset(data_path+task_list[0] + '.csv', smiles_field='smiles',
                                                                            label_field=label, seed=seed,
                                                                            batch_size=batch_size, a=len(label),
                                                                            addH=True).get_data()
    train_dataset2, test_dataset2, val_dataset2 = Graph_Classification_Dataset(data_path+task_list[1] + '.csv', smiles_field='smiles',
                                                                            label_field=label, seed=seed,
                                                                            batch_size=batch_size, a=len(label),
                                                                            addH=True).get_data()
    train_dataset3, test_dataset3, val_dataset3 = Graph_Classification_Dataset(data_path+task_list[2] + '.csv', smiles_field='smiles',
                                                                            label_field=label, seed=seed,
                                                                            batch_size=batch_size, a=len(label),
                                                                            addH=True).get_data()
    train_dataset4, test_dataset4, val_dataset4 = Graph_Classification_Dataset(data_path+task_list[3] + '.csv', smiles_field='smiles',
                                                                            label_field=label, seed=seed,
                                                                            batch_size=batch_size, a=len(label),
                                                                            addH=True).get_data()
    train_dataset5, test_dataset5, val_dataset5 = Graph_Classification_Dataset(data_path+task_list[4] + '.csv', smiles_field='smiles',
                                                                            label_field=label, seed=seed,
                                                                            batch_size=batch_size, a=len(label),
                                                                            addH=True).get_data()
    train_dataset6, test_dataset6, val_dataset6 = Graph_Classification_Dataset(data_path+task_list[5] + '.csv', smiles_field='smiles',
                                                                            label_field=label, seed=seed,
                                                                            batch_size=batch_size, a=len(label),
                                                                            addH=True).get_data()
    train_dataset7, test_dataset7, val_dataset7 = Graph_Classification_Dataset(data_path+task_list[6] + '.csv', smiles_field='smiles',
                                                                            label_field=label, seed=seed,
                                                                            batch_size=batch_size, a=len(label),
                                                                            addH=True).get_data()
    train_dataset8, test_dataset8, val_dataset8 = Graph_Classification_Dataset(data_path+task_list[7] + '.csv', smiles_field='smiles',
                                                                            label_field=label, seed=seed,
                                                                            batch_size=batch_size, a=len(label),
                                                                            addH=True).get_data()
    test_dataset_list = [test_dataset1,test_dataset2,test_dataset3,test_dataset4,test_dataset5,test_dataset6,
                         test_dataset7,test_dataset8]
    # 对每个任务创建模型，具有共享的编码层
    x, adjoin_matrix, y = next(iter(train_dataset1.take(1)))
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]

    model_origin = EncoderModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size,
                         a=len(label),
                         dense_dropout=dense_dropout)
    model1 = CostomModel(model_origin,dense_dropout)
    model2 = CostomModel(model_origin,dense_dropout)
    model3 = CostomModel(model_origin,dense_dropout)
    model4 = CostomModel(model_origin,dense_dropout)
    model5 = CostomModel(model_origin,dense_dropout)
    model6 = CostomModel(model_origin,dense_dropout)
    model7 = CostomModel(model_origin,dense_dropout)
    model8 = CostomModel(model_origin,dense_dropout)
    model_list= [model1,model2,model3,model4,model5,model6,model7,model8]
    if pretraining:
        temp = BertModel(num_layers=num_layers, d_model=d_model,
                         dff=dff, num_heads=num_heads, vocab_size=vocab_size)

        pred = temp(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)
        temp.load_weights(
            arch['path'] + '/bert_weights{}_{}.h5'.format(arch['name'], trained_epoch))
        temp.encoder.save_weights(
            arch['path'] + '/bert_weights_encoder{}_{}.h5'.format(arch['name'], trained_epoch))
        del temp

        pred = model_origin(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)
        model_origin.encoder.load_weights(
            arch['path'] + '/bert_weights_encoder{}_{}.h5'.format(arch['name'], trained_epoch))
        print('load_wieghts')
    # 定义优化器和损失
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    auc = -10
    stopping_monitor = 0
    for epoch in range(200):
        loss1 = train_one_epoch(model1,train_dataset1,optimizer=optimizer,loss_object=loss_object)
        loss2 = train_one_epoch(model2,train_dataset2,optimizer=optimizer,loss_object=loss_object)
        loss3 = train_one_epoch(model3,train_dataset3,optimizer=optimizer,loss_object=loss_object)
        loss4 = train_one_epoch(model4,train_dataset4,optimizer=optimizer,loss_object=loss_object)
        loss5 = train_one_epoch(model5,train_dataset5,optimizer=optimizer,loss_object=loss_object)
        loss6 = train_one_epoch(model6,train_dataset6,optimizer=optimizer,loss_object=loss_object)
        loss7 = train_one_epoch(model7,train_dataset7,optimizer=optimizer,loss_object=loss_object)
        loss8 = train_one_epoch(model8,train_dataset8,optimizer=optimizer,loss_object=loss_object)
        loss = np.array([loss1.numpy().item(),loss2.numpy().item(),loss3.numpy().item(),loss4.numpy().item(),
                         loss5.numpy().item(),loss6.numpy().item(),loss7.numpy().item(),loss8.numpy().item()]).mean()
        print('epoch: ', epoch, 'loss: {:.4f}'.format(loss))
        auc1 = validate(model1,val_dataset1)
        auc2 = validate(model2,val_dataset2)
        auc3 = validate(model3,val_dataset3)
        auc4 = validate(model4,val_dataset4)
        auc5 = validate(model5,val_dataset5)
        auc6 = validate(model6,val_dataset6)
        auc7 = validate(model7,val_dataset7)
        auc8 = validate(model8,val_dataset8)

        auc_new = np.array([auc1,auc2,auc3,auc4,auc5,auc6,auc7,auc8]).mean()


        print('val auc:{:.4f}'.format(auc_new))
        if auc_new > auc:
            auc = auc_new
            stopping_monitor = 0
            for task,model in (zip(task_list,model_list)):
                model.save_weights('classification_weights/{}_{}.h5'.format(task, seed))
            print('save model weights')
        else:
            stopping_monitor += 1
        print('best val auc: {:.4f}'.format(auc))
        if stopping_monitor > 0:
            print('stopping_monitor:', stopping_monitor)
        if stopping_monitor > 30:
            break
    auc_list=[]
    y_tr_dict={}
    y_pr_dict={}
    for task,model,test_dataset in (zip(task_list,model_list,test_dataset_list)):
        auc_single_list,y_tr_single_dict,y_pr_single_dict = test_result(model,task,test_dataset,seed)
        auc_list.append(auc_single_list[0])
        y_tr_dict[task] = y_tr_single_dict[0]
        y_pr_dict[task] = y_pr_single_dict[0]


    test_auc = np.nanmean(auc_list)
    print('test auc:{:.4f}'.format(test_auc))

    return auc, test_auc, auc_list, y_tr_dict, y_pr_dict


space = {"dense_dropout": hp.quniform("dense_dropout", 0, 0.5, 0.05),
         "learning_rate": hp.loguniform("learning_rate", np.log(3e-5), np.log(15e-5)),
         "batch_size": hp.choice("batch_size", [8, 16, 32, 48, 64]),
         "num_heads": hp.choice("num_heads", [4, 8]),
         }


def hy_main(args):
    auc_list = []
    test_auc_list = []
    test_all_auc_list = []
    x = 0
    for seed in [1, 2, 3]:
        print(seed)
        auc, test_auc, a_list, y_true, y_pred = main(seed, args)
        auc_list.append(auc)
        test_auc_list.append(test_auc)
        test_all_auc_list.append(a_list)
        x += test_auc
    auc_list.append(np.mean(auc_list))
    test_auc_list.append(np.mean(test_auc_list))
    print(auc_list)
    print(test_auc_list)
    print(test_all_auc_list)
    print(args["dense_dropout"])
    print(args["learning_rate"])
    print(args["batch_size"])
    print(args["num_heads"])
    return -x / 3



def score(y_true, y_pred):
    auc_roc_score = roc_auc_score(y_true, y_pred)
    prec, recall, _ = precision_recall_curve(y_true, y_pred)
    prauc = auc(recall, prec)
    y_pred_print = [round(y, 0) for y in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_print).ravel()
    se = tp / (tp + fn)
    sp = tn / (tn + fp)  # 也是R
    q = (tp + tn) / (tp + fn + tn + fp)
    mcc = (tp * tn - fn * fp) / math.sqrt((tp + fn) * (tp + fp) * (tn + fn) * (tn + fp))
    P = tp / (tp + fp)
    F1 = (P * se * 2) / (P + se)
    BA = (se + sp) / 2
    return tp, tn, fn, fp, se, sp, mcc, q, auc_roc_score, F1, BA, prauc



def score_all(args):
    idx = 0
    for seed in [1, 2, 3]:
        print(seed)
        _, _, _, y_true_final, y_pred_final = main(seed, args)
        if idx == 0:
            writer.writerow(['tasks', 'tp', 'tn', 'fn',
                             'fp', 'se', 'sp', 'mcc', 'q', 'auc_roc_score', 'F1', 'BA', 'prauc'])
            idx = 1
        for task in task_list:
            tp, tn, fn, fp, se, sp, mcc, q, auc_roc_score, F1, BA, prauc = score(y_true_final[task], y_pred_final[task])
            writer.writerow([task, tp, tn, fn, fp, se, sp, mcc, q, auc_roc_score, F1, BA, prauc])
    return None


if __name__ == '__main__':
    data_path = './data/'
    task_list = ['ABTS','DPPH','FRAP','HRSA','MCA','NOSA','ORAC','SOD']
    if not os.path.exists('classification_weights'):
        os.makedirs('classification_weights')
    best = fmin(hy_main, space, algo=tpe.suggest, max_evals=10)
    print(best)
    best_dict = {}
    a = [8, 16, 32, 48, 64]
    b = [4, 8]
    best_dict["dense_dropout"] = best["dense_dropout"]
    best_dict["learning_rate"] = best["learning_rate"]
    best_dict["batch_size"] = a[best["batch_size"]]
    best_dict["num_heads"] = b[best["num_heads"]]
    print(best_dict)

    with open('test' + '_results.csv', 'a+', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        score_all(best_dict)


