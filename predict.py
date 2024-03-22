import tensorflow as tf
import pandas as pd
import os
from dataset_scoffold_random_dc import Inference_Dataset
from muti_model import EncoderModel, CostomModel
import argparse


def predict(model_path,model_list,smiles_list,output):
    # 加载数据
    inference_dataset = Inference_Dataset(smiles_list,addH=True,padded_batch=512).get_data()
    x1, adjoin_matrix1, _, _ = next(iter(inference_dataset.take(1)))
    seq = tf.cast(tf.math.equal(x1, 0), tf.float32)
    mask1 = seq[:, tf.newaxis, tf.newaxis, :]
    # 初始化模型结构
    medium = {'name':'Medium','num_layers': 6, 'num_heads': 4, 'd_model': 256,'path':'medium_weights','addH':True}
    arch = medium  ## small 3 4 128   medium: 6  6  256     large:  12 8 516
    num_layers = arch['num_layers']
    num_heads = arch['num_heads']
    d_model = arch['d_model']
    label = ['Label']
    dff = d_model * 2
    vocab_size = 18
    dense_dropout=0
    model_origin = EncoderModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size,
                            a=len(label),
                            dense_dropout=dense_dropout)
    model = CostomModel(model_origin,dense_dropout)
    pred = model(x1,mask=mask1,training=False,adjoin_matrix=adjoin_matrix1)
    
    # 替换成对应模型预测
    for model_file in model_list:
        task_name = model_file[:-3]
        model.load_weights(model_path+model_file)
        y_preds=[]
        for smi in smiles_list:
            inference_dataset = Inference_Dataset([smi],addH=True,padded_batch=512).get_data()
            x, adjoin_matrix, _, _ = next(iter(inference_dataset.take(1)))
            seq = tf.cast(tf.math.equal(x, 0), tf.float32)
            mask = seq[:, tf.newaxis, tf.newaxis, :]
            pred = model(x,mask=mask,training=False,adjoin_matrix=adjoin_matrix).numpy()[0][0]
            y_preds.append(pred)
        df[task_name]=y_preds
    df.to_csv(output,index=None)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input','-i', type=str, default='none',
                    help='The path of input CSV file to predict.')
    p.add_argument('--smiles','-s', type=str, default='none',
                    help='The SMILES string of molecule.')
    p.add_argument('--output','-o', type=str,default='output.csv',
                    help='The path of output CSV file.')
    p.add_argument('--model_path','-m', type=str,default='./model/best/',
                    help='The path of h5 models.')
    p.add_argument('--assays_list','-l', nargs='+', default=["all"],
                    help='Which antioxidant assays are predicted.')
    args = p.parse_args()
    input = args.input
    output = args.output
    model_path = args.model_path
    model_list = []
    for i in args.assays_list:
        if i=='all':
            model_list = os.listdir(model_path)
            break
        else:
            model_list.append(i+'.h5')
    if args.input!='none':
        df = pd.read_csv(input)
        smiles_list = df['SMILES'].values
    if args.smiles!='none':
        smiles_list = [args.smiles]
        df = pd.DataFrame({'SMILES':[args.smiles]})
    predict(model_path=model_path,model_list=model_list,smiles_list=smiles_list,output=output)


    