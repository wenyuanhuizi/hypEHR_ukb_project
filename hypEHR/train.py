import os
import time
import torch
import pickle
import argparse

import numpy as np
import os.path as osp
import scipy.sparse as sp
import torch_sparse
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm, trange

from layers import *
from models import *
from preprocessing import *

from convert_dataset_to_pygDataset import dataset_Hypergraph
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import f1_score

from copy import deepcopy
import random

def parse_method(args, data):
    model = None
    if args.dname == 'ukb':
        model = SetGNN(args, data)
    return model

# random seed 
def seed_everything(seed=0):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False   
    os.environ["PYTHONHASHSEED"] = str(seed)

@torch.no_grad()
# def evaluate(model, data, split_idx, eval_func, epoch, method, dname, args):
#     valid_acc_gf = valid_auc_gf = valid_aupr_gf = valid_f1_macro_gf = \
#     test_acc_gf = test_auc_gf = test_aupr_gf = test_f1_macro_gf = \
#     valid_acc_gcf = valid_auc_gcf = valid_aupr_gcf = valid_f1_macro_gcf = \
#     test_acc_gcf = test_auc_gcf = test_aupr_gcf = test_f1_macro_gcf = 0

#     model.eval()

#     # use original graph (G)
#     out_score_g_logits, edge_feat, node_feat, weight_tuple = model(data)
#     out_g = torch.sigmoid(out_score_g_logits)

#     valid_acc_g, valid_auc_g, valid_aupr_g, valid_f1_macro_g = eval_func(
#         data.y[split_idx['valid']], out_g[split_idx['valid']],
#         epoch, method, dname, args, mode='dev_g', threshold=args.threshold)
#     test_acc_g, test_auc_g, test_aupr_g, test_f1_macro_g = eval_func(data.y[split_idx['test']],
#                                                                      out_g[split_idx['test']],
#                                                                      epoch, method, dname, args,
#                                                                      mode='test_g',
#                                                                      threshold=args.threshold)

#     if args.vanilla:
#         edge_index = weight_tuple[0]
#         edge_weight = weight_tuple[1].reshape(-1)
#         num_hyperedges = data.num_hyperedges

#     else:
#         # get the edge weight
#         view_learner.eval()
#         weight_logits = view_learner(data, device)

#         # gumbel softmax
#         bias = 0.0 + 0.0001  # If bias is 0, we run into problems
#         eps = (bias - (1 - bias)) * torch.rand(weight_logits.size()) + (1 - bias)
#         gate_inputs = torch.log(eps) - torch.log(1 - eps)
#         gate_inputs = gate_inputs.to(device)
#         gate_inputs = (gate_inputs + weight_logits) / args.temperature
#         aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()

#         # use factual graph (G')
#         out_score_gf_logits, _, _, _ = model(data, edge_weight=aug_edge_weight)  # use augmented graph
#         out_gf = torch.sigmoid(out_score_gf_logits)

#         valid_acc_gf, valid_auc_gf, valid_aupr_gf, valid_f1_macro_gf = eval_func(
#             data.y[split_idx['valid']],
#             out_gf[split_idx['valid']],
#             epoch, method, dname, args, mode='dev_gf', threshold=args.threshold)
#         test_acc_gf, test_auc_gf, test_aupr_gf, test_f1_macro_gf = eval_func(
#             data.y[split_idx['test']], out_gf[split_idx['test']],
#             epoch, method, dname, args, mode='test_gf', threshold=args.threshold)

#         # use counterfactual graph (G-G')
#         out_score_gcf_logits, _, _, _ = model(data, edge_weight=1 - aug_edge_weight)  # use augmented graph
#         out_gcf = torch.sigmoid(out_score_gcf_logits)

#         valid_acc_gcf, valid_auc_gcf, valid_aupr_gcf, valid_f1_macro_gcf = eval_func(
#             data.y[split_idx['valid']],
#             out_gcf[split_idx['valid']],
#             epoch, method, dname, args,
#             mode='dev_gcf', threshold=args.threshold)
#         test_acc_gcf, test_auc_gcf, test_aupr_gcf, test_f1_macro_gcf = eval_func(
#             data.y[split_idx['test']], out_gcf[split_idx['test']],
#             epoch, method, dname, args,
#             mode='test_gcf', threshold=args.threshold)

#         if epoch == args.epochs - 1:
#             get_subset_ranking(aug_edge_weight, data.edge_index, data.num_hyperedges, args)

#     return valid_acc_g, valid_auc_g, valid_aupr_g, valid_f1_macro_g, \
#            test_acc_g, test_auc_g, test_aupr_g, test_f1_macro_g, \
#            valid_acc_gf, valid_auc_gf, valid_aupr_gf, valid_f1_macro_gf, \
#            test_acc_gf, test_auc_gf, test_aupr_gf, test_f1_macro_gf, \
#            valid_acc_gcf, valid_auc_gcf, valid_aupr_gcf, valid_f1_macro_gcf, \
#            test_acc_gcf, test_auc_gcf, test_aupr_gcf, test_f1_macro_gcf

def evaluate(model, data, split_idx, eval_func, epoch, method, dname, args):
    valid_acc_gf = valid_auc_gf = valid_aupr_gf = valid_f1_macro_gf = \
    test_acc_gf = test_auc_gf = test_aupr_gf = test_f1_macro_gf = \
    valid_acc_gcf = valid_auc_gcf = valid_aupr_gcf = valid_f1_macro_gcf = \
    test_acc_gcf = test_auc_gcf = test_aupr_gcf = test_f1_macro_gcf = 0

    model.eval()

    # Use original graph (G)
    out_score_g_logits, edge_feat, node_feat, weight_tuple = model(data)

    # Extract probabilities for the positive class (index 1)
    out_g = torch.sigmoid(out_score_g_logits[:, 1])

    # Ensure consistent dimensions for y_true and y_pred
    valid_idx = split_idx['valid']
    test_idx = split_idx['test']
    valid_y_true = data.y[valid_idx].view(-1)  # Reshape to 1D
    valid_y_pred = out_g[valid_idx]
    test_y_true = data.y[test_idx].view(-1)  # Reshape to 1D
    test_y_pred = out_g[test_idx]

    assert valid_y_true.shape == valid_y_pred.shape, f"Validation mismatch: {valid_y_true.shape} vs {valid_y_pred.shape}"
    assert test_y_true.shape == test_y_pred.shape, f"Test mismatch: {test_y_true.shape} vs {test_y_pred.shape}"

    valid_acc_g, valid_auc_g, valid_aupr_g, valid_f1_macro_g = eval_func(
        valid_y_true, valid_y_pred, epoch, method, dname, args, mode='dev_g', threshold=args.threshold)
    test_acc_g, test_auc_g, test_aupr_g, test_f1_macro_g = eval_func(
        test_y_true, test_y_pred, epoch, method, dname, args, mode='test_g', threshold=args.threshold)

    if args.vanilla:
        edge_index = weight_tuple[0]
        edge_weight = weight_tuple[1].reshape(-1)
        num_hyperedges = data.num_hyperedges
    else:
        # Get the edge weight
        view_learner.eval()
        weight_logits = view_learner(data, device)

        # Gumbel softmax
        bias = 0.0 + 0.0001
        eps = (bias - (1 - bias)) * torch.rand(weight_logits.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(device)
        gate_inputs = (gate_inputs + weight_logits) / args.temperature
        aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()

        # Use factual graph (G')
        out_score_gf_logits, _, _, _ = model(data, edge_weight=aug_edge_weight)
        out_gf = torch.sigmoid(out_score_gf_logits[:, 1])

        valid_y_pred_gf = out_gf[valid_idx]
        test_y_pred_gf = out_gf[test_idx]
        assert valid_y_true.shape == valid_y_pred_gf.shape, f"Validation mismatch (GF): {valid_y_true.shape} vs {valid_y_pred_gf.shape}"
        assert test_y_true.shape == test_y_pred_gf.shape, f"Test mismatch (GF): {test_y_true.shape} vs {test_y_pred_gf.shape}"

        valid_acc_gf, valid_auc_gf, valid_aupr_gf, valid_f1_macro_gf = eval_func(
            valid_y_true, valid_y_pred_gf, epoch, method, dname, args, mode='dev_gf', threshold=args.threshold)
        test_acc_gf, test_auc_gf, test_aupr_gf, test_f1_macro_gf = eval_func(
            test_y_true, test_y_pred_gf, epoch, method, dname, args, mode='test_gf', threshold=args.threshold)

        # Use counterfactual graph (G-G')
        out_score_gcf_logits, _, _, _ = model(data, edge_weight=1 - aug_edge_weight)
        out_gcf = torch.sigmoid(out_score_gcf_logits[:, 1])

        valid_y_pred_gcf = out_gcf[valid_idx]
        test_y_pred_gcf = out_gcf[test_idx]
        assert valid_y_true.shape == valid_y_pred_gcf.shape, f"Validation mismatch (GCF): {valid_y_true.shape} vs {valid_y_pred_gcf.shape}"
        assert test_y_true.shape == test_y_pred_gcf.shape, f"Test mismatch (GCF): {test_y_true.shape} vs {test_y_pred_gcf.shape}"

        valid_acc_gcf, valid_auc_gcf, valid_aupr_gcf, valid_f1_macro_gcf = eval_func(
            valid_y_true, valid_y_pred_gcf, epoch, method, dname, args, mode='dev_gcf', threshold=args.threshold)
        test_acc_gcf, test_auc_gcf, test_aupr_gcf, test_f1_macro_gcf = eval_func(
            test_y_true, test_y_pred_gcf, epoch, method, dname, args, mode='test_gcf', threshold=args.threshold)

        if epoch == args.epochs - 1:
            get_subset_ranking(aug_edge_weight, data.edge_index, data.num_hyperedges, args)

    return valid_acc_g, valid_auc_g, valid_aupr_g, valid_f1_macro_g, \
           test_acc_g, test_auc_g, test_aupr_g, test_f1_macro_g, \
           valid_acc_gf, valid_auc_gf, valid_aupr_gf, valid_f1_macro_gf, \
           test_acc_gf, test_auc_gf, test_aupr_gf, test_f1_macro_gf, \
           valid_acc_gcf, valid_auc_gcf, valid_aupr_gcf, valid_f1_macro_gcf, \
           test_acc_gcf, test_auc_gcf, test_aupr_gcf, test_f1_macro_gcf



def get_subset_ranking(edge_weight, edge_index, num_hyperedges, args):
    edge_index_clone = edge_index.clone().detach().to('cpu').numpy()
    edge_weight_clone = edge_weight.reshape(1, -1).clone().detach().to('cpu').numpy()
    index_weight_concat = np.concatenate((edge_index_clone, edge_weight_clone), axis=0)

    index_weight_concat = index_weight_concat[:, index_weight_concat[2, :].argsort()[::-1]]

    edge_dict = {}
    for i in range(num_hyperedges):
        edge_dict[i] = []
    for i in tqdm(range(index_weight_concat.shape[1])):
        if index_weight_concat[1][i] < num_hyperedges:  # self loop
            edge_dict[index_weight_concat[1][i]].append(index_weight_concat[0][i])
    sorted_edge_dict = dict(sorted(edge_dict.items()))

    vanilla = ""
    if args.vanilla: vanilla = "_vanilla"
    with open(f"outputs/deleted_output_{args.method}{vanilla}_{args.dname}.txt", "w") as f_del, \
            open(f"outputs/remained_output_{args.method}{vanilla}_{args.dname}.txt", "w") as f_rem:
        for hyperedge in list(sorted_edge_dict.values()):
            rem_size = int(len(hyperedge) * args.remain_percentage)
            if rem_size < 5 and len(hyperedge) >= 5:
                rem_size = 5
            elif rem_size < 5 and len(hyperedge) < 5:
                rem_size = len(hyperedge)
            remain = [str(int(x)) for x in hyperedge[:rem_size]]
            f_rem.write(",".join(remain))
            f_rem.write('\n')
            delete = [str(int(x)) for x in hyperedge[rem_size:]]
            f_del.write(",".join(delete))
            f_del.write('\n')

def eval_ukb(y_true, y_pred, epoch, method, dname, args, mode='dev', threshold=0.5):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    pred = np.array(y_pred > threshold).astype(int)
    correct = (pred == y_true)
    accuracy = correct.sum() / correct.size
    f1_macro = f1_score(y_true.reshape(-1), pred.reshape(-1), average="macro")
    roc_auc = roc_auc_score(y_true.reshape(-1), y_pred.reshape(-1))
    aupr = average_precision_score(y_true.reshape(-1), y_pred.reshape(-1))

    return accuracy, roc_auc, aupr, f1_macro

if __name__ == '__main__':
    # os.chdir('/local/scratch3/yxie289/EHR/benchmark/hypehr/src')  # working dir
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prop', type=float, default=0.7)
    parser.add_argument('--valid_prop', type=float, default=0.1)
    parser.add_argument('--dname', default='ukb')
    parser.add_argument('--method', default='AllSetTransformer')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--cuda', default='0', type=str)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--wd', default=1e-3, type=float)
    parser.add_argument('--view_lr', default=1e-2, type=float)
    parser.add_argument('--view_wd', default=1e-3, type=float)
    # How many layers of full NLConvs
    parser.add_argument('--All_num_layers', default=2, type=int)
    parser.add_argument('--MLP_num_layers', default=2,
                        type=int)  # How many layers of encoder
    parser.add_argument('--MLP_hidden', default=48,
                        type=int)  # Encoder hidden units
    parser.add_argument('--Classifier_num_layers', default=2,
                        type=int)  # How many layers of decoder
    parser.add_argument('--Classifier_hidden', default=64,
                        type=int)  # Decoder hidden units
    parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean'])
    # ['all_one','deg_half_sym']
    parser.add_argument('--normtype', default='all_one')
    parser.add_argument('--add_self_loop', action='store_false')
    # NormLayer for MLP. ['bn','ln','None']
    parser.add_argument('--normalization', default='ln')
    parser.add_argument('--num_features', default=0, type=int)  # Placeholder
    parser.add_argument('--num_labels', default=2, type=int)  # set the default for now
    parser.add_argument('--num_nodes', default=3957, type=int)  
    # 'all' means all samples have labels, otherwise it indicates the first [num_labeled_data] rows that have the labels
    parser.add_argument('--num_labeled_data', default='all', type=str)
    parser.add_argument('--feature_dim', default=128, type=int)  # feature dim of learnable node feat
    parser.add_argument('--LearnFeat', action='store_true')
    # whether the he contain self node or not
    parser.add_argument('--PMA', action='store_true')
    #     Args for Attentions
    parser.add_argument('--heads', default=1, type=int)  # Placeholder
    parser.add_argument('--output_ ', default=1, type=int)  # Placeholder

    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--view_alpha', type=float, default=0.5)
    parser.add_argument('--view_lambda', type=float, default=5)
    parser.add_argument('--model_lambda', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=1)  # 0.5 | 5; temperature for gumbel softmax

    parser.add_argument('--vanilla', action='store_true')
    parser.add_argument('--remain_percentage', default=0.3, type=float)
    parser.add_argument('--rand_seed', default=0, type=int)
    parser.add_argument('--random_split', action='store_true', default=False)
    parser.set_defaults(PMA=True)
    parser.set_defaults(add_self_loop=True)
    parser.set_defaults(LearnFeat=False)

    args = parser.parse_args()
    
    seed_everything(args.rand_seed) 

    existing_dataset = ['ukb']

    synthetic_list = ['ukb']

    dname = args.dname
    p2raw = '../data'
    dataset = dataset_Hypergraph(name=dname, root='../data/pyg_data/hypergraph_dataset/',
                                 p2raw=p2raw, num_nodes=args.num_nodes)
    data = dataset.data
    args.num_features = dataset.num_features
    if args.dname in ['ukb']:
        # Shift the y label to start with 0
        data.y = data.y - data.y.min()
    if not hasattr(data, 'n_x'):
        data.n_x = torch.tensor([data.x.shape[0]])
    if not hasattr(data, 'num_hyperedges'):
        # note that we assume the he_id is consecutive.
        data.num_hyperedges = torch.tensor(
            [data.edge_index[0].max() - data.n_x[0] + 1])

    if args.method == 'AllSetTransformer':
        data = ExtractV2E(data)
        data.totedges = data.edge_index.size(1)
        data = norm_contruction(data, option=args.normtype)

    model = parse_method(args, data)
    view_learner = ViewLearner(parse_method(args, data), args.MLP_hidden)
    # put things to device
    if args.cuda != '-1':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    #     Get splits
    # split_idx = rand_train_test_idx(data.y, train_prop=args.train_prop, valid_prop=args.valid_prop, rand_seed=args.rand_seed)
    
    # train_idx = split_idx['train'].to(device)
    
    split_idx = rand_train_test_idx(data.y, train_prop=args.train_prop, valid_prop=args.valid_prop, rand_seed=args.rand_seed, random_split=args.random_split)
    train_idx = split_idx['train']
    valid_idx = split_idx['valid']
    test_idx = split_idx['test']

    model, view_learner, data = model.to(device), view_learner.to(device), data.to(device)

    criterion = nn.BCELoss()

    model.train()
    model.reset_parameters()
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    view_optimizer = torch.optim.Adam(view_learner.parameters(), lr=args.view_lr, weight_decay=args.view_wd)

    with open(f'../data/hyperedges-{args.dname}.txt', 'r') as f:
        total_edges = []
        maxlen = 0
        for lines in f:
            line = lines.strip().split(',')
            line = list(map(int, line))
            if len(line) > maxlen:
                maxlen = len(line)
            total_edges.append(line)
        total_edges_padded = []
        for edge in total_edges:
            total_edges_padded.append(edge + [-1] * (maxlen - len(edge)))

    if args.num_labeled_data != 'all':
        N = int(args.num_labeled_data)  # the first x visits have labels
    elif args.num_labeled_data == 'all':
        N = len(total_edges_padded)  # all the samples in cradle have labels
    train_num = int(N * args.train_prop)
    valid_num = int(N * args.valid_prop)
    train_input = torch.LongTensor(total_edges_padded[:train_num]).to(device)
    dev_input = torch.LongTensor(total_edges_padded[train_num:train_num + valid_num]).to(device)
    test_input = torch.LongTensor(total_edges_padded[train_num + valid_num:N]).to(device)

    edge_id_dict = None
    with torch.autograd.set_detect_anomaly(True):
        for epoch in trange(args.epochs):
            if args.vanilla:  # VANILLA - Use attention weight to get an important set for each encounter
                model.train()
                model.zero_grad()

                out_score_logits, _, _, weight_tuple = model(data)
                out = torch.sigmoid(out_score_logits)

                model_loss = criterion(out[train_idx], data.y[train_idx]) + args.view_lambda * torch.mean(
                    weight_tuple[1].reshape(-1))
                model_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                model_optimizer.step()
            else:  # CACHE
                if (epoch + 1) % 50 == 0:
                    args.view_lambda *= 0.5
                """STEP ONE - TRAIN THE LEARNER"""
                view_learner.train()
                view_learner.zero_grad()
                model.eval()

                out_score_logits, out_edge_feat, _, _ = model(data)
                out = torch.sigmoid(out_score_logits)

                weight_logits = view_learner(data, device)

                # gumbel softmax
                # temperature = 1.0
                bias = 0.0 + 0.0001  # If bias is 0, we run into problems
                eps = (bias - (1 - bias)) * torch.rand(weight_logits.size()) + (1 - bias)
                gate_inputs = torch.log(eps) - torch.log(1 - eps)
                gate_inputs = gate_inputs.to(device)
                gate_inputs = (gate_inputs + weight_logits) / args.temperature
                aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()

                # factual prediction
                out_score_f_logits, out_edge_feat_f, _, _ = model(data, edge_weight=aug_edge_weight)
                out_f = torch.sigmoid(out_score_f_logits)

                # regularization - not to drop too many edges
                edge_dropout_prob = 1 - aug_edge_weight
                reg = torch.mean(edge_dropout_prob)

                # counterfactual prediction
                out_score_cf_logits, out_edge_feat_cf, _, _ = model(data, edge_weight=edge_dropout_prob)
                out_cf = torch.sigmoid(out_score_cf_logits)

                # factual loss
                coef = out.detach().clone()
                coef[out >= 0.5] = 1
                coef[out < 0.5] = -1
                loss_f = torch.mean(torch.clamp(torch.add(coef * (0 - out_score_f_logits), args.gamma), min=0))

                # counterfactual loss
                coef = out.detach().clone()
                coef[out >= 0.5] = -1
                coef[out < 0.5] = 1
                loss_cf = torch.mean(torch.clamp(torch.add(coef * (0 - out_score_cf_logits), args.gamma), min=0))

                # factual and counterfactual view loss
                loss = args.view_alpha * loss_f + (1 - args.view_alpha) * loss_cf

                view_loss = loss + args.view_lambda * torch.mean(aug_edge_weight)
                view_loss.backward()
                torch.nn.utils.clip_grad_norm_(view_learner.parameters(), 1)
                view_optimizer.step()

                """STEP TWO - TRAIN THE MAIN MODEL"""
                model.train()
                model.zero_grad()
                view_learner.eval()

                out_score_logits, out_edge_feat, _, _ = model(data)
                out = torch.sigmoid(out_score_logits)

                # learn the edge weight (augmentation policy)
                weight_logits = view_learner(data, device)

                # gumbel softmax
                # temperature = 1.0
                bias = 0.0 + 0.0001  # If bias is 0, we run into problems
                eps = (bias - (1 - bias)) * torch.rand(weight_logits.size()) + (1 - bias)
                gate_inputs = torch.log(eps) - torch.log(1 - eps)
                gate_inputs = gate_inputs.to(device)
                gate_inputs = (gate_inputs + weight_logits) / args.temperature
                aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()

                # factual prediction
                out_score_f_logits, out_edge_feat_f, _, _ = model(data, edge_weight=aug_edge_weight)
                out_f = torch.sigmoid(out_score_f_logits)

                # counterfactual prediction
                edge_dropout_prob = 1 - aug_edge_weight
                out_score_cf_logits, out_edge_feat_cf, _, _ = model(data, edge_weight=edge_dropout_prob)
                out_cf = torch.sigmoid(out_score_cf_logits)

                # factual loss
                coef = out.detach().clone()
                coef[out >= 0.5] = 1
                coef[out < 0.5] = -1
                loss_f = torch.mean(torch.clamp(torch.add(coef * (0 - out_score_f_logits), args.gamma), min=0))

                # counter factual loss
                coef = out.detach().clone()
                coef[out >= 0.5] = -1
                coef[out < 0.5] = 1
                loss_cf = torch.mean(torch.clamp(torch.add(coef * (0 - out_score_cf_logits), args.gamma), min=0))

                # factual and counterfactual view loss
                loss = args.view_alpha * loss_f + (1 - args.view_alpha) * loss_cf

                # model_loss = criterion(out[train_idx-1], data.y[train_idx]) + args.model_lambda * loss
                # Ensure the model outputs a single probability for binary classification
                out = out.squeeze(-1) if out.shape[-1] == 1 else out[:, 1]  # Handle cases where output has two dimensions

                # Ensure the target tensor matches the shape of the model's output
                target = data.y.view_as(out)

                # Compute the binary cross-entropy loss
                model_loss = criterion(out[train_idx], target[train_idx]) + args.model_lambda * loss
                model_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                model_optimizer.step()

                eval_function = eval_ukb
                valid_acc_g, valid_auc_g, valid_aupr_g, valid_f1_macro_g, \
                test_acc_g, test_auc_g, test_aupr_g, test_f1_macro_g, \
                valid_acc_gf, valid_auc_gf, valid_aupr_gf, valid_f1_macro_gf, \
                test_acc_gf, test_auc_gf, test_aupr_gf, test_f1_macro_gf, \
                valid_acc_gcf, valid_auc_gcf, valid_aupr_gcf, valid_f1_macro_gcf, \
                test_acc_gcf, test_auc_gcf, test_aupr_gcf, test_f1_macro_gcf = \
                    evaluate(model, data, split_idx, eval_function, epoch, args.method, args.dname,
                            args)

                fname_dev = ''
                fname_test = ''
                vanilla = ""
                if args.vanilla: vanilla = "_vanilla"
                fname_dev = f'outputs/ukb_dev_{args.method}{vanilla}.txt'
                fname_test = f'outputs/ukb_test_{args.method}{vanilla}.txt'
         
            # dev set
            with open(fname_dev, 'a+', encoding='utf-8') as f:
                f.write(
                    'Epoch: {}, Threshold: {:.2f}, lr: {:.2e}, wd: {:.2e}, view_lr: {:.2e}, view_wd: {:.2e}, '
                    'view_alpha:{:.2f}, view_lambda:{:.3f}, model_lambda:{:.3f}, gamma:{:.2f}, ACC_G: {:.5f}, '
                    'AUC_G: {:.5f}, AUPR_G: {:.5f}, F1_MACRO_G: {:.5f}, ACC_Gf: {:.5f}, AUC_Gf: {:.5f}, AUPR_Gf: {:.5f}, F1_MACRO_Gf: {:.5f}, '
                    'ACC_Gcf: {:.5f}, AUC_Gcf: {:.5f}, AUPR_Gcf: {:.5f}, F1_MACRO_Gcf: {:.5f}\n '
                        .format(epoch + 1, args.threshold, args.lr, args.wd, args.view_lr, args.view_wd,
                                args.view_alpha, args.view_lambda, args.model_lambda, args.gamma, valid_acc_g,
                                valid_auc_g, valid_aupr_g, valid_f1_macro_g, valid_acc_gf, valid_auc_gf, valid_aupr_gf,
                                valid_f1_macro_gf,
                                valid_acc_gcf, valid_auc_gcf, valid_aupr_gcf, valid_f1_macro_gcf))
            # test set
            with open(fname_test, 'a+', encoding='utf-8') as f:
                f.write(
                    'Epoch: {}, Threshold: {:.2f}, lr: {:.2e}, wd: {:.2e}, view_lr: {:.2e}, view_wd: {:.2e}, '
                    'view_alpha:{:.2f}, view_lambda:{:.3f}, model_lambda:{:.3f}, gamma:{:.2f}, ACC_G: {:.5f}, '
                    'AUC_G: {:.5f}, AUPR_G: {:.5f}, F1_MACRO_G: {:.5f}, ACC_Gf: {:.5f}, AUC_Gf: {:.5f}, AUPR_Gf: {:.5f}, F1_MACRO_Gf: {:.5f}, '
                    'ACC_Gcf: {:.5f}, AUC_Gcf: {:.5f}, AUPR_Gcf: {:.5f}, F1_MACRO_Gcf: {:.5f}\n'
                        .format(epoch + 1, args.threshold, args.lr, args.wd, args.view_lr, args.view_wd,
                                args.view_alpha, args.view_lambda, args.model_lambda, args.gamma, test_acc_g,
                                test_auc_g, test_aupr_g, test_f1_macro_g, test_acc_gf, test_auc_gf, test_aupr_gf,
                                test_f1_macro_gf, test_acc_gcf, test_auc_gcf, test_aupr_gcf, test_f1_macro_gcf))

    
    # # get the generated embedding
    # model.eval()
    # embeddings = model.get_embedding(data)
    # embeddings = embeddings.cpu().detach().numpy()
    # np.save('embeddings.npy', embeddings)
    print('All done! Exit python code')
    quit()
