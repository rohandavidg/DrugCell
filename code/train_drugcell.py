import sys
import os
import numpy as np
import torch
import pandas as pd
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import util
from torchmetrics.functional import mean_absolute_error
from scipy.stats import spearmanr
import sklearn
import sklearn.metrics
from util import *
from drugcell_NN import *
import argparse
import numpy as np
import time
from time import time
import logging

#set up logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class Timer:
    """
    Measure runtime.
    """
    def __init__(self):
        self.start = time()

    def timer_end(self):
        self.end = time()
        time_diff = self.end - self.start
        return time_diff

    def display_timer(self, print_fn=print):
        time_diff = self.timer_end()
        if (time_diff)//3600 > 0:
            print_fn("Runtime: {:.1f} hrs".format( (time_diff)/3600) )
        else:
            print_fn("Runtime: {:.1f} mins".format( (time_diff)/60) )

def calc_mae(y_true, y_pred):
    return sklearn.metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred)

def calc_r2(y_true, y_pred):
    target_mean = torch.mean(y_pred)
    ss_tot = torch.sum((y_pred - target_mean) ** 2)
    ss_res = torch.sum((y_pred - y_true) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def calc_pcc(x, y):
    xx = x - torch.mean(x)
    yy = y - torch.mean(y)
    return torch.sum(xx*yy) / (torch.norm(xx, 2)*torch.norm(yy,2))


# build mask: matrix (nrows = number of relevant gene set, ncols = number all genes)
# elements of matrix are 1 if the corresponding gene is one of the relevant genes
def create_term_mask(term_direct_gene_map, gene_dim, CUDA_ID):
    term_mask_map = {}
    for term, gene_set in term_direct_gene_map.items():
        mask = torch.zeros(len(gene_set), gene_dim)
        for i, gene_id in enumerate(gene_set):
            mask[i, gene_id] = 1
            mask_gpu = torch.autograd.Variable(mask.cuda(CUDA_ID))
            term_mask_map[term] = mask_gpu
    return term_mask_map


def train_model(root, term_size_map, term_direct_gene_map, dG, train_data,
                gene_dim, drug_dim, model_save_folder, train_epochs,
                batch_size, learning_rate, num_hiddens_genotype, num_hiddens_drug,
                num_hiddens_final, cell_features, drug_features, wd):

    epoch_start_time = time()
    best_model = 0
    max_corr = 0
    logger = logging.getLogger('DrugCell')
    # dcell neural network
    model = drugcell_nn(term_size_map, term_direct_gene_map, dG, gene_dim,
                        drug_dim, root, num_hiddens_genotype, num_hiddens_drug, num_hiddens_final)

    device = torch.device("cuda")
    model.to(device)
    model.cuda(CUDA_ID)
    train_feature, train_label, test_feature, test_label = train_data

    train_label_gpu = torch.autograd.Variable(train_label.cuda(CUDA_ID))
    test_label_gpu = torch.autograd.Variable(test_label.cuda(CUDA_ID))
    term_mask_map = create_term_mask(model.term_direct_gene_map, num_genes, CUDA_ID)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.99),
        eps=1e-05, weight_decay=wd)


    optimizer.zero_grad()
#    timer.display_timer()

    # collecting metrics
    scores = {}
    epoch_list = []
    train_loss_list = []
    train_corr_list = []
    test_loss_list = []
    test_corr_list = []
    train_scc_list = []
    test_scc_list = []
    for name, param in model.named_parameters():
        term_name = name.split('_')[0]
        if '_direct_gene_layer.weight' in name:
            param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
        else:
            param.data = param.data * 0.1

    train_loader = du.DataLoader(du.TensorDataset(
        train_feature, train_label), batch_size=batch_size, shuffle=False)
    test_loader = du.DataLoader(du.TensorDataset(
        test_feature, test_label), batch_size=batch_size, shuffle=False)

    for epoch in range(train_epochs):
        model.train()
        epoch_list.append(epoch)
        train_predict = torch.zeros(0, 0).cuda(CUDA_ID)
        train_loss_mean = 0
        t = time()
        for i, (inputdata, labels) in enumerate(train_loader):
            features = build_input_vector(inputdata, cell_features, drug_features)
            cuda_features = torch.autograd.Variable(features.cuda(CUDA_ID))
            cuda_labels = torch.autograd.Variable(labels.cuda(CUDA_ID))

            optimizer.zero_grad()
            aux_out_map, _ = model(cuda_features)

            if train_predict.size()[0] == 0:
                train_predict = aux_out_map['final'].data
            else:
                train_predict = torch.cat([train_predict, aux_out_map['final'].data], dim=0)

            train_loss = 0
            count = 0
            for name, output in aux_out_map.items():
                count += 1
                loss = nn.MSELoss()
                if name == 'final':
                    train_loss += loss(output, cuda_labels)
                else:
                    train_loss += 0.2 * loss(output, cuda_labels)
            train_loss.backward()
            train_loss_mean = train_loss
            for name, param in model.named_parameters():
                if '_direct_gene_layer.weight' not in name:
                    continue
                term_name = name.split('_')[0]
                param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])

            optimizer.step()

        train_loss_list.append(train_loss_mean.cpu().detach().numpy()/len(train_loader))
        logger.info(
            "\t **** TRAINING ****   "
            f"Epoch [{epoch + 1}/epochs], "
            f"loss: {train_loss_mean / len(train_loader):.5f}. "
            f"This took {time() - t:.1f} secs."
        )

        train_corr = pearson_corr(train_predict, train_label_gpu)
        train_corr_list.append(train_corr.cpu().detach().numpy())
        torch.save(model, model_save_folder + '/model_' + str(epoch) + '.pt')

        train_predictions = np.array([p.cpu() for preds in train_predict for p in preds], dtype=float)
        train_predictions = train_predictions[0:len(train_predictions)]
        train_labels = np.array([l.cpu() for label in train_label_gpu for l in label], dtype=float)
        train_scc = spearmanr(train_labels, train_predictions)[0]
        train_scc_list.append(train_scc)

        model.eval()

        test_predict = torch.zeros(0, 0).cuda(CUDA_ID)

        test_loss = 0
        for i, (inputdata, labels) in enumerate(test_loader):
            features = build_input_vector(inputdata, cell_features, drug_features)
            cuda_features = Variable(features.cuda(CUDA_ID))
            aux_out_map, _ = model(cuda_features)
            cuda_labels = torch.autograd.Variable(labels.cuda(CUDA_ID))
            #values = inputdata.cpu().detach().numpy().tolist()
            #keys = [i for i in feature_dict for x in values if feature_dict[i] == x]
            #tissue = [i.split(';')[0] for i in keys]
            #drug = [i.split(';')[1] for i in keys]
            loss = nn.MSELoss()
            if test_predict.size()[0] == 0:
                test_predict = aux_out_map['final'].data
                print(test_predict.shape, cuda_labels.shape)
                loss_a = loss(test_predict, cuda_labels)
                test_loss += loss_a.item()
            else:
                test_predict = torch.cat([test_predict, aux_out_map['final'].data], dim=0)
                loss_a = loss(test_predict, cuda_labels)
                test_loss += loss_a.item()
        logger.info(
            "\t **** TEST ****   "
            f"Epoch [{epoch + 1}/epochs], "
            f"loss: {test_loss / len(test_loader):.5f}. "
            f"This took {time() - t:.1f} secs."
        )
        predictions = np.array(
            [p.cpu() for preds in test_predict for p in preds], dtype=float)
        predictions = predictions[0:len(predictions)]
        labels = np.array([l.cpu()
                          for label in labels for l in label], dtype=float)
        labels = labels[0:len(labels)]
        test_pearson_a = calc_pcc(torch.Tensor(
            predictions), torch.Tensor(labels))
        test_spearman_a = spearmanr(labels, predictions)[0]
        test_mean_absolute_error = sklearn.metrics.mean_absolute_error(
            y_true=labels, y_pred=predictions)
        test_r2 = sklearn.metrics.r2_score(y_true=labels, y_pred=predictions)
        test_rmse_a = np.sqrt(np.mean((predictions - labels)**2))
        test_loss_a = test_loss / len(test_loader)
        epoch_end_time = time()
        test_loss_a = test_loss/len(test_loader)
        test_loss_list.append(test_loss_a)
        test_corr_list.append(test_pearson_a.cpu().detach().numpy())
        test_scc_list.append(test_spearman_a)
        if epoch == 0:
            min_test_loss = test_loss_a
            scores['test_loss'] = min_test_loss
            scores['test_pcc'] = test_pearson_a.cpu().detach().numpy().tolist()
            scores['test_MSE'] = test_mean_absolute_error
            scores['test_r2'] = test_r2
            scores['test_scc'] = test_spearman_a
        if test_loss_a < min_test_loss:
            min_test_loss = test_loss_a
            scores['test_loss'] = min_test_loss
            scores['test_pcc'] = test_pearson_a.cpu().detach().numpy().tolist()
            scores['test_MSE'] = test_mean_absolute_error
            scores['test_r2'] = test_r2
            scores['test_scc'] = test_spearman_a
        if test_spearman_a >= max_corr:
            max_corr = test_spearman_a
            best_model = epoch
            pred = pd.DataFrame(
                {"True": labels, "Pred": predictions}).reset_index()
            pred_fname = str(model_save_folder+'/results/test_pred.csv')
            pred.to_csv(pred_fname, index=False)
        epoch_start_time = epoch_end_time
    torch.save(model, model_save_folder + '/model_final.pt')
    print("Best performed model (epoch)\t%d" % best_model)
    cols = ['epoch', 'train_loss', 'train_corr',
            'test_loss', 'test_corr', 'test_scc_list']
    epoch_train_test_df = pd.DataFrame(
        columns=cols, index=range(train_epochs))
    epoch_train_test_df['epoch'] = epoch_list
    epoch_train_test_df['train_loss'] = train_loss_list
    epoch_train_test_df['train_corr'] = train_corr_list
    epoch_train_test_df['train_scc'] = train_scc_list
    epoch_train_test_df['test_loss'] = test_loss_list
    epoch_train_test_df['test_corr'] = test_corr_list
    epoch_train_test_df['test_scc'] = test_scc_list
    loss_results_name = 'loss_results.csv'
    epoch_train_test_df.to_csv(loss_results_name, index=False)
    print(scores)
    return scores


parser = argparse.ArgumentParser(description='Train dcell')
parser.add_argument(
    '-onto', help='Ontology file used to guide the neural network', type=str)
parser.add_argument('-train', help='Training dataset', type=str)
parser.add_argument('-test', help='Validation dataset', type=str)
parser.add_argument(
    '-epochs', help='Training epochs for training', type=int, default=300)
parser.add_argument('-lr', help='Learning rate', type=float, default=0.001)
parser.add_argument('-wd', help='Weight Decay', type=float, default=0)
parser.add_argument('-batchsize', help='Batchsize', type=int, default=5000)
parser.add_argument(
    '-modeldir', help='Folder for trained models', type=str, default='MODEL/')
parser.add_argument('-cuda', help='Specify GPU', type=int, default=0)
parser.add_argument('-gene2id', help='Gene to ID mapping file', type=str)
parser.add_argument('-drug2id', help='Drug to ID mapping file', type=str)
parser.add_argument('-cell2id', help='Cell to ID mapping file', type=str)

parser.add_argument('-genotype_hiddens',
                    help='Mapping for the number of neurons in each term in genotype parts', type=int, default=6)
parser.add_argument(
    '-drug_hiddens', help='Mapping for the number of neurons in each layer', type=str, default='100,50,6')
parser.add_argument(
    '-final_hiddens', help='The number of neurons in the top layer', type=int, default=6)

parser.add_argument(
    '-genotype', help='Mutation information for cell lines', type=str)
parser.add_argument(
    '-fingerprint', help='Morgan fingerprint representation for drugs', type=str)

# call functions
opt = parser.parse_args()
torch.set_printoptions(precision=5)

#logging
logger = logging.getLogger('DrugCell')

logger.info("Start data preprocessing...")


# load input data
train_data, cell2id_mapping, drug2id_mapping = prepare_train_data(
    opt.train, opt.test, opt.cell2id, opt.drug2id)
gene2id_mapping = load_mapping(opt.gene2id)

# load cell/drug features
cell_features = np.genfromtxt(opt.genotype, delimiter=',')
drug_features = np.genfromtxt(opt.fingerprint, delimiter=',')

num_cells = len(cell2id_mapping)
num_drugs = len(drug2id_mapping)
num_genes = len(gene2id_mapping)
drug_dim = len(drug_features[0, :])

# load ontology
dG, root, term_size_map, term_direct_gene_map = load_ontology(
    opt.onto, gene2id_mapping)

# load the number of hiddens #######
num_hiddens_genotype = opt.genotype_hiddens

num_hiddens_drug = list(map(int, opt.drug_hiddens.split(',')))

num_hiddens_final = opt.final_hiddens
#####################################

CUDA_ID = opt.cuda


train_model(root, term_size_map, term_direct_gene_map, dG,
            train_data, num_genes, drug_dim, opt.modeldir, opt.epochs,
            opt.batchsize, opt.lr, num_hiddens_genotype, num_hiddens_drug,
            num_hiddens_final, cell_features, drug_features, opt.wd)
