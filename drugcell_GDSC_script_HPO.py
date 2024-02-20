#%% import anndata
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torch.utils.data import Dataset, DataLoader, random_split
from torch.distributions import MultivariateNormal
from codes.utils.util import *
import argparse
import candle
import copy
import tqdm
from pathlib import Path
import logging
import sys
import pandas as pd
import sklearn
import os
import torch.optim as optim
from torchmetrics.functional import mean_absolute_error
from scipy.stats import spearmanr
import time
from time import time

from sklearn.metrics import roc_auc_score


# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

file_path = os.path.dirname(os.path.realpath(__file__))
fdir = Path('__file__').resolve().parent


required = None
additional_definitions = None

# This should be set outside as a user environment variable
os.environ['CANDLE_DATA_DIR'] = os.environ['HOME'] + '/tianshu/DrugCell/hpo_data/'
data_dir = str(fdir) + '/hpo_data/'
print(data_dir)

# additional definitions
additional_definitions = [
    {
        "name": "num_hiddens_genotype",
        "type": int,
        "help": "total number of hidden genotypes",
    },
    {
        "name": "num_hiddens_final",
        "type": int,
        "help": "number of hideen final",
    },
    {   
        "name": "drug_hiddens",
        "type": str,
        "help": "list of hidden drugs",
    },
    {
        "name": "learning_rate",
        "type": float,
        "help": "learning rate of the model",
    },
    {   
        "name": "betas_adam",
        "type": str, 
        "help": "tuple of values ",
    },
    {   
        "name": "cuda",
        "type": int, 
        "help": "CUDA ID",
    },
    {   
        "name": "eps_adam",
        "type": float, 
        "help": "episilon of the optimizer",
    },
    {   
        "name": "direct_gene_weight_param",
        "type": int, 
        "help": "weight of the genes",
    },
    {   
        "name": "batch_size",
        "type": int, 
        "help": "batch size for data processing",
    },
    {   
        "name": "beta_kl",
        "type": float, 
        "help": "KL divergenece beta",
    },
    {   
        "name": "optimizer",
        "type": str, 
        "help": "type of optimerze",
    },        
    {  
        "name": "epochs",
        "type": int, 
        "help": "total number of epochs",
    },
]

# required definitions
required = [
    "genotype",
    "fingerprint",
]

# initialize class
class DrugCell_candle(candle.Benchmark):
    def set_locals(self):
        """
        Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the benchmark.
        """
        if required is not None: 
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definisions = additional_definitions


def initialize_parameters():
    preprocessor_bmk = DrugCell_candle(file_path,
        'DrugCell_params.txt',
        'pytorch',
        prog='DrugCell',
        desc='tianshu drugcell'
    )
    #Initialize parameters
    candle_data_dir = os.getenv("CANDLE_DATA_DIR")
    gParameters = candle.finalize_parameters(preprocessor_bmk)
    return gParameters
#num_hiddens_drug = list(map(int, drug_hiddens.split(',')))
class GDSCData(Dataset):
    
    def __init__(self, response, gene_tensor, chem_tensor):
        self.response = response
        self.gene_tensor = gene_tensor
        self.chem_tensor = chem_tensor
        
    def __len__(self):
        return self.response.shape[0]
    
    def __getitem__(self, index):
        sample = self.response[index,:]        
        X_gene = self.gene_tensor[sample[0].long() ,:]
        X_chem = self.chem_tensor[sample[1].long() ,:]
        y = sample[2]
        X = torch.cat((X_gene, X_chem), 0)
        return X, y

def load_params(params, data_dir):
    print(os.environ['CANDLE_DATA_DIR'])
    args = candle.ArgumentStruct(**params)
    drug_tensor_path = data_dir + params['drug_tensor']
    params['drug_tensor'] = drug_tensor_path
    data_tensor_path = data_dir + params['data_tensor']
    params['data_tensor'] = data_tensor_path
    response_data_path = data_dir + params['response_data']
    params['response_data'] = response_data_path
    train_data_path = data_dir + params['train_data']
    params['train_data'] = train_data_path    
    test_data_path = data_dir + params['test_data']
    params['test_data'] = test_data_path
    val_data_path = data_dir + params['val_data']
    params['val_data'] = val_data_path
    onto_data_path = data_dir + params['onto_file']
    params['onto'] = onto_data_path
    cell2id_path = data_dir + params['cell2id'] 
    params['cell2id'] = cell2id_path
    drug2id_path  = data_dir + params['drug2id']
    params['drug2id'] = drug2id_path
    gene2id_path = data_dir + params['gene2id']
    params['gene2id'] = gene2id_path
    genotype_path = data_dir + params['genotype']
    params['genotype'] = genotype_path
    fingerprint_path = data_dir + params['fingerprint']
    params['fingerprint'] = fingerprint_path
    hidden_path = data_dir + params['hidden']
    params['hidden_path'] = hidden_path
    output_dir_path = data_dir + params['output_dir']
    params['output_dir'] = output_dir_path
    params['result'] = output_dir_path
    return(params)


class Drugcell_Vae(nn.Module):

    def __init__(self, term_size_map, term_direct_gene_map, dG, ngene, ndrug, root, 
                 num_hiddens_genotype, num_hiddens_drug, num_hiddens_final, 
                 n_class, inter_loss_penalty = 0.2):

        super().__init__()

        self.root = root
        self.num_hiddens_genotype = num_hiddens_genotype
        self.num_hiddens_drug = num_hiddens_drug
                
        self.num_hiddens_final = num_hiddens_final
        self.n_class = n_class
        self.inter_loss_penalty = inter_loss_penalty
        self.dG = copy.deepcopy(dG)

        # dictionary from terms to genes directly annotated with the term
        self.term_direct_gene_map = term_direct_gene_map

        self.term_visit_count = {}
        self.init_term_visits(term_size_map)
        
        # calculate the number of values in a state (term): term_size_map is the number of all genes annotated with the term
        self.term_dim_map = {}
        self.cal_term_dim(term_size_map)

        # ngenes, gene_dim are the number of all genes
        self.gene_dim = ngene
        self.drug_dim = ndrug

        # add modules for neural networks to process genotypes
        self.contruct_direct_gene_layer()
        self.construct_NN_graph(self.dG)

        # add modules for neural networks to process drugs
        self.construct_NN_drug()

        # add modules for final layer TODO: modify it into VAE
        final_input_size = num_hiddens_genotype + num_hiddens_drug[-1]
        self.add_module('final_linear_layer', nn.Linear(final_input_size, num_hiddens_final * 2))
        self.add_module('final_batchnorm_layer', nn.BatchNorm1d(num_hiddens_final * 2))
        self.add_module('final_aux_linear_layer', nn.Linear(num_hiddens_final * 2, 1))
        self.add_module('final_linear_layer_output', nn.Linear(1, 1))
        
        self.decoder_affine = nn.Linear(num_hiddens_final, 1)

    def init_term_visits(self, term_size_map):
        
        for term in term_size_map:
            self.term_visit_count[term] = 0
    
    # calculate the number of values in a state (term)
    def cal_term_dim(self, term_size_map):

        for term, term_size in term_size_map.items():
            num_output = self.num_hiddens_genotype

            # log the number of hidden variables per each term
            num_output = int(num_output)
#            print("term\t%s\tterm_size\t%d\tnum_hiddens\t%d" % (term, term_size, num_output))
            self.term_dim_map[term] = num_output


    # build a layer for forwarding gene that are directly annotated with the term
    def contruct_direct_gene_layer(self):

        for term, gene_set in self.term_direct_gene_map.items():
            if len(gene_set) == 0:
                print('There are no directed asscoiated genes for', term)
                sys.exit(1)

            # if there are some genes directly annotated with the term, add a layer taking in all genes and forwarding out only those genes
            self.add_module(term+'_direct_gene_layer', nn.Linear(self.gene_dim, len(gene_set)))


    # add modules for fully connected neural networks for drug processing
    def construct_NN_drug(self):
        input_size = self.drug_dim

        for i in range(len(self.num_hiddens_drug)):
            self.add_module('drug_linear_layer_' + str(i+1), nn.Linear(input_size, self.num_hiddens_drug[i]))
            self.add_module('drug_batchnorm_layer_' + str(i+1), nn.BatchNorm1d(self.num_hiddens_drug[i]))
            self.add_module('drug_aux_linear_layer1_' + str(i+1), nn.Linear(self.num_hiddens_drug[i],1))
            self.add_module('drug_aux_linear_layer2_' + str(i+1), nn.Linear(1,1))

            input_size = self.num_hiddens_drug[i]

    # start from bottom (leaves), and start building a neural network using the given ontology
    # adding modules --- the modules are not connected yet
    def construct_NN_graph(self, dG):

        self.term_layer_list = []   # term_layer_list stores the built neural network
        self.term_neighbor_map = {}

        # term_neighbor_map records all children of each term
        for term in dG.nodes():
            self.term_neighbor_map[term] = []
            for child in dG.neighbors(term):
                self.term_neighbor_map[term].append(child)

        while True:
            leaves = [n for n in dG.nodes() if dG.out_degree(n) == 0]
            #leaves = [n for n,d in dG.out_degree().items() if d==0]
            #leaves = [n for n,d in dG.out_degree() if d==0]

            if len(leaves) == 0:
                break

            self.term_layer_list.append(leaves)

            for term in leaves:

                # input size will be #chilren + #genes directly annotated by the term
                input_size = 0

                for child in self.term_neighbor_map[term]:
                    input_size += self.term_dim_map[child]

                if term in self.term_direct_gene_map:
                    input_size += len(self.term_direct_gene_map[term])

                # term_hidden is the number of the hidden variables in each state
                term_hidden = self.term_dim_map[term]

                self.add_module(term+'_linear_layer', nn.Linear(input_size, term_hidden))
                self.add_module(term+'_batchnorm_layer', nn.BatchNorm1d(term_hidden))
                self.add_module(term+'_aux_linear_layer1', nn.Linear(term_hidden, term_hidden))
                self.add_module(term+'_aux_linear_layer2', nn.Linear(term_hidden, 1))

            dG.remove_nodes_from(leaves)

    # definition of encoder
    def encoder(self, x):
        gene_input = x.narrow(1, 0, self.gene_dim)
        drug_input = x.narrow(1, self.gene_dim, self.drug_dim)
        
        # define forward function for genotype dcell #############################################
        term_gene_out_map = {}

        for term, _ in self.term_direct_gene_map.items():
            term_gene_out_map[term] = self._modules[term + '_direct_gene_layer'](gene_input)

        term_NN_out_map = {}
        aux_out_map = {}

        for i, layer in enumerate(self.term_layer_list):

            for term in layer:

                child_input_list = []

                self.term_visit_count[term] += 1
                
                for child in self.term_neighbor_map[term]:
                    child_input_list.append(term_NN_out_map[child])

                if term in self.term_direct_gene_map:
                    child_input_list.append(term_gene_out_map[term])

                child_input = torch.cat(child_input_list,1)

                term_NN_out = self._modules[term+'_linear_layer'](child_input)

                Tanh_out = torch.tanh(term_NN_out)
                term_NN_out_map[term] = self._modules[term+'_batchnorm_layer'](Tanh_out)
                aux_layer1_out = torch.tanh(self._modules[term+'_aux_linear_layer1'](term_NN_out_map[term]))
                aux_out_map[term] = self._modules[term+'_aux_linear_layer2'](aux_layer1_out)

        drug_out = drug_input

        for i in range(1, len(self.num_hiddens_drug)+1, 1):
            drug_out = self._modules['drug_batchnorm_layer_'+str(i)]( torch.tanh(self._modules['drug_linear_layer_' + str(i)](drug_out)))
            term_NN_out_map['drug_'+str(i)] = drug_out

            aux_layer1_out = torch.tanh(self._modules['drug_aux_linear_layer1_'+str(i)](drug_out))
            aux_out_map['drug_'+str(i)] = self._modules['drug_aux_linear_layer2_'+str(i)](aux_layer1_out)


        # connect two neural networks at the top #################################################
        final_input = torch.cat((term_NN_out_map[self.root], drug_out), 1)

        out = self._modules['final_batchnorm_layer'](torch.tanh(self._modules['final_linear_layer'](final_input)))
        term_NN_out_map['final'] = out

        aux_layer_out = torch.tanh(self._modules['final_aux_linear_layer'](out))
        aux_out_map['final'] = self._modules['final_linear_layer_output'](aux_layer_out)

        return aux_out_map, term_NN_out_map
    
    def forward(self, x):
        
        aux_out_map, term_NN_out_map = self.encoder(x)
        
        mu = term_NN_out_map['final'][..., :self.num_hiddens_final]
        log_var = term_NN_out_map['final'][..., :self.num_hiddens_final]  # T X batch X z_dim
        std_dec = log_var.mul(0.5).exp_()
        # std_dec = 1
        
        latent = MultivariateNormal(loc = mu, 
                                    scale_tril=torch.diag_embed(std_dec))
        z = latent.rsample()
        
        recon_mean = self.decoder_affine(z)
        recon_mean = F.sigmoid(recon_mean)

        return recon_mean, mu, log_var, aux_out_map, term_NN_out_map
    
    def loss_log_vae(self, recon_mean, y, mu, log_var, beta = 0.001):
        # y: true labels
        ori_y_shape = y.shape
        
        class_loss = F.mse_loss(recon_mean.view(-1), 
                                     y.reshape(-1), reduction = 'none').div(np.log(2)).view(*ori_y_shape)
        
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), 
                              dim = -1)
        
        log_loss = class_loss + beta * KLD
        log_loss = torch.mean(torch.logsumexp(log_loss, 0))
        
        return log_loss
    
    def intermediate_loss(self, aux_out_map, y):
        
        inter_loss = 0
        for name, output in aux_out_map.items():
            if name == 'final':
                inter_loss += 0
            else: # change 0.2 to smaller one for big terms
                ori_y_shape = y.shape
        
                term_loss = F.mse_loss(output.view(-1), 
                                             y.reshape(-1), 
                                             reduction = 'none').div(np.log(2)).view(*ori_y_shape)
                inter_loss += term_loss

        return inter_loss

def create_term_mask(term_direct_gene_map, gene_dim, device):
    term_mask_map = {}

    for term, gene_set in term_direct_gene_map.items():
        mask = torch.zeros(len(gene_set), gene_dim)
        for i, gene_id in enumerate(gene_set):
            mask[i, gene_id] = 1
        mask_gpu = torch.autograd.Variable(mask)
        term_mask_map[term] = mask_gpu.to(device)

    return term_mask_map

def preprocess_data(params):
    response_gdcs2 = torch.tensor(np.loadtxt(params['response_data'],delimiter=",", dtype=np.float32))
    gdsc_tensor = torch.tensor(np.loadtxt(params['data_tensor'],
                                          delimiter=",", dtype=np.float32))
    drug_tensor = torch.tensor(np.loadtxt(params['drug_tensor'],
                                          delimiter=",", dtype=np.float32))
    num_drugs = drug_tensor.shape[1]
    train_gdcs_idx = torch.unique(response_gdcs2[:,0], sorted=False)[:423]
    test_gdcs_idx = torch.unique(response_gdcs2[:,0], sorted=False)[423:]

    gdsc_data = GDSCData(response_gdcs2, gdsc_tensor, drug_tensor)
    gdsc_data_train = GDSCData(response_gdcs2[torch.isin(response_gdcs2[:,0], train_gdcs_idx)].float(), gdsc_tensor, drug_tensor)
    gdsc_data_test = GDSCData(response_gdcs2[torch.isin(response_gdcs2[:,0], test_gdcs_idx)].float(), gdsc_tensor, drug_tensor)
    return num_drugs, gdsc_data_train, gdsc_data_test


def process_drugcell_inputs(params):
    training_file = params['train_data']
    testing_file = params['test_data']
    val_file = params['val_data']
    cell2id_file = params['cell2id']
    drug2id_file = params['drug2id']
    genotype_file = params['genotype']
    fingerprint_file = params['fingerprint']
    onto_file = params['onto']
    gene2id_file = params['gene2id']
    
    train_data, feature_dict, cell2id_mapping, drug2id_mapping = prepare_train_data(training_file, 
                                                                                    testing_file, cell2id_file, 
                                                                                    drug2id_file)
   
    gene2id_mapping = load_mapping(gene2id_file)
    cell_features = np.genfromtxt(genotype_file, delimiter=',')
    drug_features = np.genfromtxt(fingerprint_file, delimiter=',')
    num_genes = len(gene2id_mapping)
#    # load ontology
    dG, root, term_size_map,term_direct_gene_map = load_ontology(onto_file,gene2id_mapping)
    return dG, root, term_size_map,term_direct_gene_map, num_genes


def run_train_vae(num_drugs, gdsc_data_train, gdsc_data_test, params):
    train_loader = DataLoader(gdsc_data_train, batch_size=params['batch_size'], shuffle=True)
    test_loader = DataLoader(gdsc_data_test, batch_size=params['batch_size'], shuffle=False)
    dG, root, term_size_map,term_direct_gene_map, num_genes  = process_drugcell_inputs(params)
    num_hiddens_drug = list(map(int, params['drug_hiddens'].split(',')))
    model = Drugcell_Vae(term_size_map, term_direct_gene_map, dG, num_genes, num_drugs, 
                         root, params['num_hiddens_genotype'], num_hiddens_drug, params['num_hiddens_final'], 
                         inter_loss_penalty=params['inter_loss_penalty'],
                         n_class = 0)

    DEVICE='cuda:' + str(params['cuda'])
    model.to(DEVICE)
    term_mask_map = create_term_mask(model.term_direct_gene_map, num_genes, device = DEVICE)
    
    best_loss = 1000
    training_loss_list = []
    testing_loss_list = []
    epoch_list = []
    accu_list = []

    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], betas=params['betas_adam'], eps=params['eps_adam'])
    term_mask_map = create_term_mask(model.term_direct_gene_map, gene_dim=num_genes, device=DEVICE)

    optimizer.zero_grad()

    for name, param in model.named_parameters():
        term_name = name.split('_')[0]

        if '_direct_gene_layer.weight' in name:
            param.data = torch.mul(param.data, term_mask_map[term_name].to(DEVICE)) * params['direct_gene_weight_param']
        else:
            param.data = param.data * params['direct_gene_weight_param']

    mse_tmp_testing = torch.tensor(0, device=DEVICE)
    for epoch in range(params['epochs']):
        model.train()
        train_predict = torch.zeros(0, 0).to(DEVICE)

        tloader = tqdm.tqdm(enumerate(train_loader))
        for i, (data, response) in tloader:
            # Convert torch tensor to Variable
            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer

            # Here term_NN_out_map is a dictionary
            recon_mean, mu, log_var, aux_out_map, term_NN_out_map = model(data.to(DEVICE))

            if train_predict.size()[0] == 0:
                train_predict = aux_out_map["final"].data
            else:
                train_predict = torch.cat([train_predict, aux_out_map["final"].data], dim=0)

            total_loss = 0
            loss_vae = model.loss_log_vae(
                recon_mean=recon_mean, y=response.to(DEVICE), mu=mu, log_var=log_var, beta=params['beta_kl']
            )

            loss_intermidiate = model.intermediate_loss(aux_out_map, response.to(DEVICE))
            total_loss = torch.mean(loss_vae + model.inter_loss_penalty * loss_intermidiate)
            tmp_loss = total_loss.item()
            total_loss.backward()
            for name, param in model.named_parameters():
                if "_direct_gene_layer.weight" not in name:
                    continue
                term_name = name.split("_")[0]
                # print name, param.grad.data.size(), term_mask_map[term_name].size()
                param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])

            optimizer.step()

            if i % 6 == 0:
                with torch.no_grad():
                    (inputdata, response) = next(iter(test_loader))
                    recon_mean, mu, log_var, aux_out_map, term_NN_out_map = model(inputdata.to(DEVICE))
                    mse_tmp_testing = F.mse_loss(recon_mean.detach().squeeze().cpu(), response.squeeze())

                    tloader.set_postfix({"Epoch": epoch, 
                                         "Training Loss": tmp_loss, 
                                         "Testing Loss": mse_tmp_testing.item()})
                
                    training_loss_list.append(tmp_loss)
                    testing_loss_list.append(mse_tmp_testing.item())
                    epoch_list.append(epoch)
        with torch.no_grad():
            (inputdata, response) = next(iter(test_loader))
            recon_mean, mu, log_var, aux_out_map, term_NN_out_map = model(inputdata.to(DEVICE))
            mse_tmp_testing = F.mse_loss(recon_mean.detach().squeeze().cpu(), response.squeeze())
        
            if mse_tmp_testing < best_loss:
                pass
            # torch.save(model, "gdsc_drug_epoch_new.pt")
    # if epoch % 10 == 0:
    # torch.save(model, "gdsc_50.pt")

# %%

def run(params):
    num_drugs, gdsc_data_train, gdsc_data_test = preprocess_data(params)
    run_train_vae(num_drugs, gdsc_data_train, gdsc_data_test, params)
    
def candle_main():
    params = initialize_parameters()
    params = load_params(params, data_dir)
    run(params)

if __name__ == "__main__":
    candle_main()
