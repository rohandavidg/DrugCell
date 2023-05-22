#!/homes/ac.rgnanaolivu/miniconda3/envs/rohan_python/bin/python

import sys
import os
import numpy as np
import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from code.drugcell_NN import *
import argparse
import numpy as np
import pandas as pd
import candle
import time
import logging
import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag
from pathlib import Path
import improve_utils

file_path = os.path.dirname(os.path.realpath(__file__))
fdir = Path('__file__').resolve().parent
source = "csa_data/raw_data/splits/"
required = None
additional_definitions = None

# This should be set outside as a user environment variable
os.environ['CANDLE_DATA_DIR'] = os.environ['HOME'] + '/improve_data_dir/'


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
        prog='DrugCell_candle',
        desc='Data Preprocessor'
    )
    #Initialize parameters
    candle_data_dir = os.getenv("CANDLE_DATA_DIR")
    gParameters = candle.finalize_parameters(preprocessor_bmk)
    return gParameters


def mkdir(directory):
    directories = directory.split("/")   

    folder = ""
    for d in directories:
        folder += d + '/'
        if not os.path.exists(folder):
            print('creating folder: %s'%folder)
            os.mkdir(folder)


def preprocess(params):
    print(os.environ['CANDLE_DATA_DIR'])
    #requirements go here
    keys_parsing = ["output_dir", "hidden", "result", "metric", "data_type"]
    data_dir = os.environ['CANDLE_DATA_DIR'] + "/DrugCell/Improve/Data/"
    mkdir(data_dir)
#    data_download_filepath = candle.get_file(params['original_data'], params['data_url'],
#                                        datadir = params['data_dir'],
#                                        cache_subdir = None)
#    print('download_path: {}'.format(data_download_filepath))
#    predict_download_filepath = candle.get_file(params['data_predict'], params['predict_url'],
#                                        datadir = params['data_dir'],
#                                        cache_subdir = None)
#    print('download_path: {}'.format(predict_download_filepath))
#    model_download_filepath = candle.get_file(params['data_model'], params['model_url'],
#                                        datadir = params['data_dir'],
#                                        cache_subdir = None)
#    print('download_path: {}'.format(model_download_filepath))

    model_param_key = []
    for key in params.keys():
        if key not in keys_parsing:
                model_param_key.append(key)
    model_params = {key: params[key] for key in model_param_key}
    params['model_params'] = model_params
    args = candle.ArgumentStruct(**params)
    train_data_path = data_dir + params['train_data']
    params['train_data'] = train_data_path
    test_data_path = data_dir + params['test_data']
    params['test_data'] = test_data_path
    val_data_path = data_dir + params['val_data']
    params['val_data'] = val_data_path
    onto_data_path = data_dir + params['onto']
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
    result_path = data_dir + params['result']
    params['result'] = result_path
    return(params)


def map_smiles(df, metric):
    smiles_df = improve_utils.load_smiles_data()
    data_smiles_df = df.merge(smiles_df, on = "improve_chem_id", how='left') 
    data_smiles_df = data_smiles_df[~data_smiles_df[metric].isna()]
    data_smiles_df = data_smiles_df[['improve_sample_id', 'smiles', metric]]
    data_smiles_df = data_smiles_df.drop_duplicates()
    data_smiles_df = data_smiles_df.reset_index(drop=True)
    return data_smiles_df


def generate_drugdata(params):
    data_type = params['data_type']
    metric =  params['metric']
    train_out =  params['train_data']
    test_out = params['test_data']
    val_out = params['val_data']
    rs_all = improve_utils.load_single_drug_response_data(source=data_type, split=0,
                                                          split_type=["train", "test", 'val'],
                                                          y_col_name=metric)
    rs_train = improve_utils.load_single_drug_response_data(source=data_type,
                                                            split=0, split_type=["train"],
                                                            y_col_name=metric)
    rs_test = improve_utils.load_single_drug_response_data(source=data_type,
                                                           split=0,
                                                           split_type=["test"],
                                                           y_col_name=metric)
    rs_val = improve_utils.load_single_drug_response_data(source=data_type,
                                                          split=0,
                                                          split_type=["val"],
                                                          y_col_name=metric)
    train_df = map_smiles(rs_train, metric)
    train_df.to_csv(train_out, header=None, index=None, sep ='\t')
    print("wrote out train data at {0}".format(train_out))
    test_df = map_smiles(rs_test, metric)
    test_df.to_csv(test_out, header=None, index=None, sep='\t')
    print("wrote out test data at {0}".format(test_out))
    val_df = map_smiles(rs_val, metric)
    val_df.to_csv(val_out, header=None, index=None, sep ='\t')
    print("wrote out val data at {0}".format(val_out))    
    return rs_all
    
def generate_index_files(params, data_df):
    drug_index_out = params['drug2id']
    cell_index_out = params['cell2id']
    gene_index_out = params['gene2id']
    cell_mutation_out = params['genotype']
    drug_fingerprint_out = params['fingerprint']

    #gene index file
    mutation_data = improve_utils.load_cell_mutation_data(gene_system_identifier="Entrez")
    mutation_data = mutation_data.reset_index()
    gene_data = mutation_data.columns
    gene_data = gene_data[1:]
    gene_list = list(set(list(gene_data)))
    gene_df = pd.DataFrame(gene_data)
    gene_df.to_csv(gene_index_out, sep='\t', header=None)

    #improve id
    improve_data_list = list(set(data_df.improve_sample_id.tolist()))
    
    #cell2mutation file
    data_df = data_df[data_df['improve_sample_id'].isin(improve_data_list)]
    cell2mut_df = mutation_data.drop(columns=['improve_sample_id'])
    cell2mut_df.to_csv(cell_mutation_out, header=None, index=None)
    
    #cell2ind txt file
    cellind_df = pd.DataFrame(data_df.improve_sample_id).drop_duplicates()
    cellind_df = cellind_df.reset_index(drop=True)
    cellind_df.to_csv(cell_index_out, sep='\t', header=None)

    #drug2fingerprint file
    drug_list = list(set(data_df.improve_chem_id.tolist()))
    fp = improve_utils.load_morgan_fingerprint_data()
    fp_df = fp.reset_index()
    fp_df = fp_df[fp_df['improve_chem_id'].isin(drug_list)]
    fp_df = fp_df.drop(columns=['improve_chem_id'])
    fp_df.to_csv(drug_fingerprint_out, index=None, header=None)

    #drug2ind
    se = improve_utils.load_smiles_data()
    data_df = data_df.merge(se, on = 'improve_chem_id', how='left')
    drug_only = data_df.smiles.drop_duplicates()
    drug_only = drug_only.reset_index(drop=True)
    drug_only.to_csv(drug_index_out, sep='\t', header=None)
    return gene_list
    
def create_ont(ont_in, ont_out, gene_list):
    ont_df = pd.read_csv(ont_in, sep='\t', header=None)
    ont_default_df = ont_df[ont_df[2] == 'default']
    ont_gene_df = ont_df[ont_df[2] == 'gene']
    ont_gene_df = ont_gene_df[ont_df[1].isin(gene_list)]
    GO_list = list(set(ont_gene_df[0].tolist()))
    ont_default_df = ont_default_df[(ont_default_df[0].isin(GO_list)) | (ont_default_df[1].isin(GO_list))]
    ont_cat_df = pd.concat([ont_default_df, ont_gene_df])
    ont_cat_df.to_csv(ont_out, sep='\t', index=None, header=None)

def candle_main():
    params = initialize_parameters()
    params =  preprocess(params)
    data_df = generate_drugdata(params)
    gene_list = generate_index_files(params, data_df)
    create_ont(params['onto_in'],params['onto'], gene_list) 
if __name__ == "__main__":
    candle_main()