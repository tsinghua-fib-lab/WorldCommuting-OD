import os
import json
from copy import deepcopy

import numpy as np

import geopandas as gpd

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.metrics import accuracy
from utils.tool import *


class Logger():
    def __init__(self, config):
        self.config = config

        # directories
        summary_path = "exp/run/" + config["exp_name"]
        os.makedirs(summary_path, exist_ok=True)

        self.exp_directory = config["exp_path"]
        self.exp_name = config["exp_name"]
        self.exp_path = self.exp_directory + "log/" + self.exp_name + "_" + str(self.config["random_seed"]) + ".json"
        os.makedirs(self.exp_directory + "log/", exist_ok=True)
       
        self.model_directory = config["exp_path"] + "model/" + self.exp_name
        self.model_path = self.model_directory + "/model_" + str(self.config["random_seed"]) + '.pkl'
        os.makedirs(self.model_directory, exist_ok=True)

        self.optimizer_path = self.exp_directory + "optm/" + self.exp_name + "/optm_" + str(self.config["random_seed"]) + '.pkl'
        os.makedirs(self.exp_directory + "optm/" + self.exp_name, exist_ok=True)

        self.generation_directory = self.exp_directory + "generation/" + self.exp_name + "/" 
        os.makedirs(self.generation_directory, exist_ok=True)

        self.denoising_directory = self.exp_directory + "denoising/" + self.exp_name + "/"
        os.makedirs(self.denoising_directory, exist_ok=True)

        # records
        self.training_losses = []
        self.valid_losses = []
        self.valid_metrics = []
        self.best_valid_gen_cpc = 0
        self.best_valid_comp_cpc = 0
        self.best_valid_missgen_cpc = 0
        self.best_valid_misscomp_cpc = 0
        self.pre_train_losses = []
        self.pre_train_metrics = {}

        # early stopping
        self.overfit_flag = 0
        
        # log dicts
        self.train_log = {
            "num_epochs" : 0,

            "train_loss" : self.training_losses,
            "valid_loss" : self.valid_losses,

            "best_valid_gen_cpc" : self.best_valid_gen_cpc,
            "best_valid_comp_cpc" : self.best_valid_comp_cpc,
            "best_valid_missgen_cpc" : self.best_valid_missgen_cpc,
            "best_valid_misscomp_cpc" : self.best_valid_misscomp_cpc,

            "valid_metrics" : self.valid_metrics
        }

        self.exp_log = {
                         "config" : self.config,
                         "train_log" : self.train_log
        }

        # tensorboard writer
        self.summary_writer = SummaryWriter(log_dir=summary_path, flush_secs=3)


    # utils function
    def save_model_optm(self, model, optimizer, tag):
        torch.save(model.state_dict(), self.model_path[:-4] + "_" + tag + ".pkl")
        torch.save(optimizer.state_dict(), self.optimizer_path[:-4] + "_" + tag + ".pkl")

    def summary_record(self, variable, name, iteration):
        self.summary_writer.add_scalar(name, variable, iteration)


    # pretrain log
    def pretrain_loss_record(self, t_loss, v_loss, epoch):
        self.summary_writer.add_scalars("pretrain_loss", {"train": t_loss,
                                                          "valid": v_loss}, epoch)
    
    def pretrain_metrics_record(self, metrics, epoch):
        self.summary_writer.add_scalars("pretrain_metrics", {"avg_cpc": metrics["all"]["CPC"],
                                                             "large_cpc": metrics["(500, 1000]"]["CPC"]}, epoch)

    # train log
    def once_train_record(self, epoch, loss, model, optimizer):
        self.training_losses.append(float(loss))
        self.train_log["num_epochs"] += 1
        self.save_model_optm(model, optimizer, tag="lastest")
        self.summary_writer.add_scalars("loss", {"train": loss}, epoch)


    # valid log
    def once_valid_record(self, epoch, loss, metrics, model, optimizer):
        # log
        self.valid_losses.append({epoch : float(loss)})
        self.valid_metrics.append({epoch : metrics})

        self.overfit_flag = self.overfit_flag + 1
        # saving model optm
        # self.save_model_optm(model, optimizer, tag="EPOCH"+str(epoch))
        if metrics["CPC"] > self.best_valid_gen_cpc:
            self.best_valid_gen_cpc = metrics["CPC"]
            self.save_model_optm(model, optimizer, tag="best")
            self.overfit_flag = 0

        # tensorboard
        self.summary_writer.add_scalars("loss", {"valid": loss}, epoch)
        self.metrics_to_tensorboard(metrics, epoch)

    def metrics_to_tensorboard(self, metrics, epoch):
        for metric, value in metrics.items():
            self.summary_writer.add_scalar("metrics/"+metric, value, epoch)
        
    
    # reload when continue/eval
    def load_exp_log(self):
        print("*********** load exp log **************", "\n")
        self.exp_log = json.load(open(self.exp_path, "r"))
        self.exp_log["config"]["device"] = torch.device(int(self.exp_log["config"]["device"]))

        self.training_losses = self.exp_log["train_log"]["train_loss"]
        self.valid_losses = self.exp_log["train_log"]["valid_loss"]
        self.best_valid_gen_cpc = self.exp_log["train_log"]["best_valid_gen_cpc"]
        self.valid_metrics = self.exp_log["train_log"]["valid_metrics"]

        self.train_log = self.exp_log["train_log"]
        self.exp_log["train_log"] = self.train_log

        # recover tensorboard
        print("*********** load tensorboard history **************", "\n")
        for i, loss in enumerate(self.training_losses):
            self.summary_writer.add_scalars("loss", {"train": loss}, i)
        for _, loss in enumerate(self.valid_losses):
            (epoch, loss), = loss.items()
            self.summary_writer.add_scalars("loss", {"valid": loss}, epoch)
        for _, metrics in enumerate(self.valid_metrics):
            (epoch, metrics), = metrics.items()
            self.metrics_to_tensorboard(metrics, epoch)


    # save log to file
    def save_exp_log(self):
        exp_log = deepcopy(self.exp_log)
        exp_log["config"]["device"] = int(exp_log["config"]["device"].index)
        json.dump(exp_log, open(self.exp_path, "w"), indent=4)


    # plot test generation od arc
    def plot_generated_od(self, geoid, generated_od):
        low = np.percentile(generated_od, 80)
        high = np.percentile(generated_od, 95)
        geometries = gpd.read_file(f"data/cities_global_1625_shp/{geoid}/regions.shp").to_crs(epsg=4326)

        f = plot_od_arc_chart(generated_od, geometries, low, high)
        
        f.savefig(self.generation_directory + geoid + "/generation.png", bbox_inches='tight')