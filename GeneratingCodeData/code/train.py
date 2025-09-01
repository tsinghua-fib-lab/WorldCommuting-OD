import time
import random
import warnings
warnings.filterwarnings("ignore")

from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from tqdm import tqdm
from pprint import pprint

from utils.tool import *
from utils.metrics import *
from data_load import *



def train(config, diff_model, train_Dloader, valid_Dloader, logger, scalers):
    optm = torch.optim.AdamW(diff_model.model.parameters(), lr=config["learning_rate"], weight_decay=0.001)
    scheduler = StepLR(optm, step_size=5000, gamma=0.9)
    if config["mode"] == "CONTINUE":
        path = logger.optimizer_path[:-4] + "_lastest.pkl"
        state_dict = torch.load(path, map_location=config["device"])
        optm.load_state_dict(state_dict)
        start_epochs = logger.exp_log["train_log"]["num_epochs"]
    else:
        start_epochs = 0

    print("--------------------------------------------------")
    Ts = [x for x in range(config["T"])]
    for epoch in range(start_epochs, config["EPOCH"]):
        ###################### train
        diff_model.train()
        print("  ** Training. Epoch:", epoch)
        train_loss = 0
        for i, batch in tqdm(enumerate(train_Dloader), total=len(train_Dloader), unit="batch"):
            _, batched_graph, batched_dis, ods = batch

            optm.zero_grad()

            # t
            t = torch.LongTensor([random.choice(Ts) for _ in ods]).to(config["device"])
            # n
            n = batched_graph.ndata["nfeat"].to(config["device"])
            # e
            dim = sum([x.shape[0] for x in ods])
            batched_od = torch.full((dim, dim), 0, dtype=torch.float32)
            batchlization = torch.full((dim, dim), 0, dtype=torch.float32).to(config["device"]) # There are zero-noise from blocking the OD matrix.
            l, r = 0, 0
            for i, od in enumerate(ods):
                l = r
                r = r + od.shape[0]
                batched_od[l:r, l:r] = ods[i]
                batchlization[l:r, l:r] = 1
            e = batched_od.to(config["device"])

            # net
            net = (n, e)
            # loss
            loss = diff_model.loss(net, t, batched_dis.to(config["device"]), batchlization)
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diff_model.parameters(), 1.)
            optm.step()
        scheduler.step()
        train_loss = train_loss / len(train_Dloader)
        logger.once_train_record(epoch, train_loss, diff_model, optm)
        print("--------------------------------------------------")


        ###################### valid
        diff_model.eval()
        if (epoch % config["valid_period"] == 0):
            print("--------------------------------------------------")
            with torch.no_grad():
                print("  ** Valid. Epoch:", epoch)
                valid_evaluator = validEvaluator(config)
                valid_loss = 0
                for i, batch in tqdm(enumerate(valid_Dloader), total=len(valid_Dloader), unit="batch"):
                    _, batched_graph, batched_dis, ods = batch

                    # t
                    t = torch.LongTensor([random.choice(Ts)]).to(config["device"])
                    # n
                    n = batched_graph.ndata["nfeat"].to(config["device"])
                    # e
                    dim = sum([x.shape[0] for x in ods])
                    batched_od = torch.full((dim, dim), 0, dtype=torch.float32)
                    batchlization = torch.full((dim, dim), 0, dtype=torch.float32).to(config["device"]) # There are zero-noise from blocking the OD matrix.
                    l, r = 0, 0
                    for i, od in enumerate(ods):
                        l = r
                        r = r + od.shape[0]
                        batched_od[l:r, l:r] = ods[i]
                        batchlization[l:r, l:r] = 1
                    e = batched_od.to(config["device"])

                    # net
                    net = (n, e)
                    # dis
                    batched_dis = batched_dis.to(config["device"])
                    # loss
                    loss = diff_model.loss(net, t, batched_dis, batchlization)
                    valid_loss += loss.item()
                    # metrics
                    c = (net, batched_dis, batchlization)
                    # generate
                    e_hats = []
                    for _ in range(config["sample_times"]):
                        e_hat = diff_model.DDIM_sample_loop(n.shape, e.shape, c)[-1]
                        e_hats.append(e_hat)
                    e_hat = torch.mean(torch.stack(e_hats), dim=0)

                    # eval
                    pred = e_hat.cpu().numpy()
                    valid_evaluator.evaluate_batch(pred=pred,
                                                   gt=ods,
                                                   scalers=scalers)

                valid_loss = valid_loss / len(valid_Dloader)
                metrics = valid_evaluator.summary_all_metrics()

                logger.once_valid_record(epoch, valid_loss, metrics, diff_model, optm)
            logger.save_exp_log()
            print("--------------------------------------------------")
            if logger.overfit_flag >= config["overfit_tolerance"]:
                print(" ** Early stop!")
                print("--------------------------------------------------")
                break
