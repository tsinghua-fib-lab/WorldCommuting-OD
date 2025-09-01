import os
import json
import random
import warnings
warnings.filterwarnings("ignore")

import torch

from tqdm import tqdm
from pprint import pprint

from utils.tool import mean_the_denoising_process, plot_generated_od
from utils.metrics import *


def GEN_specific(config, diff_model, GEN_Dloader, logger, scalers):
    print("--------------------------------------------------")
    with torch.no_grad():
        print("  ** Generation. :")
        for i, batch in tqdm(enumerate(GEN_Dloader), total=len(GEN_Dloader)):
            geoid, batched_graph, batched_dis, ods = batch
            if os.path.exists(logger.generation_directory + geoid[0] + "/generation.npy"):
                if os.path.exists(logger.generation_directory + geoid[0] + "/generation.png"):
                    continue
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

            diff_model.eval()
            c = (net, batched_dis, batchlization)
            # generate
            e_hats = []
            for _ in range(config["sample_times"]):
                e_hat = diff_model.DDIM_sample_loop(n.shape, e.shape, c)[-1]
                e_hats.append(e_hat.cpu().numpy())
            e_hat = np.mean(np.stack(e_hats), axis=0)

            od_hat = e_hat
            od_hat = scalers["od"].inverse_transform(od_hat.reshape([-1, 1])).reshape([od_hat.shape[0], od_hat.shape[1]])
            od_hat = scalers["od_normer"].inverse_transform(od_hat)
            od_hat[od_hat < 0] = 0
            for i in range(od_hat.shape[0]): # consistant with groundtruth
                od_hat[i,i] = 0
            od_hat = np.floor(od_hat)
            os.makedirs(logger.generation_directory + geoid[0], exist_ok=True)
            np.save(logger.generation_directory + geoid[0] + "/generation.npy", od_hat)

            logger.plot_generated_od(geoid[0], od_hat)
    print("--------------------------------------------------")


def GEN_specific_UK(config, diff_model, UK_Dloader, logger, scalers):
    print("--------------------------------------------------")
    with torch.no_grad():
        print("  ** Generation. :")
        for i, batch in tqdm(enumerate(UK_Dloader), total=len(UK_Dloader)):
            geoid, batched_graph, batched_dis, ods = batch
            if os.path.exists("exp/generation/US2UK/" + geoid[0] + "/generation.npy"):
                if os.path.exists("exp/generation/US2UK/" + geoid[0] + "/generation.png"):
                    continue
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

            diff_model.eval()
            c = (net, batched_dis, batchlization)
            # generate
            e_hats = []
            for _ in range(config["sample_times"]):
                e_hat = diff_model.DDIM_sample_loop(n.shape, e.shape, c)[-1]
                e_hats.append(e_hat.cpu().numpy())
            e_hat = np.mean(np.stack(e_hats), axis=0)

            od_hat = e_hat
            od_hat = scalers["od"].inverse_transform(od_hat.reshape([-1, 1])).reshape([od_hat.shape[0], od_hat.shape[1]])
            od_hat = scalers["od_normer"].inverse_transform(od_hat)
            od_hat[od_hat < 0] = 0
            for i in range(od_hat.shape[0]): # consistant with groundtruth
                od_hat[i,i] = 0
            od_hat = np.floor(od_hat)

            os.makedirs("exp/generation/US2UK/" + geoid[0], exist_ok=True)
            np.save("exp/generation/US2UK/" + geoid[0] + "/generation.npy", od_hat)
            plot_generated_od("data/UK_shp/" + geoid[0] + "/area_shp/" + geoid[0] + ".shp", od_hat, "exp/generation/US2UK/" + geoid[0] + "/generation.png")
    print("--------------------------------------------------")


def GEN_specific_GHSL(config, diff_model, GHSL_Dloader, logger, scalers):
    print("--------------------------------------------------")
    with torch.no_grad():
        print("  ** Generation. :")
        for i, batch in tqdm(enumerate(GHSL_Dloader), total=len(GHSL_Dloader)):
            geoid, batched_graph, batched_dis, ods = batch
            if os.path.exists("exp/generation/GHSL_744/" + geoid[0] + "/generation.npy"):
                if os.path.exists("exp/generation/GHSL_744/" + geoid[0] + "/generation.png"):
                    continue

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
            
            diff_model.eval()
            c = (net, batched_dis, batchlization)
            # generate
            e_hats = []
            for _ in range(config["sample_times"]):
                e_hat = diff_model.DDIM_sample_loop(n.shape, e.shape, c)[-1]
                e_hats.append(e_hat.cpu().numpy())
            e_hat = np.mean(np.stack(e_hats), axis=0)

            od_hat = e_hat
            od_hat = scalers["od"].inverse_transform(od_hat.reshape([-1, 1])).reshape([od_hat.shape[0], od_hat.shape[1]])
            od_hat = scalers["od_normer"].inverse_transform(od_hat)
            od_hat[od_hat < 0] = 0
            for i in range(od_hat.shape[0]): # consistant with groundtruth
                od_hat[i,i] = 0
            od_hat = np.floor(od_hat)

            os.makedirs("exp/generation/GHSL_744/" + geoid[0], exist_ok=True)
            np.save("exp/generation/GHSL_744/" + geoid[0] + "/generation.npy", od_hat)
            plot_generated_od("data/GHSL_744_shp/" + geoid[0] + "/" + "regions.shp", od_hat, "exp/generation/GHSL_744/" + geoid[0] + "/generation.png")
    print("--------------------------------------------------")