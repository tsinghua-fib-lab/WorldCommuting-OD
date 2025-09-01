import os
from random import shuffle, choice

import pickle as pkl
import numpy as np

import torch
import dgl

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from utils.tool import *



def prepare_data(config):

    if config["skew_norm"] == "none":
        OD_normer = None_transformer()
    elif config["skew_norm"] == "log":
        OD_normer = Log_transformer()
    elif config["skew_norm"] == "boxcox":
        OD_normer = BoxCox_transformer()
    else:
        raise Exception("Unknown skew_norm type")


    cities = os.listdir(config["data_path"])
    if config["shuffle_cities"] == 1:
        shuffle(cities)

    # load data
    print("***************** load data *****************")
    # worldpop part
    worldpop = np.load("data/worldpop_us.npy")
    worldpop = worldpop[:, np.argsort(worldpop[0])]

    data = []
    for city in tqdm(cities):
        one = {
                "GEOID" : city,
                "demo": np.load(config["data_path"] + city + "/nfeat/demos.npy"),
                "pois": np.load(config["data_path"] + city + "/nfeat/pois.npy"),
                "imgfeat" : np.load(config["data_path"] + city + "/nfeat/imgfeat.npy"),
                "dis": np.load(config["data_path"] + city + "/adj/dis.npy"),
                "od": OD_normer.fit_transform(np.load(config["data_path"] + city + "/od/od.npy"))
            }

        for i in range(one["od"].shape[0]):
            one["od"][i,i] = 0

        worldpop_feat = worldpop[:, (worldpop[0] // 1e6) == int(city)][1:, :].transpose()
        if worldpop_feat.shape[0] != one["imgfeat"].shape[0]:
            continue
        one["pop"] = np.log1p(worldpop_feat) # log scale
        one["nfeat"] = np.concatenate([one["pop"], one["demo"], one["pois"], one["imgfeat"]], axis=1)

        data.append(one)

    print("  ** constructing dataset...", end="")
    train, valid = split_data_intoTV(data, config)
    print("done, totally", len(data), "cities")

    # normalization
    print("  ** normalizing dataset...", end="")
    if config["attr_MinMax"] == 1:
        scaler_feat = MinMaxer(np.concatenate([x["nfeat"] for x in train], axis=0))
        for i, nfeat in enumerate([x["nfeat"] for x in train]):
            train[i]["nfeat"] = scaler_feat.transform(nfeat)
        for i, nfeat in enumerate([x["nfeat"] for x in valid]):
            valid[i]["nfeat"] = scaler_feat.transform(nfeat)
        scaler_dis = MinMaxer(np.concatenate([x["dis"].reshape([-1, 1]) for x in train], axis=0))
        for i, dis in enumerate([x["dis"] for x in train]):
            train[i]["dis"] = scaler_dis.transform(dis.reshape([-1, 1])).reshape([dis.shape[0], dis.shape[1]])
        for i, dis in enumerate([x["dis"] for x in valid]):
            valid[i]["dis"] = scaler_dis.transform(dis.reshape([-1, 1])).reshape([dis.shape[0], dis.shape[1]])
    else:
        scaler_feat = None
        scaler_dis = None

    if config["od_MinMax"] == 1:
        scaler_od = MinMaxer(np.concatenate([x["od"].reshape([-1, 1]) for x in train], axis=0))
        for i, od in enumerate([x["od"] for x in train]):
            train[i]["od"] = scaler_od.transform(od.reshape([-1, 1])).reshape([od.shape[0], od.shape[1]])
        for i, od in enumerate([x["od"] for x in valid]):
            valid[i]["od"] = scaler_od.transform(od.reshape([-1, 1])).reshape([od.shape[0], od.shape[1]])
    else:
        scaler_od = None
    scalers = {
        "nfeat" : scaler_feat,
        "dis" : scaler_dis,
        "od" : scaler_od,
        "od_normer" : OD_normer,
    }
    print("done")
    
    return data, train, valid, scalers


def prepare_UK_dataloader(config, scalers):
    test_cities_dir = "data/UK/"
    cities = os.listdir(test_cities_dir)

    # load data
    print("\n***************** load UK data *****************")
    data = []
    for city in tqdm(cities):
        one = {
                "GEOID" : city,
                "pop": np.load(test_cities_dir + city + "/nfeat/worldpop.npy"),
                "demo": np.load(test_cities_dir + city + "/nfeat/demos.npy"), 
                "pois": np.load(test_cities_dir + city + "/nfeat/pois.npy"),
                "imgfeat" : np.load(test_cities_dir + city + "/nfeat/imgfeat.npy"),
                "dis": np.load(test_cities_dir + city + "/adj/dis.npy"),
                "od": np.load(test_cities_dir + city + "/od/od.npy")
            }

        one["nfeat"] = np.concatenate([np.log1p(one["pop"]), one["demo"], one["pois"], one["imgfeat"]], axis=1)
        data.append(one)

    # normalization
    print("  ** normalizing UK dataset...", end="")
    if config["attr_MinMax"] == 1:
        for i, nfeat in enumerate([x["nfeat"] for x in data]):
            data[i]["nfeat"] = scalers["nfeat"].transform(nfeat)
        for i, dis in enumerate([x["dis"] for x in data]):
            data[i]["dis"] = scalers["dis"].transform(dis.reshape([-1, 1])).reshape([dis.shape[0], dis.shape[1]])
    
    test_set = SpecificDataset(data, config, "GEN")
    ttBS = MyBatchSampler(test_set, 1, 10000)
    from dgl.dataloading import GraphDataLoader
    UK_Dloader = GraphDataLoader(test_set, batch_sampler=ttBS, collate_fn=collate_fn)

    return UK_Dloader



def prepare_GHSL_dataloader(config, scalers):
    test_cities_dir = "data/GHSL_744/"
    cities = os.listdir(test_cities_dir)
    
    # load data
    print("\n***************** load GHSL_744 data *****************")
    data = []
    for city in tqdm(cities):
        one = {
                "GEOID" : city,
                "pop": np.load(test_cities_dir + city + "/nfeat/worldpop.npy"),
                "demo": np.load(test_cities_dir + city + "/nfeat/demos.npy"), 
                "pois": np.load(test_cities_dir + city + "/nfeat/pois.npy"),
                "imgfeat" : np.load(test_cities_dir + city + "/nfeat/imgfeat.npy"),
                "dis": np.load(test_cities_dir + city + "/adj/dis.npy"),
                "od": np.ones_like(np.load(test_cities_dir + city + "/adj/dis.npy"))
            }

        one["nfeat"] = np.concatenate([np.log1p(one["pop"]), one["demo"], one["pois"], one["imgfeat"]], axis=1)
        data.append(one)

    # normalization
    print("  ** normalizing GHSL_744 dataset...", end="")
    if config["attr_MinMax"] == 1:
        for i, nfeat in enumerate([x["nfeat"] for x in data]):
            data[i]["nfeat"] = scalers["nfeat"].transform(nfeat)
        for i, dis in enumerate([x["dis"] for x in data]):
            data[i]["dis"] = scalers["dis"].transform(dis.reshape([-1, 1])).reshape([dis.shape[0], dis.shape[1]])
    
    test_set = SpecificDataset(data, config, "GEN")
    ttBS = MyBatchSampler(test_set, 1, 10000)
    from dgl.dataloading import GraphDataLoader
    GHSL_Dloader = GraphDataLoader(test_set, batch_sampler=ttBS, collate_fn=collate_fn)

    return GHSL_Dloader

def prepare_specific_dataloader(config, scalers):

    test_cities_dir = "data/global_cities/"

    cities = os.listdir(test_cities_dir)
    error_list = ["1022_GL_Nuuk", "hzy-405_LBN_Nabatiyet_et_Tahta", "hzy-513_POL_Zielona_Gora",
                  "hzy-938_POL_Wroclaw", "215_PL_Leszno", "hzy-787_POL_Jelenia_Gora",
                  "1059_IL_Haifa", "292_BR_Santo_Antonio", "35_US_San_Luis", "655_PL_Opole",
                  "740_RU_Pskov", "668_PL_Legnica", "897_DE_Magdeburg"]
    # load data
    print("\n***************** load specific data *****************")
    data = []
    for city in tqdm(cities):
        if city in error_list:
            continue
        one = {
                "GEOID" : city,
                "pop": np.load(test_cities_dir + city + "/nfeat/worldpop.npy"),
                "demo": np.load(test_cities_dir + city + "/nfeat/demos.npy"), 
                "pois": np.load(test_cities_dir + city + "/nfeat/pois.npy"),
                "imgfeat" : np.load(test_cities_dir + city + "/nfeat/imgfeat.npy"),
                "dis": np.load(test_cities_dir + city + "/adj/dis.npy"),
                "od": np.ones_like(np.load(test_cities_dir + city + "/adj/dis.npy"))
            }

        one["nfeat"] = np.concatenate([np.log1p(one["pop"]), one["demo"], one["pois"], one["imgfeat"]], axis=1)
        data.append(one)

    # normalization
    print("  ** normalizing specific dataset...", end="")
    if config["attr_MinMax"] == 1:
        for i, nfeat in enumerate([x["nfeat"] for x in data]):
            data[i]["nfeat"] = scalers["nfeat"].transform(nfeat)
        for i, dis in enumerate([x["dis"] for x in data]):
            data[i]["dis"] = scalers["dis"].transform(dis.reshape([-1, 1])).reshape([dis.shape[0], dis.shape[1]])
    
    test_set = SpecificDataset(data, config, "GEN")
    ttBS = MyBatchSampler(test_set, 1, 10000)
    from dgl.dataloading import GraphDataLoader
    specific_Dloader = GraphDataLoader(test_set, batch_sampler=ttBS, collate_fn=collate_fn)

    return specific_Dloader


class MyDataset(dgl.data.DGLDataset):
    def __init__(self, data, config, Type):
        self.config = config
        self.data = data
        self.type = Type
        super(MyDataset,self).__init__(name="OD_graph_dataset")

    def process(self):
        self.GEOIDs = []
        self.graphs = []
        self.dises = []
        self.ods = []
        for one in self.data:
            self.GEOIDs.append(one["GEOID"])
            g = dgl.graph(one["od"].nonzero(), num_nodes=one["od"].shape[0])
            g.ndata["nfeat"] = torch.FloatTensor(one["nfeat"])
            self.graphs.append(g)
            self.dises.append(torch.FloatTensor(one["dis"]))
            self.ods.append(torch.FloatTensor(one["od"]))

    def get_size(self, index):
        return self.dises[index].shape[0]

    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, index):
        if self.type != "train":
            return self.GEOIDs[index], self.graphs[index], self.dises[index], self.ods[index]
        if self.config["pert_node"] == 1:
            geoid = self.GEOIDs[index]
            graph = self.graphs[index]
            dis = self.dises[index]
            od = self.ods[index]
            graph, dis, od = permut_nodes_order(graph, dis, od)
            return geoid, graph, dis, od
        else:
            return self.GEOIDs[index], self.graphs[index], self.dises[index], self.ods[index]

    
class SpecificDataset(dgl.data.DGLDataset):
    def __init__(self, data, config, Type):
        self.config = config
        self.data = data
        self.type = Type
        super(SpecificDataset,self).__init__(name="SpecificDataset")

    def process(self):
        self.GEOIDs = []
        self.graphs = []
        self.dises = []
        self.ods = []
        for one in self.data:
            self.GEOIDs.append(one["GEOID"])
            g = dgl.graph(one["od"].nonzero(), num_nodes=one["od"].shape[0])
            g.ndata["nfeat"] = torch.FloatTensor(one["nfeat"])
            self.graphs.append(g)
            self.dises.append(torch.FloatTensor(one["dis"]))
            self.ods.append(torch.FloatTensor(one["od"]))

    def get_size(self, index):
        return self.dises[index].shape[0]

    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, index):
        if self.type != "train":
            return self.GEOIDs[index], self.graphs[index], self.dises[index], self.ods[index]



class MyBatchSampler:
    def __init__(self, dataset, batch_size, max_value):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_value = max_value

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        while indices:
            batch = []
            current_sum = 0
            for _ in range(len(indices)):
                index = choice(indices)
                item_size = self.dataset.get_size(index)
                if (current_sum + item_size <= self.max_value) and (len(batch) < self.batch_size):
                    batch.append(index)
                    current_sum += item_size
                    indices.remove(index)
                else:
                    break

            if len(batch) > 0:
                yield batch

        if batch:
            yield batch

    
    def __len__(self):
        indices = list(range(len(self.dataset)))
        count = 0

        while indices:
            batch = []
            current_sum = 0
            for _ in range(len(indices)):
                index = choice(indices)
                item_size = self.dataset.get_size(index)
                if (current_sum + item_size <= self.max_value) and (len(batch) < self.batch_size):
                    batch.append(index)
                    current_sum += item_size
                    indices.remove(index)
                else:
                    break

            if len(batch) > 0:
                count += 1

        return count