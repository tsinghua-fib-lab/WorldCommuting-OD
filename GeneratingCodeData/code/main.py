from torch.utils.data import DataLoader
from dgl.dataloading import GraphDataLoader

from train import *
from eval import GEN_specific, GEN_specific_UK, GEN_specific_GHSL
from utils.procedure import *
from utils.tool import collate_fn
from data_load import *
from model import *
from utils.MyLogger import Logger


def main(config):
    # meta
    print("\n****** experiment name:", config["exp_name"], " ******\n")
    setRandomSeed(config["random_seed"])

    # load data
    data, trainSet, validSet, scalers = prepare_data(config)
    config["num_train_samples"] = len(trainSet)
    config["num_valid_samples"] = len(validSet)
    
    # set input dim for model
    config["n_indim"] = data[0]["nfeat"].shape[1]
    config["e_indim"] = 1
    config["n_outdim"] = config["n_indim"]
    config["e_outdim"] = 1
    config["img_dim"] = data[0]["imgfeat"].shape[1]

    print("  ** constructing dataloader...", end="")
    train_set = MyDataset(trainSet, config, "train")
    valid_set = MyDataset(validSet, config, "valid")
    tnBS = MyBatchSampler(train_set, config["batch_size"], config["max_nodes"])
    vdBS = MyBatchSampler(valid_set, config["batch_size"], config["max_nodes"])
    train_Dloader = GraphDataLoader(train_set, batch_sampler=tnBS, collate_fn=collate_fn)
    valid_Dloader = GraphDataLoader(valid_set, batch_sampler=vdBS, collate_fn=collate_fn)
    print("done")

    # logger
    print("  ** preparing logger...", end="")
    logger = Logger(config)
    if config["mode"] == "CONTINUE":
        logger.load_exp_log()
    print("done")

    # model
    print("  ** preparing model...", end="")
    diff_model = Diffusion(config).to(config["device"])
    if config["mode"] == "CONTINUE":
        path = logger.model_path[:-4] + "_lastest.pkl"
        state_dict = torch.load(path, map_location=config["device"])
        diff_model.load_state_dict(state_dict)
    print("done \n")

    if config["mode"] != "EVAL":
        # train
        train(config, diff_model, train_Dloader, valid_Dloader, logger, scalers)

    # test
    # load best valid model first
    diff_model.load_state_dict(torch.load(logger.model_path[:-4] + "_best.pkl", map_location=config["device"]))
    if config["test_specific"] == 1:
        # # global cities
        # GEN_Dloader = prepare_specific_dataloader(config, scalers)
        # GEN_specific(config, diff_model, GEN_Dloader, logger, scalers)
        
        # # UK
        # UK_Dloader = prepare_UK_dataloader(config, scalers)
        # GEN_specific_UK(config, diff_model, UK_Dloader, logger, scalers)

        # # 6 specific cities for eval
        # cities_6_Dloader = prepare_6_TEST_cities_dataloader(config, scalers)
        # GEN_specific_6TESTcities(config, diff_model, cities_6_Dloader, logger, scalers)

        # 733 cities for GHSL
        GHSL_Dloader = prepare_GHSL_dataloader(config, scalers)
        GEN_specific_GHSL(config, diff_model, GHSL_Dloader, logger, scalers)

    print("  ** Finished!")



if __name__ == "__main__":

    # normal train
    config = get_config("exp/config/us.json")
    main(config)