import os
import json
import yaml
import torch
import argparse
from pytorch_lightning import seed_everything

from main_model import TSB_eICU
from dataset import get_dataloader, get_ext_dataloader
from utils import train, evaluate, test
from tab_transformer import FTTransformer



def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MortiGen")
    parser.add_argument("--config", type=str, default="base.yaml")
    parser.add_argument('--device', default='cuda:0', help='Device for Attack')
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--testmissingratio", type=float, default=0.1)
    parser.add_argument("--nfold", type=int, default=10)
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument("--nsample", type=int, default=100)

    return parser.parse_args()


def load_config(config_path):
    """Load the configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_model(model_name):
    """
    Finds a pre-trained model, downloading it if necessary.
    Alternatively, loads a model from a local path.
    """
    assert os.path.isfile(model_name), f'Could not find model checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    return checkpoint.get("ema", checkpoint)  # supports checkpoints from train.py


def load_model_weights(model, checkpoint_path):
    """Load model weights from a checkpoint file."""
    state_dict = find_model(checkpoint_path)
    model.load_state_dict(state_dict)
    del state_dict


def main():
    args = parse_arguments()
    print(args)

    seed_everything(args.seed)

    config_path = os.path.join("config", args.config)
    config = load_config(config_path)

    config["model"]["test_missing_ratio"] = args.testmissingratio
    print(json.dumps(config, indent=4))

    train_loader, valid_loader = get_dataloader(
        seed=args.seed,
        nfold=args.nfold,
        batch_size=config["train"]["batch_size"],
        missing_ratio=config["model"]["test_missing_ratio"],
    )

    ext_loader = get_ext_dataloader(seed=0, batch_size=2, missing_ratio=0)

    model = TSB_eICU(config, args.device).to(args.device)

    model1 = FTTransformer(
        categories=[2, 2, 2, 2, 2, 2, 2, 2, 2, 5],
        num_continuous=31,
        dim=32,
        dim_out=1,
        depth=6,
        heads=8,
        attn_dropout=0.1,
        ff_dropout=0.1
    ).to(args.device)

    if args.modelfolder == "":
        train(
            model,
            model1,
            config["train"],
            train_loader,
            valid_loader=valid_loader
        )

    # Load the pre-trained models
    load_model_weights(model, 'E:\\diabetic\\diabetic\\AMIend2end\\logs\\model.pth')
    load_model_weights(model1, 'E:\\diabetic\\diabetic\\AMIend2end\\logs\\model1.pth')

    test(model, model1, config["train"], ext_loader)


if __name__ == "__main__":
    main()

