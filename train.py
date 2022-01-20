import warnings

import src.utils as utils
import src.configuration as C
import src.models as models
import src.callbacks as clb
import src.runners as rnr


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    args = utils.get_parser().parse_args()
    config = utils.load_config(args.config)

    global_params = config["globals"]
    utils.set_seed(global_params["seed"])

    device = C.get_device(global_params["device"])

    train_dl = C.get_loader(config=config, phase="train")
    valid_dl = C.get_loader(config=config, phase="val")

    loaders = {"train": train_dl, "valid": valid_dl}

    model = models.get_model(config).to(device)

    # for param in model.pose_model.model0.parameters():
    #     param.requires_grad = False
    for param in model.base.vgg_base.parameters():
        param.requires_grad = False

    criterion = C.get_criterion(config).to(device)
    optimizer = C.get_optimizer(model, config)
    scheduler = C.get_scheduler(optimizer, config)
    callbacks = clb.get_callbacks(config)

    runner = rnr.get_runner(config)
    runner.train(
        model=model,
        criterion=criterion,
        loaders=loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=global_params["num_epochs"],
        verbose=True,
        callbacks=callbacks)
