import hydra

from simple_ijepa.config import TrainConfig
from simple_ijepa.training.ddp_trainer import IJEPATrainerDDP


@hydra.main(version_base=None, config_name="train_config")
def main(cfg: TrainConfig) -> None:
    trainer = IJEPATrainerDDP(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
