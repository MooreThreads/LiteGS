import litegs.arguments
import litegs


if __name__ == "__main__":
    cfg = litegs.arguments.get_config(litegs.arguments.TrainConfig)
    litegs.training.start(cfg)