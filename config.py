import argparse


class Config():
    def __init__(self) -> None:
        self.args = self.get_args()

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(
            description="2023 FSU YSP Heart Disease Data Imputation Model")  # create ArgumentParser

        ##### Format of add_argument #####
        # name ==> referenced in terminal to input values (e.g. --nclients 50), no spaces allowed
        # data type
        # default value
        # choices ==> if anything else is input besides the choices are inputted, an exception will be thrown

        # Training settings (global)
        parser.add_argument('--dataset', type=str, default='heart_disease', choices=['heart_disease'],
                            help='The dataset to be trained')
        parser.add_argument('--nclients', type=int, default=4, help='Number of users participating in FL')
        parser.add_argument('--epoch', type=int, default=1000, help='Total number of global training epochs')
        parser.add_argument('--global_batch_size', type=int, default=64, help='Global batch size')

        # Model settings (local)
        parser.add_argument('--local_epoch', type=int, default=3, help='Number of local training epochs')
        parser.add_argument('--optim', type=str, default='Adam', help='Local optimizer')
        parser.add_argument('--local_batch_size', type=int, default=16, help='Local batch size for each step')
        parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate of local optimizer')

        # Reads and parses input
        args = parser.parse_args()
        return args
