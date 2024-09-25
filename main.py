from configparser import ConfigParser
from core import ActiveLearning
import argparse

# Read in the configurations from the params.ini file
config = ConfigParser()
config.read('params.ini')

if __name__ == '__main__':
    # get the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=str, help='Dataset name')
    parser.add_argument('--r', type=int, help='random seed')
    parser.add_argument('--a', type=str, help='acquisition function')
    parser.add_argument('--m', type=str, help='model name')
    args = parser.parse_args()
    al = ActiveLearning(
        acq_func=args.a,
        dataset_name=args.d,
        random_seed=args.r,
        model_name=args.m
    )
    # start Active Learning! :)
    try:
        al.run()
    except Exception as e:
        file_path = al.data_logger.log_primary_results(al.test_loss_list, al.best_dropout_rate_list, al.best_l2_reg_list, al.test_accuracy_list)
        al.log_params()
        print("An error occurred! but the results are saved!")
        print(e)