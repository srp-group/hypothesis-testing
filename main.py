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
    args = parser.parse_args()
    # start Active Learning! :)
    try:
        al = ActiveLearning(dataset_name=args.d, random_seed=args.r)
        al.run()
    except Exception as e:
        file_path = al.data_logger.log_primary_results(al.test_loss_list, al.best_dropout_rate_list, al.best_l2_reg_list, al.test_accuracy_list)
        al.visualizer.plot_results(file_path)
        al.visualizer.plot_primary_results(al.test_loss_list, al.best_dropout_rate_list, al.best_l2_reg_list, al.test_accuracy_list)
        print("An error occurred! but the results are saved!")
        print(e)