import os
import argparse
from torch.backends import cudnn
from solver import Solver
from data_loader import return_loader



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('--y_dim', type=int, default=7)
    parser.add_argument('--face_crop_size', type=int, default=256)
    parser.add_argument('--im_size', type=int, default=128)
    parser.add_argument('--g_first_dim', type=int, default=64)
    parser.add_argument('--d_first_dim', type=int, default=64)
    parser.add_argument('--enc_repeat_num', type=int, default=6)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--d_repeat_num', type=int, default=6)
    parser.add_argument('--lambda_cls', type=float, default=1)
    parser.add_argument('--lambda_id', type=float, default=10)
    parser.add_argument('--lambda_bi', type=float, default=10)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--d_train_repeat', type=int, default=5)
    parser.add_argument('--enc_lr', type=float, default=0.0001)
    parser.add_argument('--dec_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0001)

    # Training settings
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--num_epochs_decay', type=int, default=100)
    parser.add_argument('--num_iters', type=int, default=160000)
    parser.add_argument('--num_iters_decay', type=int, default=60000)
    parser.add_argument('--trained_model', type=str, default='')

    # Test settings
    parser.add_argument('--test_model', type=str, default='')

    # Set mode (train or test)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    # Path to save models and logs
    parser.add_argument('--log_path', type=str, default='/main_folder/logs')
    parser.add_argument('--model_path', type=str, default='/main_folder/models')
    parser.add_argument('--sample_path', type=str, default='/main_folder/samples')
    parser.add_argument('--test_path', type=str, default='/main_folder/results')

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=150)
    parser.add_argument('--model_save_step', type=int, default=400)

    config = parser.parse_args()
    print(config)
    cudnn.benchmark = True

    # Create directories if not exist
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.sample_path):
        os.makedirs(config.sample_path)
    if not os.path.exists(config.test_path):
        os.makedirs(config.test_path)
    face_data_loader = return_loader(config.face_crop_size,
                                  config.im_size, config.batch_size, config.mode)
    # Solver
    solver = Solver(face_data_loader,config)

    if config.mode == 'train':
        solver.train()

    elif config.mode == 'test':
        solver.test()

