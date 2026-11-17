import argparse


def get_arg_parser():
    parser = argparse.ArgumentParser('MAE abnormal', add_help=False)
    parser.add_argument("--ubnormal_path", type=str, default="/mnt/private/xwy/aed-mae/data/ubnormal")
    parser.add_argument("--input_dataset", type=str, default="/mnt/private/xwy/aed-mae/data/ped2")
    parser.add_argument("--output_dataset", type=str, default="/mnt/private/xwy/aed-mae/data/ped2_masks")
    parser.add_argument("--run_type", type=str, default="abnormal_objects") # 先create_sequences 再abnormal_objects
    parser.add_argument('--target_size', default=(360, 240), type=int, help='images input size')
    args = parser.parse_args()
    return args
