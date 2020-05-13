# from logger.logger import get_logger, setup_logging
#
# logger = get_logger("/home/tanlm/Downloads/logger", "train")
# logger.info("Hello 1")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--train_batch_size", default=3)
parser.add_argument("--vali_batch_size", default=3)
parser.add_argument("--seed", default=1)
parser.add_argument("--workers", default=3)
parser.add_argument("--lr", default=0.0001)
parser.add_argument("--weight_decay", default=4e-5)
parser.add_argument("--max_iters", default=1000)
parser.add_argument("--epoch", default=1000)

args = parser.parse_args()
for key in vars(args):
    print(key, getattr(args, key))
