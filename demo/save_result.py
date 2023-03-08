import argparse
import logging
import sys

import numpy as np
import torch
import tqdm
from torch.backends import cudnn

sys.path.append('..')

from fastreid.evaluation import evaluate_rank
from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.data import build_reid_test_loader
from predictor import FeatureExtractionDemo

cudnn.benchmark = True
logger = logging.getLogger('fastreid.visualize_result')


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        '--parallel',
        action='store_true',
        help='if use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--dataset-name",
        help="a test dataset name for visualizing ranking list."
    )

    parser.add_argument(
        "--dir_query_csv",
        help="directory path to query csv file."
    )

    parser.add_argument(
        "--dir_gallery_csv",
        help="directory path to query csv file."
    )

    parser.add_argument(
        "--dir_submit_csv",
        help="directory path to query csv file."
    )

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    logger = setup_logger()
    cfg = setup_cfg(args)
    test_loader, num_query = build_reid_test_loader(cfg, dataset_name=args.dataset_name)

    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)

    import csv

    query = []
    with open(args.dir_query_csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count > 0:
                query.append(row[1])
            line_count += 1

    gallery = []
    with open(args.dir_gallery_csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count > 0:
                gallery.append(row[1])
            line_count += 1

    logger.info("Start extracting image features")
    feats = []
    pids = []
    camids = []
    for (feat, pid, camid) in tqdm.tqdm(demo.run_on_loader(test_loader), total=len(test_loader)):
        feats.append(feat)
        pids.extend(pid)
        camids.extend(camid)

    feats = torch.cat(feats, dim=0)
    q_feat = feats[:num_query]
    g_feat = feats[num_query:]
    q_pids = np.asarray(pids[:num_query])
    g_pids = np.asarray(pids[num_query:])
    q_camids = np.asarray(camids[:num_query])
    g_camids = np.asarray(camids[num_query:])

    # compute cosine distance
    distmat = 1 - torch.mm(q_feat, g_feat.t())
    distmat = distmat.numpy()

    array_ids = []

    import re

    for item in distmat:
        for item_small in np.sort(item):
            string1 = str(np.where(item == item_small))
            index_num = int(re.search(r'\d+', string1).group())
            if not "gallery_add_on" in gallery[index_num]:
                array_ids.append(gallery[index_num])
                break

    counter = 0
    new_query = []
    new_result = []
    import re
    for item in array_ids:
        text = (query[counter])
        m = re.search('query/(.+?).jpg', text)
        if m:
            found = m.group(1)
            new_query.append(found)

        text = item
        m = re.search('test/(.+?)C', text)
        if m:
            found = m.group(1)
            new_result.append(found)
        counter += 1

    import pandas as pd

    submit = {'image_id': new_query, 'person_id': new_result}
    submit_df = pd.DataFrame(submit)
    submit_df.to_csv(args.dir_submit_csv, index=False)
    print('result saved as ', args.dir_submit_csv)

