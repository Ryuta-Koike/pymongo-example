import argparse
import os, sys, logging, time
from termcolor import colored
import pandas as pd
from collections import defaultdict

import sys, os
# PYMONGO_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
PYMONGO_DIR = os.path.abspath(os.path.join( os.path.dirname(os.path.abspath(__file__)) , ".." ))
sys.path.insert(0, PYMONGO_DIR)
from database import save_dict

LOG_LEVEL = 'DEBUG'
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s [%(levelname)s] %(module)s | %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# mode: labelToScore
def conv_label2score(dic):
    l =  dic['label']
    dic['score'] = 1 if l=='p' else -1 if l=='n' else 0
    return dic

def save_data_from_file(file_name, delimiter, flg_have_header, mode):
    # load file
    logger.info(file_name)
    if flg_have_header:
        pd_dic = pd.read_csv(file_name, sep=delimiter)
    else:
        pd_dic = pd.read_csv(file_name, sep=delimiter, header=None)
    # print('loaded file:', pd_dic.shape, 'columns', pd_dic.columns, pd_dic.head(10))

    if mode == "labelToScore":
        print(colored("using mode of " + mode,"green"))
        final_dict = [conv_label2score(dic) for index, dic in pd_dic.to_dict(orient="index").items() if index!=0]
    else:
        final_dict = [dic for index, dic in pd_dic.to_dict(orient="index").items() if index!=0]
    # print(final_dict[0])
    save_dict("politely_JP", final_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("MODE", help="Specify mode",type=str)
    parser.add_argument("-d", "--DELIMITER", help="Specify delimiter",type=str)
    parser.add_argument("--HEADER", help="Y/N, Does a file have header line?",type=str)
    parser.add_argument("-m", "--MODE", help="Specify mode, normal or labelToScore",type=str,
                        nargs='?',default='normal',const='normal')
    parser.add_argument("-f", "--FILE", help="Specify file. default=this",type=str)
    args = parser.parse_args()
    
    FILE = args.FILE
    DELIMITER = args.DELIMITER
    HEADER = True if args.HEADER=='Y' else False
    MODE = args.MODE
    print(colored("start loading of "+ FILE + "with delimiter of ["+ DELIMITER+ "]", "green"))
    save_data_from_file(FILE, DELIMITER, HEADER, MODE)
    print(colored("finished all procesure", "red"))
