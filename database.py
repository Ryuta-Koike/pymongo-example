# -*- coding: utf-8 -*-
import os, sys, logging, time, configparser
from pymongo import MongoClient,  DESCENDING
import pandas as pd

# app_home = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# # sys.path.append(app_home)
# sys.path.append(os.path.join(app_home, "lib"))

LOG_LEVEL = 'DEBUG'
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s [%(levelname)s] %(module)s | %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Const of database name
DICTIONARY_DB = "dicttionaries"
PL_COLLECTION_NAME = "politely"

POLITELY_DICT_DB = "politely_dict"
NPTABLE_COLLECTION_NAME = "np_table"
SENTIDIC_COLLECTION_NAME = "sent_dic"

# NAMING CONVENTIONS:
# "get" - should be a fast find() or find_one()
# "load" - potentially slow find() that usually returns a large set of data
# "init" - called once and only once when the collection is created
#          (usually for setting indexes)
# "save" - fast update() or update_one(). Avoid using insert() as much
#          as possible, because it is not idempotent.


def get_db(db_name):
    config = configparser.ConfigParser()
    config.read( './config.ini')
    client = MongoClient('localhost')
    client['admin'].authenticate(config.get('mongo', 'id'), config.get('mongo', 'password'))
    # client = get_MongoClient()
    db = client[db_name]
    return db

def load_politly_dic(collection_name):
    db = get_db(POLITELY_DICT_DB)
    cursor = db[collection_name].find()
    df = pd.DataFrame.from_dict(list(cursor)).astype(object)
    return df

# TODO: multi word (ex.あきれる た
def get_sentidic_score(headword, *, type=None):
    db = get_db(POLITELY_DICT_DB)
    # TODO: type
    res = db[SENTIDIC_COLLECTION_NAME].find_one({'headword':headword})
    logger.info(res)
    score = 0
    if res:
        score = res['score']
        return score

if __name__ == "__main__":
    df = load_politly_dic(SENTIDIC_COLLECTION_NAME)
    print(df.shape, df.dtypes, df.columns, df.index)
    print(df.head(10))

    words = ['あきれる', '優れる']
    for w in words:
        print('extract of '+w, df[df['headword']==w])

    # pandas tips
    # ----------------------------------------
    # df = df[df['source'] != 'some value'] 
    # df['date'] = pd.to_datetime(df['date'])
    # df = df.replace(np.nan, ' ')

    # update syntax
    # ----------------------------------------
    # db[NPTABLE_COLLECTION_NAME].update({'headword':'優れる', 'POS_jp':'動詞' },
    #                                    {'$set':{'headword':'優れる', 'reading': 'すぐれる', 'POS':'VERB', 'POS_jp':'動詞', 'eng':'be excellent'}},
    #                                    upsert=True)
