import os
import re
import glob
import datetime
import json
# import ast
import pandas as pd
import numpy as np
from pymongo import MongoClient, DESCENDING
import operator
from collections import Counter, defaultdict


# NAMING CONVENTIONS:
# "get" - should be a fast find() or find_one()
# "load" - potentially slow find() that usually returns a large set of data
# "init" - called once and only once when the collection is created
#          (usually for setting indexes)
# "save" - fast update() or update_one(). Avoid using insert() as much
#          as possible, because it is not idempotent.


def get_db(db_name):
  config = configparser.ConfigParser()
  config.read(app_home + './config.ini')
  client = MongoClient('localhost')
  client['admin'].authenticate(config.get('mongo', 'id'), config.get('mongo', 'password'))
  # client = get_MongoClient()
  db = client[db_name]
  return db


# Retrieves the list of brands that was passed to trendinganalysis.py with monthly setting.
# TODO: allow loading by weekly, error handling if no data at all
def get_brands(collection_name):
    data = load_result_data(collection_name, "trending_brandproductperformance_monthly")
    return set(data['brand'])

def get_products(collection_name, brand = 'All_Brands'):
    data = load_result_data(collection_name, "trending_brandproductperformance_monthly")
    if (brand != 'All_Brands'):
        data = data[data['brand'] == brand]
    df = pd.DataFrame(data.groupby(['product_id','title']).number_rev.sum()).reset_index().sort_values('number_rev', ascending=False)
    df['product_id_title'] = df['product_id'] + " has_title " + df['title']
    products = df['product_id_title'].unique()
    products = tuple([tuple(x.split(" has_title ")) for x in products])
    return products

# TODO: dedup get_reviews_* functions if possible.
#get reviews in a category across all brands, products
def get_reviews_category(category,word, *, language=""):
    db = get_db(REVIEW_DB_NAME)
    wd=str(".*"+word+"*")
    if language=="japanese":
        # japanese is not sepalated language
        wd=wd.replace(" ", "")
    resp={}
    cur=db[category].find({"$or":[{"review_text": {'$regex':wd}},{"review_title":{'$regex':wd}}]},{"review_text":1,"product_id":1,'review_date':1,'review_stars':1,'review_title':1,'_id':0}).sort('review_date',-1)[:100]
    resp['count']=db[category].find({"$or":[{"review_text": {'$regex':wd}},{"review_title":{'$regex':wd}}]}).count()
    resp['reviews']=list(cur)
    return resp

def get_reviews_brand(category, word, products, *, language=""):
    db = get_db(REVIEW_DB_NAME)
    wd=str(".*"+word+"*")
    if language=="japanese":
        # japanese is not sepalated language
        wd=wd.replace(" ", "")
    resp={}
    cur=db[category].find({"$and": [{"$or":[{"review_text": {'$regex':wd}},{"review_title":{'$regex':wd}}]}, {"product_id": {"$in": products}} ]},{"review_text":1,"product_id":1,'review_date':1,'review_stars':1,'review_title':1,'_id':0})
    resp['count']=cur.count()
    resp['reviews']=list(cur.sort('review_date',-1)[:100])
    return resp

def get_ranked_products_by_mentions(category, word, products, *, language=""):
    db = get_db(REVIEW_DB_NAME)
    wd=str(".* "+word+" *")
    if language=="japanese":
        # japanese is not sepalated language
        wd=wd.replace(" ", "")
    resp={}
    if (products == 0):
        cur=db[category].find({"$or":[{"review_text": {'$regex':wd}},{"review_title":{'$regex':wd}}]},{"product_id":1, '_id':0})
    else:
        cur=db[category].find({"$and": [{"$or":[{"review_text": {'$regex':wd}},{"review_title":{'$regex':wd}}]}, {"product_id": {"$in": products}} ]},{"product_id":1, '_id':0})
    
    resp['total_count']=cur.count()
    product_ids_list = list(map(operator.itemgetter('product_id'), list(cur)))
    resp['product_ids']= Counter(product_ids_list).most_common(1000)    
    resp['product_ids'] = list(map(list, resp['product_ids']))    
    resp['image_urls'] = list()
    for i in range(len(resp['product_ids'])):
        resp['image_urls'].append(get_image_urls(category, resp['product_ids'][i][0]))  
    resp['brands'] = list()
    for i in range(len(resp['product_ids'])):
        resp['brands'].append(get_brand_from_prod_ids(category, resp['product_ids'][i][0]))
    for i in range(len(resp['product_ids'])):
        resp['product_ids'][i][0] = get_name_from_prod_id_brand(category, resp['product_ids'][i][0])
    return resp

def get_brand_from_prod_ids(category, product_id):    
    db = get_db(PRODUCT_DATA_DB_NAME)
    cur = db[category].find({"product_id": product_id}, {'brand': 1, '_id':0})
    try:
        brand = list(cur)[0]['brand']
    except:
        brand = 'No brand'
    return brand

def get_name_from_prod_id_brand(category, product_id):
    db = get_db(PRODUCT_DATA_DB_NAME)
    try:
        product = (dict(db[category].find_one({'product_id':product_id },{'title':1, "_id":0})))
        product_title = product['title']
        return product_title
    except:
        return 'null'
    

def get_name_to_prod_id_brand(category,product_title):
    db = get_db(PRODUCT_DATA_DB_NAME)
    product_id = (dict(db[category].find_one({'title':product_title },{'product_id':1, "_id":0})))['product_id']
    return product_id
    
#get reviews for a prticular product_id
def get_reviews_product(category,word,product_id,*,language=""):
    db = get_db(REVIEW_DB_NAME)
    resp={}
    wd=str(".*"+word+"*")
    if language=="japanese":
        # japanese is not sepalated language
        wd=wd.replace(" ", "")
    cur=db[str(category)].find({"$and":[{"$or":[{"review_text": {'$regex':wd}},{"review_title":{'$regex':wd}}]},{"product_id":product_id}]},{"review_text":1,"product_id":1,'review_date':1,'review_stars':1,'review_title':1,'_id':0}).sort('review_date',-1)[:100]
    resp['count']=db[category].find({"$and":[{"$or":[{"review_text": {'$regex':wd}},{"review_title":{'$regex':wd}}]},{"product_id":product_id}]}).count()
    resp['reviews']=list(cur)
    return resp

def get_reviews_product_for_month(category,word,product_id, date, *, language=""):
    db = get_db(REVIEW_DB_NAME)
    resp={}
    wd=str(".* "+word+" *")
    if language=="japanese":
        # japanese is not sepalated language
        wd=wd.replace(" ", "")
    cur=db[str(category)].find({"$and":[{"$or":[{"review_text": {'$regex':wd, '$options': 'i'}},
                                                {"review_title":{'$regex':wd, '$options': 'i'}}]},
                                        {"product_id":product_id}, 
                                        {"review_date": {'$regex': str(date + "-*")}}]},{"review_text":1,"product_id":1,'review_date':1,'review_stars':1,'review_title':1,'_id':0}).sort('review_date',-1)[:100]
    resp['count']=db[category].find({"$and":[{"$or":[{"review_text": {'$regex':wd, '$options': 'i'}},{"review_title":{'$regex':wd, '$options': 'i'}}]},
                                             {"product_id":product_id},
                                             {"review_date": {'$regex': str(date + "-*")}}]}).count()
    resp['reviews']=list(cur)
    return resp
    

def load_reviews(collection_name):
    db = get_db(REVIEW_DB_NAME)
    cursor = db[collection_name].find()
    df = pd.DataFrame.from_dict(list(cursor)).astype(object)
    df = df[df['source'] != 'walmart']         # Walmart has some date discrepancies like '0003-01-20' or
                                               # '0006-05-20' that needs to be fixed via regathering data
    df['review_date'] = pd.to_datetime(df['review_date'])
    df = df.replace(np.nan, ' ')
    return df

def load_product_data(collection_name):
    db = get_db(PRODUCT_DATA_DB_NAME)
    cursor = db[collection_name].find()
    df = pd.DataFrame.from_dict(list(cursor))
    df = df.replace(np.nan, ' ')
    return df


# Load and save params (for now only clustering-related ones).
def load_clustering_params(collection_name, param_type):
    db = get_db(PARAMETER_DB_NAME)
    if param_type == "model":        
        return db[collection_name].find_one({"param_type":param_type})
    elif param_type.startswith("strong") or param_type.startswith("verystrong"):
        return db[collection_name].find_one({"type": param_type})["param"]

def save_clustering_params(collection_name, param, param_type):
    db = get_db(PARAMETER_DB_NAME)
    if param_type == "model":
        db[collection_name].insert({"param_type": param_type,"param":param},check_keys=False)  
    elif param_type.startswith("strong") or param_type.startswith("verystrong"):
        db[collection_name].update_one(
            {"type": param_type},
            { "$set": {"type": param_type, "param": param}},
            upsert=True)
    return


# Remove this function entirely, computationally expensive
def load_sentiment_data(collection_name):
    db = get_db(SENTIMENT_DB_NAME)
    cursor = db[collection_name].find()
    df = pd.DataFrame(list(cursor))
    df['review_date'] = pd.to_datetime(df['review_date'])
    return df

def load_sentiment_posneg(collection_name):
    db = get_db(SENTIMENT_DB_NAME)
    cursor = db[collection_name].find({}, {
      "review_date": 1,
      "product_id": 1,
      "review_pos_sentiment": 1,
      "review_neg_sentiment": 1
    })
    df = pd.DataFrame(list(cursor))
    df['review_date'] = pd.to_datetime(df['review_date'])
    return df

# 'load_sentiment_data' function takes long time, as converting huge JSON retrieved from mongDB to 
#  DataFrame is computationally expensive
# The following function only retrieves objects for only specific products
def load_sentiment_data_for_products(collection_name, products):
    db = get_db(SENTIMENT_DB_NAME)
    cursor = db[collection_name].find({"product_id": {"$in": products}})
    df = pd.DataFrame(list(cursor))
    df['review_date'] = pd.to_datetime(df['review_date'])
    return df

def save_sentiment_data(data, collection_name):
    db = get_db(SENTIMENT_DB_NAME)
    for i in range(len(data)):    
        row = {}
        for column in list(data.columns):
            row[column] = data[column].ix[i]        
        db[collection_name].insert(row)   
    return 


def save_result_data(data, collection_name, result_type):
    db = get_db(RESULTS_DB_NAME)
    return db[collection_name].update_one(
              {"type": result_type},
              { "$set": {"type": result_type, "data": data}},
              upsert=True)

def load_result_data(collection_name, result_type):
    db = get_db(RESULTS_DB_NAME)
    res = json.loads(db[collection_name].find_one({"type": result_type})["data"])       #res is a list of dictionaries
    return pd.DataFrame.from_dict(res)



def init_rev_meta(collection_name):
    db = get_db(REV_META_DB_NAME)
    coll = db[collection_name]
    coll.create_index([("date",DESCENDING),
                       ("product_id",DESCENDING)])
    coll.create_index([("year",DESCENDING),
                       ("month",DESCENDING),
                       ("product_id",DESCENDING)])
    coll.create_index([("year",DESCENDING),
                       ("week",DESCENDING),
                       ("product_id",DESCENDING)])
    return

def save_rev_meta(collection_name, meta_objs):
    db = get_db(REV_META_DB_NAME)
    coll = db[collection_name]
    for meta_obj in meta_objs:
      coll.update({
          "date": meta_obj['date'],
          "product_id": meta_obj['product_id']
        },
        meta_obj, upsert=True)
    return

# Example query_params: {"year": 2017, "week": 40, "product_id": "1234"}
def get_rev_meta(collection_name, query_params):
    db = get_db(REV_META_DB_NAME)
    cursor = db[collection_name].find(query_params)
    return pd.DataFrame.from_dict(list(cursor))

# Convenience function for helping with db syntax
def get_rev_meta_for_products(collection_name, product_ids, query_params):
    query_params["product_id"] = {"$in": product_ids}
    return get_rev_meta(collection_name, query_params)

def get_latest_date(collection_name, query_params):
    db = get_db(REV_META_DB_NAME)
    cursor = db[collection_name].find(query_params).sort([('date',DESCENDING)]).limit(1)
    for result in cursor:
      return result['date']

def get_latest_date_for_products(collection_name, product_ids):
    db = get_db(REV_META_DB_NAME)
    return get_latest_date(collection_name, {"product_id": {"$in": product_ids}})



def save_plotly_link(collection_name, filename, url):
    db = get_db(PLOTLY_DB_NAME)
    return db[collection_name].update_one({"filename": filename},
                { "$set": {"filename": filename, "URL": url}}, upsert=True)


def get_plotly_link(collection_name, filename):
    db = get_db(PLOTLY_DB_NAME)
    obj = db[collection_name].find_one({"filename": filename})
    if obj:
        return obj["URL"]
    return None


def get_summary_data(collection_name, filename):
    db = get_db(SUMMARY_DB_NAME)
    obj = db[collection_name].find_one({"filename": filename})
    if obj:
        return obj["summary"]
    return None
    
def save_summary_data(collection_name, filename, summary):
    db = get_db(SUMMARY_DB_NAME)
    return db[collection_name].update_one({"filename": filename},
                { "$set": {"filename": filename, "summary": summary}}, upsert=True)


# Input:
# obj = { "product_id": [...], "price": [...],
#    "salesrank": [...] } where price and salesrank are
#    arrays with alternating date (in epoch seconds) and value
# Output:
# pandas DataFrame with columns for date, price, and salesrank,
#    sorted by date, with date in datetime format.
def convert_tracking_obj_to_dfs(obj):
    # TODO: loop on each field that's not product_id or _id
    # TODO: Return join, have convenience function for pulling the
    #   series back out? for now just leave separate, only 2 and no
    #   immediate plans to scale. Also avoids NaN handling.
    price_arr = [obj["price"][0::2], obj["price"][1::2]]
    pdf = pd.DataFrame(price_arr, index = ["date", "price"])
    pdf = pdf.transpose()
    pdf['price'] = pdf['price'].apply(lambda x: x / 100)
    pdf['date'] = pd.to_datetime(pdf['date'],unit='s')
    pdf = pdf.sort_values('date')
    salesrank_arr = [obj["salesrank"][0::2], obj["salesrank"][1::2]]
    sdf = pd.DataFrame(salesrank_arr, index = ["date", "salesrank"])
    sdf = sdf.transpose()
    sdf['date'] = pd.to_datetime(sdf['date'],unit='s')
    sdf = sdf.sort_values('date')
    return pdf, sdf

def load_tracking_data_for_product(collection_name, product_id):
    db = get_db(TRACKING_DB_NAME)
    obj = db[collection_name].find_one({"product_id": product_id})
    if obj:
      return convert_tracking_obj_to_dfs(obj)
    else:
      emptydf = pd.DataFrame()
      return emptydf, emptydf

def load_tracking_data(collection_name):
    db = get_db(TRACKING_DB_NAME)
    cursor = db[collection_name].find()
    df = pd.DataFrame()
    for obj in cursor:
      pdf, sdf = convert_tracking_obj_to_df(obj)
      obj_df = pdf.join(sdf, how='outer')
      obj_df["product_id"] = obj.product_id
      df.append(obj_df)
    return df


def get_wordfrequencies_product(collection_name, product):
    db = get_db(WORDFREQ_DB_NAME)
    res = (db[collection_name].find_one({"product": product})["vocab_freq"])
    return res

def get_wordfrequencies_brand(collection_name, brand):
    db = get_db(WORDFREQ_DB_NAME)
    res = (db[collection_name].find_one({"brand": brand})["vocab_freq"])
    return res

def save_wordfrequencies_product(collection_name, vocab_freq):
    db = get_db(WORDFREQ_DB_NAME)
    for product in vocab_freq.keys():
        db[collection_name].update_one({'product': product},
                                       {"$set": {'product': product, 'vocab_freq': vocab_freq[product]}}, upsert=True)
    return

def save_wordfrequencies_brand(collection_name, brand, vocab_freq):
    db = get_db(WORDFREQ_DB_NAME)
    db[collection_name].update_one({'brand': brand},
                                   {"$set": {'brand': brand, 'vocab_freq': vocab_freq}}, upsert=True)
    return

def save_wordfrequencies_time(collection_name, time, final_dict):
    db = get_db(WORDFREQ_DB_NAME)
    db[collection_name].update_one({'time': time},
                                   {"$set": {'time': time, 'vocab_freq': final_dict}}, upsert=True)
    return

def save_wordfrequencies_collection(collection_name, final_dict):
    db = get_db(WORDFREQ_DB_NAME)
    for word in final_dict.keys():
        db[collection_name].update_one({'word': word},
                                   {"$set": {'word': word, 'vocab_freq': final_dict[word], 'level': 'Category' }}, upsert=True)
    return


def wordfrequencies_time(collection_name, time):
    db = get_db(WORDFREQ_DB_NAME)
    res = (db[collection_name].find_one({"time": time})["vocab_freq"])
    return res

def wordfrequencies_collection(collection_name, words):
    db = get_db(WORDFREQ_DB_NAME)
    res = defaultdict(list)
    for word in words:
        res[word] = (db[collection_name].find_one({'word': word, 'level': 'Category'})['vocab_freq'])
    return res

def get_all_tokens_collection(collection_name):
    db = get_db(WORDFREQ_DB_NAME)
    cur = db[collection_name].find({'level': 'Category'}, {'_id':0, 'word':1})
    return list(map(operator.itemgetter('word'), list(cur)))

def save_dropdown_list(collection_name, list_):
    db = get_db(WORDFREQ_DB_NAME)
    db[collection_name].update_one({"type": 'dropdown_tokens_list'},
            { "$set": {"type": 'dropdown_tokens_list', "dropdown_tokens_list": list_}}, upsert=True)
    return

def get_reviews_count_by_stars(collection_name, product_id, rating, time):
    db = get_db(REVIEW_DB_NAME)
    return db[collection_name].find({"product_id": product_id, "review_stars": rating, "review_date": {'$regex':time}}).count()

def get_reviews_counts(collection_name, product_id):
    db = get_db(REVIEW_DB_NAME)
    return db[collection_name].find({"product_id": product_id }).count() 
    
def get_reviews_data(category):
    db = get_db(REVIEW_DB_NAME)
    cursor = db[category].find()
    return pd.DataFrame(list(cursor))

def save_ratings_graph_data(category, final_dict):
    db = get_db(RATINGS_DB_NAME)
    for key in final_dict.keys():
        db[category].update_one({"product_id": key},
                                { "$set": {"product_id": key, "timeseries_data": final_dict[key]}}, upsert=True)
    return

def get_ratings_graph_data(category, product_id):
    db = get_db(RATINGS_DB_NAME)
    cur = db[category].find({"product_id": product_id}, {"timeseries_data": 1, "_id": 0})
    return list(cur)[0]

def save_ratings_breakdown_data(category, product_id, ratings_breakdown_list):
    db = get_db(RATINGS_DB_NAME)
    db[category].update({"product_id": product_id},
                        { "$set": {"ratings_breakdown": ratings_breakdown_list}})
    return

def get_ratings_breakdown_data(category, product_id):
    db = get_db(RATINGS_DB_NAME)
    cur = db[category].find({"product_id": product_id}, {"ratings_breakdown": 1, "_id": 0})
    return list(cur)[0]

def save_product_matching_data(category, final_dict, product_id):
    db = get_db(PRODUCT_MATCHING_DB_NAME)
    db[category].update_one({"product_id": product_id},
                            { "$set": final_dict}, upsert=True)
    return

def get_product_matching_data(category, product_id):
    db = get_db(PRODUCT_MATCHING_DB_NAME)
    cur = db[category].find({"$or": [{"sim_product_id": product_id}, {"product_id": product_id}]}, {"_id": 0, 'source': 1, 'sim_source':1})
    objs = list(cur)
    if (objs != []):
        source_list = []
        for i in objs:
            source_list.append(i['source'])
            source_list.append(i['sim_source'])
        return list(set(source_list))
    else:
        db = get_db(PRODUCT_DATA_DB_NAME)
        cur = db[category].find({"product_id": product_id})
        obj = list(cur)
        '''
        NOTE: The following scheme is done to return 'No Source' when a particular product_id is
              present in product_reviews database, but not in product_data
        '''
        if (obj != []):
            return [obj[0]['source']]
        else:
            return 'No Source'

def get_image_urls(category, product_id):
    db = get_db(PRODUCT_DATA_DB_NAME)
    cur = db[category].find({"product_id": product_id}, {'images.url': 1, '_id':0})
    try:
        url = list(cur)[0]['images'][0]['url']
    except:
        url = 'No url'
    return url

def get_descriptions_and_matched_products(category, product_id):
    db = get_db(PRODUCT_MATCHING_DB_NAME)
    cur = db[category].find({"sim_product_id": product_id}, {"_id": 0, 'product_id': 1})
    objs = list(cur)
    if (objs != []):
        product_id_list = [product_id]
        for i in objs:
            product_id_list.append(i['product_id'])
        product_id_list = list(set(product_id_list))
    else:
        product_id_list = [product_id]
    
    descriptions_list = list()
    pd_db = get_db(PRODUCT_DATA_DB_NAME)
    for product_id in product_id_list:
        cur = pd_db[category].find({"product_id": product_id}, {'description':1, '_id':0, 'source': 1})
        obj = list(cur)
        if (obj != []):
            description_dict = obj[0]
            descriptions_list.append({"description": description_dict['description'],
                                  "source": description_dict['source']})
        else:
            # NOTE: return empty dict when a particular product_id is present in product_reviews database, but not in product_data
            descriptions_list.append({})
    return descriptions_list, product_id_list


def get_dropdown_tokens_list(category, type_):
    db = get_db(WORDFREQ_DB_NAME)
    cur = db[category].find({"type": type_}, {'dropdown_tokens_list':1, '_id':0})
    return list(cur)[0]['dropdown_tokens_list']

def get_attribute_sentiment(product_id, category):
    db = get_db(ATTRIBUTE_SENTIMENT_DB_NAME)
    '''
    The 'try-except' prevents the 'topic-trends' and 'topics' page from breaking
    TODO: Remove it when 'attribute_sentiment.py' is run majority of brands
    '''
    try:
        cur = db[category].find({'product_id': product_id}, {'attribute_sentiment':1, '_id':0})
        return list(cur)[0]['attribute_sentiment']
    except:
        return False

def get_attr_sent_dropdown_dict(category, type_):
    db = get_db("attribute_sentiment")
    cur = db[category].find({"type": type_}, {'attr_sent_dropdown_dict':1, '_id':0})
    return list(cur)[0]['attr_sent_dropdown_dict']

# Load and save plotly chart URLs with their associated chart formatting
#   information (legend, axes, type, etc)
# "dtm" is the data type mapping object from results.py
#   It MUST be sorted when saved or loaded from the db (see sort_dtm)
# The goal of these functions is to save and load the charts based on what they look like
#   and what's in them, rather than an arbitrary filename which could get overwritten unintentionally.
# TODO: revisit these and results.py to dedup all the plot creation in models.py and possibly
#   use other chart types/double axes/etc.


# Note: this changes a dtm dict into a nested string, ex:
# "{\"field\": \"trending_brandperformance\", \"insight\": \"obv\", \"layout\": \"{\\\"bar\\\": {\\\"name\\\": \\\"brand\\\", \\\"x\\\": \\\"review_date_agg\\\", \\\"y\\\": \\\"review_pos_sentiment\\\"}, \\\"line\\\": {\\\"name\\\": \\\"brand\\\", \\\"x\\\": \\\"review_date_agg\\\", \\\"y\\\": \\\"review_pos_sentiment\\\"}}\"}"
# Very ugly to look at, but preserves order well.
#def sort_dtm(dtm):
#    dtm_copy = dtm.copy()   # avoid changing the passed obj
#    dtm_copy["layout"] = json.dumps(dtm_copy["layout"], sort_keys=True)   # sort_keys is not recursive
#    return json.dumps(dtm_copy, sort_keys=True)

#def save_plotly_link(collection_name, filename, dtm, url):
#    db = get_plotly_db()
#    sorted_dtm = sort_dtm(dtm)
#    return db[collection_name].update_one({"sorted_dtm": sorted_dtm},
#                { "$set": {"filename": filename, "sorted_dtm": sorted_dtm, "dtm": dtm, "URL": url}}, upsert=True)

#def get_plotly_link(collection_name, dtm):
#    db = get_plotly_db()
#    obj = db[collection_name].find_one({"sorted_dtm": sort_dtm(dtm)})
#   if obj:
#      return obj["URL"]
#    return None

def get_stopwords(*, suffix=''):
    db = get_db(DICTIONARY_DB_NAME)
    word_objs = db['stopword'+suffix].find({}, {'headword':1, '_id':0})
    return [w['headword'] for w in word_objs]

def get_politely_score(headword, *, suffix=''):
    db = get_db(DICTIONARY_DB_NAME)
    score_obj = db['politely'+suffix].find_one({'headword':headword}, {'score':1, '_id':0})
    return score_obj['score'] if score_obj else 0

def bulk_search_for_politely_dict(headword_list, *, suffix=''):
    db = get_db(DICTIONARY_DB_NAME)
    p_obj = db['politely'+suffix].find({'headword':{'$in':headword_list}}, {'headword':1, 'score':1, '_id':0})
    politely_dict = dict()
    politely_dict.update({p['headword']:p['score'] for p in p_obj})
    # add some words with 0 score that are unfound in dictionaries..politely
    politely_dict.update({h:0 for h in headword_list if not h in politely_dict})
    return politely_dict
