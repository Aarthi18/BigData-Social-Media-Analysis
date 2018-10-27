from elasticsearch import Elasticsearch
from datetime import timedelta, datetime
import numpy as np
import numpy.random
import pandas as pd
from settings import weblink

def dataCollection(start_time):
    #origin = datetime(2018, 7, 19, 20, 0, 0)
    #start_time = datetime(2018, 8, 20, 14, 0, 0)
    # start_time = origin
    #diff = datetime.utcnow() - start_time
    #total_hours = int(np.ceil(diff.total_seconds() / 3600))

    #print("******", "Extracting tweets for hour", start_time, "******", sep=" ")
    gte = start_time.strftime("%Y-%m-%dT%H:%M:%S")
    lt = (start_time + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")
    es = Elasticsearch(weblink)
    doc = {
        'size': 10000,
        'query': {
            'range': {
                '@timestamp': {
                    "gte": gte,
                    "lt": lt
                }
            }
        }
    }

    # Data for positive Tweets
    gv_first_page = es.search(index="gun_violence", doc_type='doc', body=doc, scroll='1m')
    sid = gv_first_page['_scroll_id']
    gv_scroll_size_total = gv_scroll_size = gv_first_page['hits']['total']
    gv_final_hits = gv_first_page['hits']['hits']
    print("GV_scroll_size :", gv_scroll_size)
    while (gv_scroll_size > 0):
        # print("Scrolling...")
        gv_new_page = es.scroll(scroll_id=sid, scroll='1m')
        new_hits = gv_new_page['hits']['hits']
        # Update the scroll ID
        sid = gv_new_page['_scroll_id']
        # Get the number of results that we returned in the last scroll
        gv_scroll_size = len(gv_new_page['hits']['hits'])
        # print("scroll size: " + str(gv_scroll_size))
        gv_final_hits = gv_final_hits + new_hits

    json_gunViolence = gv_final_hits
    gunViolence = [d['_source'] for d in json_gunViolence]
    gunViolence = pd.DataFrame(gunViolence)
    gunViolence['type'] = 'gun violence'
    # gunViolence["text"] = gunViolence["text"].str.replace("gun", "")
    # gunViolence["text"] = gunViolence["text"].str.replace("violence", "")
    # gunViolence["text"] = gunViolence["text"].str.replace("active", "")
    # gunViolence["text"] = gunViolence["text"].str.replace("shooter", "")
    gunViolence["text"] = gunViolence["text"].str.replace('http\S+|www.\S+', '', case=False)
    # print("Shape before removing retweets:", np.shape(gunViolence))
    gunViolence = gunViolence[gunViolence.text.str.contains("RT") == False].reset_index(drop=True)  # Remove all retweets
    # print("Shape after removing retweets:", np.shape(gunViolence))

    # Data for negative Tweets
    non_gv_first_page = es.search(index="negative_sample", doc_type='doc', body=doc, scroll='1m')
    sid = non_gv_first_page['_scroll_id']
    non_gv_scroll_size_total = non_gv_scroll_size = 2 * gv_first_page['hits']['total']
    non_gv_final_hits = non_gv_first_page['hits']['hits']
    # print("NT_scroll_size :", non_gv_scroll_size)
    while (non_gv_scroll_size > 0):
        # print("Scrolling...")
        non_gv_new_page = es.scroll(scroll_id=sid, scroll='1m')
        new_hits = non_gv_new_page['hits']['hits']
        # Update the scroll ID
        sid = non_gv_new_page['_scroll_id']
        # Get the number of results that we returned in the last scroll
        non_gv_scroll_size = len(non_gv_new_page['hits']['hits'])
        # print("scroll size: " + str(non_gv_scroll_size))
        non_gv_final_hits = non_gv_final_hits + new_hits

    json_negativeTweets = non_gv_final_hits
    negativeTweets = [d['_source'] for d in json_negativeTweets]
    negativeTweets = pd.DataFrame(negativeTweets)
    negativeTweets['type'] = 'negative tweets'
    negativeTweets["text"] = negativeTweets["text"].str.replace('http\S+|www.\S+', '', case=False)
    # print("Shape before removing retweets:", np.shape(negativeTweets))
    negativeTweets = negativeTweets[negativeTweets.text.str.contains("RT") == False].reset_index(drop=True)  # Remove all retweets
    # print("Shape after removing retweets:", np.shape(negativeTweets))

    # Concat both tweets
    tweets = pd.concat([gunViolence, negativeTweets])
    tweets = tweets.iloc[np.random.permutation(len(tweets))].reset_index(drop=True)
    tweets['text'] = tweets['text'].apply(str)

    data = tweets[['type', 'text']]
    #TODO dropna generates warning
    data.dropna(inplace=True)
    data['type'].value_counts()
    return start_time, data, gv_scroll_size_total, non_gv_scroll_size_total
