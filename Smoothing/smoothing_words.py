from collections import Counter
import pandas as pd

def smoothing(trends_df):

    words_list = []

    trends_df_list = pd.Series(trends_df.fillna(' ').values.tolist()).str.join(' ')
    trends_df_list= trends_df_list.to_frame()
    trends_df_list = trends_df_list.replace({'gun': ''}, regex=True)
    trends_df_list = trends_df_list.replace({'violenc': ''}, regex=True)
    trends_df_list.columns = ['Trends']

    #Check for top occuring words for each day
    #Counter(" ".join(trends_df_list ["Trends"]).split()).most_common(10)

    for i in range(0, int(trends_df_list.shape[0]), 24):
        # print(i)
        words_list.append(list(Counter(" ".join(trends_df_list['Trends'][i: i + 23]).split()).most_common(10)))

    top_words = pd.DataFrame({'col': words_list})
    top_words.columns = ['Top Words']