# from wordcloud import WordCloud
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from collections import defaultdict


def wordcloud_viz(X, y, start_time, generate_wc=False):
    vocab = defaultdict(int)
    for text in X['final_text'].values:
        for elem in text.split(' '):
            vocab[elem] += 1
    #
    # if generate_wc:
    #     # Now we look at the types of words in ham and spam. We plot wordclouds for both
    #     nt_text=' '.join(X.loc[y==0,'final_text'].values)
    #     nt_wordcloud = WordCloud(background_color='white',max_words=2000).generate(nt_text)
    #     gv_text=' '.join(X.loc[y==1,'final_text'].values)
    #     gv_wordcloud = WordCloud(background_color='white',max_words=2000).generate(gv_text)
    #     plt.figure(figsize=(12,4))
    #     plt.subplot(1,2,1)
    #     plt.imshow(gv_wordcloud,interpolation='bilinear')
    #     plt.title('GUN VIOLENCE _' + str(start_time))
    #     plt.axis('off')
    #     plt.subplot(1,2,2)
    #     plt.imshow(nt_wordcloud, interpolation='bilinear')
    #     plt.axis('off')
    #     plt.title('NEGATIVE TWEETS_' + str(start_time))
    #     plt.savefig('wordCloud/WORDCLOUD_' + str(start_time) + '.png')
    return vocab
