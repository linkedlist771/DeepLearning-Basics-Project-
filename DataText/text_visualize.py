# 文本可视化

import jieba
import matplotlib.pyplot as plt
import collections

def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f]
    return lines

# 统计词频
def count_words(lines, metod="word"):
    words = []
    if metod == "char":
        for line in lines:
            words += [char for char in line]
    elif metod == "word":
        for line in lines:
            words += jieba.lcut(line)
    counter = collections.Counter(words)
    return counter


# 绘制词频图
def plot_words_frequency(counter, num=10):
    words, counts = zip(*counter.most_common(num))
    #正常显示中文
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
    plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
    plt.xlabel("词")
    plt.ylabel("词频")
    plt.bar(words, counts)
    plt.show()


#绘制词云
def plot_word_cloud(counter, num=10):
    from wordcloud import WordCloud
    wordcloud = WordCloud(font_path='simhei.ttf', background_color='white')
    wordcloud.generate_from_frequencies(counter)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    lines = read_file('原神.txt')
    counter = count_words(lines)
    plot_words_frequency(counter, num=20)
    plot_word_cloud(counter, num=20)