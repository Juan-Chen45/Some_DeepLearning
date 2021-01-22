import pandas as pd

# 将数据情感标注为empty的行全部去掉
# df = pd.read_csv(r"emotion_data.csv",encoding="utf-8")
# df_clear = df.drop(df[df["sentiment"]=="empty"].index)
# df_clear = df_clear.reset_index(drop=True)
# print(df_clear)
# df_clear.to_csv("emotion_data.csv",index=None,encoding="utf-8")

# 将sentiment全部转换为label
# df = pd.read_csv(r"emotion_data.csv", encoding="utf-8")
# df.loc[df["sentiment"] == "sadness", "sentiment"] = 0
# df.loc[df["sentiment"] == "enthusiasm", "sentiment"] = 1
# df.loc[df["sentiment"] == "neutral", "sentiment"] = 2
# df.loc[df["sentiment"] == "worry", "sentiment"] = 3
# df.loc[df["sentiment"] == "fun", "sentiment"] = 4
# df.loc[df["sentiment"] == "hate", "sentiment"] = 5
# df.loc[df["sentiment"] == "love", "sentiment"] = 6
# df.loc[df["sentiment"] == "surprise", "sentiment"] = 7
# df.loc[df["sentiment"] == "relief", "sentiment"] = 8
# df.loc[df["sentiment"] == "happiness", "sentiment"] = 9
# df.loc[df["sentiment"] == "boredom", "sentiment"] = 10
# df.loc[df["sentiment"] == "happiness", "sentiment"] = 11
# df.loc[df["sentiment"] == "anger", "sentiment"] = 12
# df.to_csv(r"emotion_data.csv",index=None,encoding="utf-8")
# print(df)

# 去除content中的 所有@XXX 这种无用字段
# df = pd.read_csv(r"emotion_data.csv", encoding="utf-8")
# 全部转换为小写,去除前后空行
# df["content"] = df["content"].str.lower().str.strip()
# # 去除@XXX这种无用的表达和里面自带的一些”“双引号
# df["content"] = df["content"].str.replace(r"@[a-zA-Z]*", "").str.strip()
# df["content"] = df["content"].str.replace('\"', "").str.strip()
# # 去除掉所有的空行
# df = df.dropna(axis=0)
# df = df.reset_index(drop=True)
# df.to_csv(r"emotion_data.csv", index=None, encoding="utf-8")
# print(df)

# 情感分析中不需要去除停用词，去掉之后效果可能不太好

'''
1.去除非文本部分
2.分词
3.去除停用词
4.对英文单词进行词干提取(stemming)和词型还原(lemmatization)
5.转为小写
'''
# 去除非文本部分
# filter_words =  "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
# df = pd.read_csv(r"emotion_data.csv", encoding="utf-8")
# df["content"] = df["content"].str.replace(filter_words,"").str.strip()
# df.to_csv(r"emotion_data.csv", index=None, encoding="utf-8")
# print(df)

# import pandas as pd
# import os
#
# all_files = []
# listFile = os.walk(r"C:\Users\70983\Desktop\aclImdb_v1\aclImdb\test\pos")
# for dirPath, dirName, fileName in listFile:
#     for names in fileName:
#         all_files.append(os.path.join(dirPath, names))
#
# df_pos = pd.DataFrame(columns=["content","label"])
# file_content = []
# for file in all_files:
#     with open(file,"r",encoding="utf-8")as f:
#         file_content.append(f.read())
#         print("processing ",file)
#
# df_pos["content"] = file_content
# df_pos["label"] = [1 for i in range(len(file_content))]
#
#
# all_files = []
# listFile = os.walk(r"C:\Users\70983\Desktop\aclImdb_v1\aclImdb\test\neg")
# for dirPath, dirName, fileName in listFile:
#     for names in fileName:
#         all_files.append(os.path.join(dirPath, names))
#
# df_neg = pd.DataFrame(columns=["content","label"])
# file_content = []
# for file in all_files:
#     with open(file,"r",encoding="utf-8")as f:
#         file_content.append(f.read())
#         print("processing ",file)
#
# df_neg["content"] = file_content
# df_neg["label"] = [0 for i in range(len(file_content))]
#
# frames = [df_pos,df_neg]
# result = pd.concat(frames)
#
# print(result)
# result.to_csv(r"C:\Users\70983\Desktop\aclImdb_v1\aclImdb\test\test_data.csv",index=False)