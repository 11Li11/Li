import pickle
import numpy as np
import paddle
mov_feat_dir = './mov_feat.pkl'
usr_feat_dir = './usr_feat.pkl'

import os
from PIL import Image
import json

def recommend_mov_for_usr(usr_id, top_k, pick_num, usr_feat_dir, mov_feat_dir, mov_info_path):
    assert pick_num <= top_k
    # 读取电影和用户的特征
    usr_feats = pickle.load(open(usr_feat_dir, 'rb'))
    mov_feats = pickle.load(open(mov_feat_dir, 'rb'))
    usr_feat = usr_feats[str(usr_id)]

    cos_sims = []

    # with dygraph.guard():
    paddle.disable_static()
    # 索引电影特征，计算和输入用户ID的特征的相似度
    for idx, key in enumerate(mov_feats.keys()):
        mov_feat = mov_feats[key]
        usr_feat = paddle.to_tensor(usr_feat)
        mov_feat = paddle.to_tensor(mov_feat)
        # 计算余弦相似度
        sim = paddle.nn.functional.common.cosine_similarity(usr_feat, mov_feat)

        cos_sims.append(sim.numpy()[0])
    # 对相似度排序
    index = np.argsort(cos_sims)[-top_k:]

    mov_info = {}
    # 读取电影文件里的数据，根据电影ID索引到电影信息
    with open(mov_info_path, 'r', encoding="ISO-8859-1") as f:
        data = f.readlines()
        for item in data:
            item = item.strip().split("::")
            mov_info[str(item[0])] = item

    print("当前的用户是：")
    print("usr_id:", usr_id)
    print("推荐可能喜欢的电影是：")
    res = []

    # 加入随机选择因素，确保每次推荐的都不一样
    while len(res) < pick_num:
        val = np.random.choice(len(index), 1)[0]
        idx = index[val]
        mov_id = list(mov_feats.keys())[idx]
        if mov_id not in res:
            res.append(mov_id)

    for id in res:
        print("mov_id:", id, mov_info[str(id)])
    #     folder_path = "./ml-1m/posters/"
    #     image_path = os.path.join(folder_path, id+'.jpg')
    #     print(image_path)
    #     img = Image.open(image_path)
    #     img.show()
    recommended_movies = []
    for id in res:
        image_path = os.path.join('./ml-1m/posters/', id + '.jpg')
        recommended_movies.append({
            'mov_id': id,
            'mov_info': mov_info[str(id)],
            'poster_path': image_path   # 添加海报图片路径
        })

    # 将推荐结果转换成 JSON 格式
    result = {
        'usr_id': usr_id,
        'recommended_movies': recommended_movies
    }
    json_result = json.dumps(result)

    return json_result

movie_data_path = "./ml-1m/movies.dat"
top_k, pick_num = 10, 6
usr_id = 3
recommend_mov_for_usr(usr_id, top_k, pick_num, './usr_feat.pkl', './mov_feat.pkl', movie_data_path)


usr_a = 3
topk=6


# ##########################################
# ## 获得ID为usr_a的用户评分过的电影及对应评分 ##
# ##########################################
rating_path = "./ml-1m/ratings.dat"
# 打开文件，ratings_data
with open(rating_path, 'r') as f:
    ratings_data = f.readlines()

usr_rating_info = {}
for item in ratings_data:
    item = item.strip().split("::")
    # 处理每行数据，分别得到用户ID，电影ID，和评分
    usr_id, movie_id, score = item[0], item[1], item[2]
    if usr_id == str(usr_a):
        usr_rating_info[movie_id] = float(score)

# 获得评分过的电影ID
movie_ids = list(usr_rating_info.keys())
print("ID为 {} 的用户，评分过的电影数量是: ".format(usr_a), len(movie_ids))

#####################################
## 选出ID为usr_a评分最高的前topk个电影 ##

ratings_topk = sorted(usr_rating_info.items(), key=lambda item: item[1])[-topk:]

movie_info_path = "./ml-1m/movies.dat"
# 打开文件，编码方式选择ISO-8859-1，读取所有数据到data中
with open(movie_info_path, 'r', encoding="ISO-8859-1") as f:
    data = f.readlines()

movie_info = {}
for item in data:
    item = item.strip().split("::")
    # 获得电影的ID信息
    v_id = item[0]

    movie_info[v_id] = item

for k, score in ratings_topk:
    print("电影ID: {}，评分是: {}, 电影信息: {}".format(k, score, movie_info[k]))

