# from flask import Flask, request, render_template
# import pickle
# import paddle
# import numpy as np
# import json
# import os
# from PIL import Image
# import base64
#
# app = Flask(__name__)
#
#
# def recommend_mov_for_usr(usr_id, top_k, pick_num, usr_feat_dir, mov_feat_dir, mov_info_path):
#     assert pick_num <= top_k, "pick_num should be less than or equal to top_k"
#
#     try:
#         with open(usr_feat_dir, 'rb') as f:
#             usr_feats = pickle.load(f)
#         with open(mov_feat_dir, 'rb') as f:
#             mov_feats = pickle.load(f)
#     except FileNotFoundError as e:
#         print(f"Error: {e}")
#         return None
#
#     usr_feat = usr_feats.get(str(usr_id))
#     if usr_feat is None:
#         print(f"User ID {usr_id} not found in user features.")
#         return None
#
#     cos_sims = []
#
#     paddle.disable_static()
#     for key in mov_feats.keys():
#         mov_feat = mov_feats[key]
#         usr_feat_tensor = paddle.to_tensor(usr_feat)
#         mov_feat_tensor = paddle.to_tensor(mov_feat)
#         sim = paddle.nn.functional.cosine_similarity(usr_feat_tensor, mov_feat_tensor)
#         cos_sims.append(sim.numpy()[0])
#
#     index = np.argsort(cos_sims)[-top_k:]
#
#     mov_info = {}
#     try:
#         with open(mov_info_path, 'r', encoding="ISO-8859-1") as f:
#             data = f.readlines()
#             for item in data:
#                 item = item.strip().split("::")
#                 mov_info[item[0]] = item
#     except FileNotFoundError as e:
#         print(f"Error: {e}")
#         return None
#
#     res = []
#     while len(res) < pick_num:
#         val = np.random.choice(len(index), 1)[0]
#         idx = index[val]
#         mov_id = list(mov_feats.keys())[idx]
#         if mov_id not in res:
#             res.append(mov_id)
#
#     recommended_movies = []
#     for id in res:
#         recommended_movies.append({
#             'mov_id': id,
#             'mov_info': mov_info[id]
#         })
#
#     result = {
#         'usr_id': usr_id,
#         'recommended_movies': recommended_movies
#     }
#     return result
#
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
#
# @app.route('/recommend', methods=['POST'])
# def recommend():
#     usr_id = request.form['usr_id']
#     top_k = int(request.form['top_k'])  # 推荐系统返回给用户的候选电影的数量
#     pick_num = 10  # 固定为6  #  推荐电影的数量
#
#     result = recommend_mov_for_usr(usr_id, top_k, pick_num, './usr_feat.pkl', './mov_feat.pkl', './ml-1m/movies.dat')
#     if result is None:
#         return "Error in recommendation process."
#
#     recommended_movies = result['recommended_movies']
#     movie_posters = []
#     for movie in recommended_movies:
#         mov_id = movie['mov_id']
#         folder_path = "./ml-1m/posters/"
#         image_path = os.path.join(folder_path, mov_id + '.jpg')
#         if os.path.exists(image_path):
#             with open(image_path, "rb") as img_f:
#                 img_str = base64.b64encode(img_f.read()).decode('utf-8')
#                 movie_posters.append({
#                     'mov_info': movie['mov_info'],
#                     'img_str': img_str
#                 })
#
#     return render_template('recommend.html', usr_id=usr_id, movie_posters=movie_posters)
#
#
# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, render_template
import os
import base64
import pickle
import numpy as np
import paddle

app = Flask(__name__)


def recommend_mov_for_usr(usr_id, top_k, pick_num, usr_feat_dir, mov_feat_dir, mov_info_path):
    assert pick_num <= top_k, "pick_num should be less than or equal to top_k"

    try:
        with open(usr_feat_dir, 'rb') as f:
            usr_feats = pickle.load(f)
        with open(mov_feat_dir, 'rb') as f:
            mov_feats = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

    usr_feat = usr_feats.get(str(usr_id))
    if usr_feat is None:
        print(f"User ID {usr_id} not found in user features.")
        return None

    cos_sims = []

    paddle.disable_static()
    for key in mov_feats.keys():
        mov_feat = mov_feats[key]
        usr_feat_tensor = paddle.to_tensor(usr_feat)
        mov_feat_tensor = paddle.to_tensor(mov_feat)
        sim = paddle.nn.functional.cosine_similarity(usr_feat_tensor, mov_feat_tensor)
        cos_sims.append(sim.numpy()[0])

    index = np.argsort(cos_sims)[-top_k:]

    mov_info = {}
    try:
        with open(mov_info_path, 'r', encoding="ISO-8859-1") as f:
            data = f.readlines()
            for item in data:
                item = item.strip().split("::")
                mov_info[item[0]] = item
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

    res = []
    while len(res) < pick_num:
        val = np.random.choice(len(index), 1)[0]
        idx = index[val]
        mov_id = list(mov_feats.keys())[idx]
        if mov_id not in res:
            res.append(mov_id)

    recommended_movies = []
    for id in res:
        recommended_movies.append({
            'mov_id': id,
            'mov_info': mov_info[id]
        })

    result = {
        'usr_id': usr_id,
        'recommended_movies': recommended_movies
    }
    return result



def get_top_rated_movies(usr_id, topk):
    rating_path = "./ml-1m/ratings.dat"
    with open(rating_path, 'r') as f:
        ratings_data = f.readlines()

    usr_rating_info = {}
    for item in ratings_data:
        item = item.strip().split("::")
        user_id, movie_id, score = item[0], item[1], item[2]
        if user_id == str(usr_id):
            usr_rating_info[movie_id] = float(score)

    movie_ids = list(usr_rating_info.keys())
    print(f"User ID {usr_id} has rated {len(movie_ids)} movies.")

    ratings_topk = sorted(usr_rating_info.items(), key=lambda item: item[1], reverse=True)[:topk]

    movie_info_path = "./ml-1m/movies.dat"
    with open(movie_info_path, 'r', encoding="ISO-8859-1") as f:
        data = f.readlines()

    movie_info = {}
    for item in data:
        item = item.strip().split("::")
        v_id = item[0]
        movie_info[v_id] = item

    top_rated_movies = []
    for k, score in ratings_topk:
        poster_path = os.path.join("./ml-1m/posters/", f"{k}.jpg")
        if os.path.exists(poster_path):
            with open(poster_path, "rb") as img_f:
                img_str = base64.b64encode(img_f.read()).decode('utf-8')
        else:
            img_str = None
        top_rated_movies.append({
            'mov_id': k,
            'score': score,
            'mov_info': movie_info[k],
            'poster_base64': img_str
        })

    return top_rated_movies


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    usr_id = request.form['usr_id']
    top_k = int(request.form['top_k'])
    pick_num = int(request.form['pick_num'])

    result = recommend_mov_for_usr(usr_id, top_k, pick_num, './usr_feat.pkl', './mov_feat.pkl', './ml-1m/movies.dat')
    if result is None:
        return "Error in recommendation process."

    recommended_movies = result['recommended_movies']
    movie_posters = []
    for movie in recommended_movies:
        mov_id = movie['mov_id']
        folder_path = "./ml-1m/posters/"
        image_path = os.path.join(folder_path, mov_id + '.jpg')
        if os.path.exists(image_path):
            with open(image_path, "rb") as img_f:
                img_str = base64.b64encode(img_f.read()).decode('utf-8')
                movie_posters.append({
                    'mov_info': movie['mov_info'],
                    'img_str': img_str
                })
        else:
            movie_posters.append({
                'mov_info': movie['mov_info'],
                'img_str': None
            })

    top_rated_movies = get_top_rated_movies(usr_id, 5)
    return render_template('recommend.html', usr_id=usr_id, movie_posters=movie_posters,
                           top_rated_movies=top_rated_movies)


if __name__ == '__main__':
    app.run(debug=True)


