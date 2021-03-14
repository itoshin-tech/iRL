"""
sim_field.py
クリスタルタスクの実行ファイル
"""
import os
import sys

# 自作モジュール
import env_field as envnow
from env_field import TaskType
import trainer
import myutil

SAVE_DIR = 'agt_data'
ENV_NAME = 'env_field'

argvs = sys.argv
if len(argvs) < 4:
    MSG = '\n' + \
        '---- 使い方 ---------------------------------------\n' + \
        '3つのパラメータを指定して実行します\n\n' + \
        '> python sim_field.py [agt_type] [task_type] [process_type]\n\n' + \
        '[agt_type]\t: tableQ, netQ\n' + \
        '[task_type]\t: %s\n' % ', '.join([t.name for t in TaskType]) + \
        '[process_type]\t:learn/L, more/M, graph/G, anime/A\n' + \
        '例 > python sim_field.py tableQ no_wall L\n' + \
        '---------------------------------------------------'
    print(MSG)
    sys.exit()

# 入力パラメータの確認 //////////
agt_type = argvs[1]
task_type = TaskType.Enum_of(argvs[2])
if task_type is None:
    MSG = '\n' + \
        '[task type] が異なります。以下から選んで指定してください。\n' + \
        '%s\n' % ', '.join([t.name for t in TaskType])
    print(MSG)
    sys.exit()
process_type = argvs[3]

# 保存用フォルダの確認・作成 //////////
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

# process_type //////////
if process_type in ('learn', 'L'):
    IS_LOAD_DATA = False
    IS_LEARN = True
    IS_SHOW_GRAPH = True
    IS_SHOW_ANIME = False
elif process_type in ('more', 'M'):
    IS_LOAD_DATA = True
    IS_LEARN = True
    IS_SHOW_GRAPH = True
    IS_SHOW_ANIME = False
elif process_type in ('graph', 'G'):
    IS_LOAD_DATA = False
    IS_LEARN = False
    IS_SHOW_GRAPH = True
    IS_SHOW_ANIME = False
    print('[q] 終了')
elif process_type in ('anime', 'A'):
    IS_LOAD_DATA = True
    IS_LEARN = False
    IS_SHOW_GRAPH = False
    IS_SHOW_ANIME = True
    print('[o] 観測の表示のON/OFF')
    print('[v] 視野の表示のON/OFF')
    print('[q] 終了')

else:
    print('process type が間違っています。')
    sys.exit()

# Envインスタンス生成 //////////
# 学習用環境
env = envnow.Env()
env.set_task_type(task_type)
obs = env.reset()

# 評価用環境
eval_env = envnow.Env()
eval_env.set_task_type(task_type)

# Agt 共通パラメータ //////////
agt_prm = {
    'input_size': obs.shape,
    'n_action': env.n_action,
    'filepath': SAVE_DIR + '/sim_' + \
                ENV_NAME + '_' + \
                agt_type + '_' + \
                task_type.name
}

# Trainer シミュレーション共通パラメータ //////////
sim_prm = {
    'n_episode': -1,
    'is_eval': True,
    'is_learn': True,
    'is_animation': False,
    'eval_n_step': -1,
    'eval_n_episode': 100,
    'eval_epsilon': 0.0,
}

# Trainer アニメーション共通パラメータ //////////
sim_anime_prm = {
    'n_step': -1,
    'n_episode': 100,
    'is_eval': False,
    'is_learn': False,
    'is_animation': True,
    'anime_delay': 0.2,
}
ANIME_EPSILON = 0.0

# グラフ表示共通パラメータ //////////
graph_prm = {}

# task_type 別のパラメータ //////////
if task_type == TaskType.fixed_wall:
    sim_prm['n_step'] = 5000
    sim_prm['eval_interval'] = 200
    agt_prm['epsilon'] = 0.4
    agt_prm['gamma'] = 0.9
    graph_prm['target_reward'] = 1.0
    graph_prm['target_step'] = 12.0

elif task_type == TaskType.no_wall:
    sim_prm['n_step'] = 5000
    sim_prm['eval_interval'] = 200
    agt_prm['epsilon'] = 0.2
    agt_prm['gamma'] = 0.9
    graph_prm['target_reward']= 0.75
    graph_prm['target_step'] = 4.0

elif task_type == TaskType.random_wall:
    sim_prm['n_step'] = 5000
    sim_prm['eval_interval'] = 1000
    agt_prm['epsilon'] = 0.4
    agt_prm['gamma'] = 0.9
    graph_prm['target_reward']= 1.4
    graph_prm['target_step'] = 22.0

# agt_type 別のパラメータ //////////
if agt_type == 'tableQ':
    agt_prm['init_val_Q'] = 0
    agt_prm['alpha'] = 0.1

elif agt_type == 'netQ':
    agt_prm['n_dense'] = 64

# メイン //////////
if (IS_LOAD_DATA is True) or \
    (IS_LEARN is True) or \
    (sim_prm['is_animation'] is True):

    # エージェントをインポートしてインスタンス作成
    if agt_type == 'tableQ':
        from agt_tableQ import TableQAgt as Agt
    elif agt_type == 'netQ':
        from agt_netQ import NetQAgt as Agt
    else:
        raise ValueError('agt_type が間違っています')

    agt = Agt(**agt_prm)

    # trainer インスタンス作成
    trn = trainer.Trainer(agt, env, eval_env)

    if IS_LOAD_DATA is True:
        # エージェントのデータロード
        try:
            agt.load_weights()
            trn.load_history(agt.filepath)
        except Exception as e:
            print(e)
            print('エージェントのパラメータがロードできません')
            sys.exit()

    if IS_LEARN is True:
        # 学習
        trn.simulate(**sim_prm)
        agt.save_weights()
        trn.save_history(agt.filepath)

    if IS_SHOW_ANIME is True:
        # アニメーション
        agt.epsilon = ANIME_EPSILON
        trn.simulate(**sim_anime_prm)

if IS_SHOW_GRAPH is True:
    # グラフ表示
    myutil.show_graph(agt_prm['filepath'], **graph_prm)
