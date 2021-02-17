"""
sim_corridor.py
廊下タスクの実行ファイル
"""
import os
import sys

# 自作モジュール
import env_corridor as envnow
from env_corridor import TaskType
import trainer
import myutil

SAVE_DIR = 'agt_data'
ENV_NAME = 'env_corridor'

argvs = sys.argv
if len(argvs) < 4:
    MSG = '\n' + \
        '---- 使い方 ---------------------------------------\n' + \
        '3つのパラメータを指定して実行します\n\n' + \
        '> python sim_corridor.py [agt_type] [task_type] [process_type]\n\n' + \
        '[agt_type]\t: tableQ, netQ\n' + \
        '[task_type]\t: %s\n' % ', '.join([t.name for t in TaskType]) + \
        '[process_type]\t:learn/L, more/M, graph/G, anime/A\n' + \
        '例 > python sim_corridor.py tableQ L4c23 L\n' + \
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
    print('グラフ表示を終了するには[q]を押します。')
elif process_type in ('anime', 'A'):
    IS_LOAD_DATA = True
    IS_LEARN = False
    IS_SHOW_GRAPH = False
    IS_SHOW_ANIME = True
    print('アニメーションを途中で止めるには[q]を押します。')
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
    'n_step': 5000,
    'n_episode': -1,
    'is_eval': True,
    'IS_LEARN': True,
    'is_animation': False,
    'eval_interval': 200,
    'eval_n_step': -1,
    'eval_n_episode': 100,
    'eval_epsilon': 0.0,
}
# Trainer アニメーション共通パラメータ //////////
sim_anime_prm = {
    'n_step': -1,
    'n_episode': 100,
    'is_eval': False,
    'IS_LEARN': False,
    'is_animation': True,
    'ANIME_DELAY': 0.2,
}
ANIME_EPSILON = 0.0

# グラフ表示共通パラメータ //////////
graph_prm = {}

# task_type 別のパラメータ //////////
if task_type == TaskType.L4c23:
    sim_prm['n_step'] = 5000
    sim_prm['eval_interval'] = 100
    agt_prm['epsilon'] = 0.4
    agt_prm['gamma'] = 1.0
    graph_prm['target_reward'] = 2.5
    graph_prm['target_step'] = 3.5
    obss = [
        [
            [1, 0, 3, 0],
            [0, 1, 3, 0],
            [0, 0, 1, 0],
            [0, 0, 3, 1],
        ],
        [
            [1, 0, 0, 3],
            [0, 1, 0, 3],
            [0, 0, 1, 3],
            [0, 0, 0, 1],
        ],
    ]
    sim_prm['obss'] = obss

elif task_type == TaskType.L5c14:
    sim_prm['n_step'] = 5000
    sim_prm['eval_interval'] = 100
    agt_prm['epsilon'] = 0.4
    agt_prm['gamma'] = 1.0
    graph_prm['target_reward'] = 2.5
    graph_prm['target_step'] = 3.5
    obss = [
        [
            [1, 0, 0, 3, 0],
            [0, 1, 0, 3, 0],
            [0, 0, 1, 3, 0],
            [0, 0, 0, 1, 0],
        ],
        [
            [1, 0, 0, 0, 3],
            [0, 1, 0, 0, 3],
            [0, 0, 1, 0, 3],
            [0, 0, 0, 1, 3],
            [0, 0, 0, 0, 1],
        ],
    ]
    sim_prm['obss'] = obss

# agt_type 別のパラメータ //////////
if agt_type == 'tableQ':
    agt_prm['init_val_Q'] = 0
    agt_prm['alpha'] = 0.1

elif agt_type == 'netQ':
    agt_prm['n_dense'] = 64
    agt_prm['n_dense2'] = None  # 数値にするとその素子数で1層追加

# メイン //////////
if (IS_LOAD_DATA is True) or \
    (IS_LEARN is True) or \
    (sim_prm['is_animation'] is True):

    # エージェントをインポートしてインスタンス作成
    if agt_type == 'tableQ':
        from agt_tableQ import Agt
    elif agt_type == 'netQ':
        from agt_netQ import Agt
    else:
        raise ValueError('agt_type が間違っています')

    agt = Agt(**agt_prm)
    agt.build_model()

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
