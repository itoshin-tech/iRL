"""
sim_corridor.py
廊下タスクの実行ファイル
"""
import os
import sys
import copy

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
    ANIME_N_EPISODE = 0
elif process_type in ('more', 'M'):
    IS_LOAD_DATA = True
    IS_LEARN = True
    IS_SHOW_GRAPH = True
    IS_SHOW_ANIME = False
    ANIME_N_EPISODE = 0
elif process_type in ('graph', 'G'):
    IS_LOAD_DATA = False
    IS_LEARN = False
    IS_SHOW_GRAPH = True
    IS_SHOW_ANIME = False
    ANIME_N_EPISODE = 0
    print('グラフ表示を終了するには[q]を押します。')
elif process_type in ('anime', 'A'):
    IS_LOAD_DATA = True
    IS_LEARN = False
    IS_SHOW_GRAPH = False
    IS_SHOW_ANIME = True
    ANIME_N_EPISODE = 100
    print('アニメーションを途中で止めるには[q]を押します。')
else:
    print('process type が間違っています。')
    sys.exit()

# task_type paramter //////////
if task_type == TaskType.L4c23:
    N_STEP = 5000
    EVAL_INTERVAL =100
    TARGET_REWARD = 2.5
    TARGET_STEP = 3.5
    AGT_EPSILON = 0.4
    AGT_ANIME_EPSILON = 0.0

elif task_type == TaskType.L5c14:
    N_STEP = 5000
    EVAL_INTERVAL =100
    TARGET_REWARD = 2.5
    TARGET_STEP = 3.5
    AGT_EPSILON = 0.4
    AGT_ANIME_EPSILON = 0.0

    """
elif task_type == TaskType.mytask:  # mytaskのパラメータ追加
    N_STEP = 5000
    EVAL_INTERVAL =200
    TARGET_STEP = None
    TARGET_REWARD = 1
    AGT_EPSILON = 0.4
    AGT_ANIME_EPSILON = 0.0
    """
else:
    N_STEP = 5000
    EVAL_INTERVAL =200
    TARGET_STEP = None
    TARGET_REWARD = 1
    AGT_EPSILON = 0.4
    AGT_ANIME_EPSILON = 0.0
    print('シミュレーションにデフォルトパラメータを設定しました。')

# 学習用環境
env = envnow.Env()
env.set_task_type(task_type)
obs = env.reset()

# 評価用環境
eval_env = envnow.Env()
eval_env.set_task_type(task_type)

# agent prameter  //////////
# agt common
agt_prm = {
    'gamma': 1.0,
    'epsilon': AGT_EPSILON,
    'input_size': obs.shape,
    'n_action': env.n_action,
    'filepath': SAVE_DIR + '/sim_' + \
                ENV_NAME + '_' + \
                agt_type + '_' + \
                task_type.name
}

if agt_type == 'tableQ':
    agt_prm['init_val_Q'] = 0
    agt_prm['alpha'] = 0.1

elif agt_type == 'netQ':
    agt_prm['n_dense'] = 64
    agt_prm['n_dense2'] = None  # 数値にすると1層追加

else:
    raise ValueError('agt_type が間違っています')

# simulation pramter //////////
sim_prm = {
    'N_STEP': N_STEP,
    'N_EPISODE': -1,
    'EVAL_INTERVAL': EVAL_INTERVAL,
    'IS_LEARN': IS_LEARN,
    'IS_ANIMATION': False,
    'SHOW_DELAY': 0.5,
    'eval_N_STEP': -1,
    'eval_N_EPISODE': 100,
    'eval_EPSILON': 0.0,
}

# animation pramter //////////
sim_anime_prm = {
    'N_STEP': -1,
    'N_EPISODE': ANIME_N_EPISODE,
    'IS_LEARN': False,
    'IS_ANIMATION': True,
    'SHOW_DELAY': 0.2,
    'eval_N_STEP': -1,
    'eval_N_EPISODE': -1,
}

# trainer paramter ///////////////
obss = None
if task_type is TaskType.L4c23:
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
elif task_type is TaskType.L5c14:
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

trn_prm = {
    'obss': obss,
    'show_header': '%s %s ' % (agt_type, task_type.name),
}

# メイン //////////
if (IS_LOAD_DATA is True) or \
    (IS_LEARN is True) or \
    (sim_prm['IS_ANIMATION'] is True):

    # エージェントをインポートしてインスタンス作成
    if agt_type == 'tableQ':
        from agt_tableQ import Agt  # pylint:disable=unused-import
    elif agt_type == 'netQ':
        from agt_netQ import Agt
    else:
        raise ValueError('agt_type が間違っています')

    agt = Agt(**agt_prm)
    agt.build_model()

    # trainer インスタンス作成
    trn = trainer.Trainer(agt, env, eval_env, **trn_prm)

    if IS_LOAD_DATA is True:
        # エージェントのデータロード
        try:
            agt.load_weights()
            trn.load_history(agt.filepath)
        except Exception:
            print('エージェントのパラメータがロードできません')
            sys.exit()

    # 学習
    if IS_LEARN is True:
        trn.simulate(**sim_prm)
        agt.save_weights()
        trn.save_history(agt.filepath)

    # アニメーション
    if IS_SHOW_ANIME is True:
        agt.epsilon = AGT_ANIME_EPSILON
        trn.simulate(**sim_anime_prm)

if IS_SHOW_GRAPH is True:
    # グラフ表示
    myutil.show_graph(
        agt_prm['filepath'],
        target_reward=TARGET_REWARD,
        target_step=TARGET_STEP,
    )

