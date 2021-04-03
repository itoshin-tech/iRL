"""
main_corridor.py
実行ファイル
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 自作モジュール
from env_corridor import CorridorEnv  # (1)
from agt_tableQ import TableQAgt      # (2)
from trainer import Trainer           # (3)
from env_corridor import TaskType

SAVE_DIR = 'agt_data'
ENV_NAME = 'env_corridor'

argvs = sys.argv
if len(argvs) < 3:
    MSG = '\n' + \
        '---- 使い方 ---------------------------------------\n' + \
        '2つのパラメータを指定して実行します\n\n' + \
        '> python main_corridor.py [task_type] [process_type]\n\n' + \
        '[task_type]\t: %s\n' % ', '.join([t.name for t in TaskType]) + \
        '[process_type]\t:learn/L, more/M, graph/G, anime/A\n' + \
        '例 > python main_corridor.py short_road L\n' + \
        '---------------------------------------------------'
    print(MSG)
    sys.exit()

# 入力パラメータの確認 //////////
task_type = TaskType.Enum_of(argvs[1])
if task_type is None:
    MSG = '\n' + \
        '[task type] が異なります。以下から選んで指定してください。\n' + \
        '%s\n' % ', '.join([t.name for t in TaskType])
    print(MSG)
    sys.exit()

# process_type //////////
process_type = argvs[2]
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
    print('[q] 終了')    
else:
    print('process type が間違っています。')
    sys.exit()

# 保存用フォルダの確認・作成 //////////
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

# 環境 インスタンス生成 //////////
# 学習用環境
env = CorridorEnv()             # (4)
env.set_task_type(task_type)    # (5)
obs = env.reset()               # (6)

# 評価用環境
eval_env = CorridorEnv()
eval_env.set_task_type(task_type)

# TableQAgt パラメータ //////////
agt_prm = {                     # (7)
    'input_size': obs.shape,
    'n_action': env.n_action,
    'init_val_Q': 0,    # Qの初期値
    'alpha': 0.1,       # 学習率
    'filepath': SAVE_DIR + '/sim_' + \
                ENV_NAME + '_' + \
                task_type.name
}

# Trainer シミュレーション共通パラメータ //////////
sim_prm = {                     # (8)
    'n_episode': -1,        # エピソード数（-1は終了条件にしない）
    'is_eval': True,        # 評価を行うか
    'is_learn': True,       # 学習を行うか
    'is_animation': False,  # アニメーションの表示をするか
    'eval_n_step': -1,      # 評価のステップ数（-1は終了条件にしない）
    'eval_n_episode': 100,  # 評価のエピソード数
    'eval_epsilon': 0.0,    # 評価時の乱雑度
}

# Trainer アニメーション共通パラメータ //////////
sim_anime_prm = {
    'n_step': -1,           # ステップ数（-1は終了条件にしない）
    'n_episode': 100,       # エピソード数
    'is_eval': False,       # 評価を行うか
    'is_learn': False,      # 学習を行うか
    'is_animation': True,   # アニメーションの表示をするか
    'anime_delay': 0.2,     # アニメーションのフレーム間の秒数
}
ANIME_EPSILON = 0.0         # アニメーション時の乱雑度

# グラフ表示共通パラメータ //////////
graph_prm = {}

# task_type 別のパラメータ //////////
if task_type == TaskType.short_road: #  (9)
    sim_prm['n_step'] = 5000        # ステップ数
    sim_prm['eval_interval'] = 100  # 評価を何ステップ毎にするか
    agt_prm['epsilon'] = 0.4        # 乱雑度
    agt_prm['gamma'] = 1.0          # 割引率
    graph_prm['target_reward'] = 2.5 # 報酬の目標値
    graph_prm['target_step'] = 3.5  # ステップ数の目標値
    obss = [                        # Q値チェック用の観測
        [
            [1, 0, 2, 0],
            [0, 1, 2, 0],
            [0, 0, 1, 0],
            [0, 0, 2, 1],
        ],
        [
            [1, 0, 0, 2],
            [0, 1, 0, 2],
            [0, 0, 1, 2],
            [0, 0, 0, 1],
        ],
    ]
    sim_prm['obss'] = obss

elif task_type == TaskType.long_road:  # (10)
    sim_prm['n_step'] = 10000
    sim_prm['eval_interval'] = 500
    agt_prm['epsilon'] = 0.4
    agt_prm['gamma'] = 1.0
    graph_prm['target_reward'] = 2.5
    graph_prm['target_step'] = 3.5
    obss = [
        [
            [1, 0, 0, 2, 0],
            [0, 1, 0, 2, 0],
            [0, 0, 1, 2, 0],
            [0, 0, 0, 1, 0],
        ],
        [
            [1, 0, 0, 0, 2],
            [0, 1, 0, 0, 2],
            [0, 0, 1, 0, 2],
            [0, 0, 0, 1, 2],
            [0, 0, 0, 0, 1],
        ],
    ]
    sim_prm['obss'] = obss

# メイン //////////
if (IS_LOAD_DATA is True) or \
    (IS_LEARN is True) or \
    (sim_prm['is_animation'] is True):  # (11)

    # Agtのインスタンス生成、ディクショナリ型でパラメータを渡すときには **　を付ける
    agt = TableQAgt(**agt_prm)  # (12)

    # Trainerのインスタンス生成
    trainer = Trainer(agt, env, eval_env)  # (13)

    if IS_LOAD_DATA is True:
        # エージェントのデータロード
        try:
            agt.load_weights()
            trainer.load_history(agt.filepath)
        except Exception as e:
            print(e)
            print('エージェントのパラメータがロードできません')
            sys.exit()

    if IS_LEARN is True:  # (14)
        # 学習
        # シミュレーション実行、ディクショナリ型は**でパラメータ指定        
        trainer.simulate(**sim_prm)  # (15)

        # エージェントの学習結果保存        
        agt.save_weights()

        # 学習履歴保存
        trainer.save_history(agt.filepath)

    if IS_SHOW_ANIME is True:
        # アニメーション
        agt.epsilon = ANIME_EPSILON
        trainer.simulate(**sim_anime_prm)

if IS_SHOW_GRAPH is True:
    # グラフ表示の関数定義
    def show_graph(pathname, target_reward=None, target_step=None):
        """
        学習曲線の表示

        Parameters
        ----------
        target_reward: float or None
            rewardの目標値に線を引く
        target_step: float or None
            stepの目標値に線を引く
        """
        hist = np.load(pathname + '.npz')
        eval_rwd = hist['eval_rwds'].tolist()
        eval_step = hist['eval_steps'].tolist()
        eval_x = hist['eval_x'].tolist()

        plt.figure(figsize=(8,4))
        plt.subplots_adjust(hspace=0.6)

        # reward / episode
        plt.subplot(211)
        plt.plot(eval_x, eval_rwd, 'b.-')
        if target_reward is not None:
            plt.plot(
                [eval_x[0], eval_x[-1]],
                [target_reward, target_reward],
                'r:')

        plt.title('rewards / episode')
        plt.grid(True)

        # steps / episode
        plt.subplot(212)
        plt.plot(eval_x, eval_step, 'b.-')
        if target_step is not None:
            plt.plot(
                [eval_x[0], eval_x[-1]],
                [target_step, target_step],
                'r:')
        plt.title('steps / episode')
        plt.xlabel('steps')
        plt.grid(True)

        plt.show()    
    
    # 学習履歴表示
    show_graph(agt_prm['filepath'], **graph_prm)



