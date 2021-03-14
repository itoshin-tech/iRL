"""
agt_tableQ.py
Q-tableを使ったQ学習アルゴリズム
"""
import sys
import pickle
import numpy as np


class TableQAgt():
    """
    Q-tableを使ったQ学習エージェント
    """
    def __init__(
        self,
        n_action=2,         # int: 行動の種類(数値はデフォルト値、以下同様)
        input_size=(7, ),   # tuple of int: 入力の次元、 例(7,), (5, 5)
        init_val_Q=0,       # float: Q値の初期値
        epsilon=0.1,        # float: 乱雑度
        alpha=0.1,          # float: 学習率
        gamma=0.9,          # float: 割引率
        max_memory=500,     # int: 記憶する最大の観測数
        filepath=None,      # str: セーブ用のファイル名
    ):
        # (A) クラス内で使用する変数（アトリビュート）の提議
        # 実際のコードではAgtの引数で設定するようにしている
        self.n_action = n_action
        self.input_size = input_size
        self.init_val_Q = init_val_Q
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.max_memory = max_memory
        self.filepath = filepath

        # 変数
        self.time = 0
        self.Q = {}
        self.len_Q = 0

    def select_action(self, obs):
        """
        観測obsに対して、行動actionを出力する
        """
        # (A) obsを文字列に変換
        obs = str(obs)

        # (B) next_obs がself.Qのキーになかったら追加する
        self._check_and_add_observation(obs)

        if np.random.rand() < self.epsilon:
            # (C)  epsilon の確率でランダムに行動を選ぶ
            act = np.random.randint(0, self.n_action)
        else:
            # (D) 1- epsilon の確率でQの値を最大とする行動を選ぶ
            act = np.argmax(self.Q[obs])
        return act

    def get_Q(self, obs):
        """
        観測obsに対するQ値を出力する
        """
        obs = str(obs)
        if obs in self.Q:
            val = self.Q[obs]
            return np.array(val)
        return (None, ) * self.n_action

    def _check_and_add_observation(self, obs):
        if obs not in self.Q:
            self.Q[obs] = [self.init_val_Q] * self.n_action
            self.len_Q += 1
            if self.len_Q > self.max_memory:
                print('観測の登録数が上限 %d に達しました。' % self.max_memory)
                sys.exit()
            if (self.len_Q < 100 and self.len_Q % 10 == 0) or (self.len_Q % 100 == 0):
                print('used memory for Q-table --- %d' % self.len_Q)

    
    def learn(self, obs, act, rwd, next_obs, next_done):
        """
        学習する
        """
        # (A) obs, next_obs を文字列に変換
        obs = str(obs)
        next_obs = str(next_obs)

        # (B) next_obsがself.Qのキーになかったら追加する
        self._check_and_add_observation(next_obs)

        # (C) 学習のtargetを作成
        if next_done is False:
            target = rwd + self.gamma * max(self.Q[next_obs])
        else:
            target = rwd

        # (D) Qをtargetに近づける
        self.Q[obs][act] = (1-self.alpha) * self.Q[obs][act] + \
                           self.alpha * target

    def save_weights(self, filepath=None):
        """
        学習をファイルに保存する
        """
        if filepath is None:
            filepath = self.filepath
        with open(filepath, mode='wb') as f:
            pickle.dump(self.Q, f)

    def load_weights(self, filepath=None):
        """
        学習をファイルから読み込む
        """
        if filepath is None:
            filepath = self.filepath
        with open(filepath, mode='rb') as f:
            self.Q = pickle.load(f)
    