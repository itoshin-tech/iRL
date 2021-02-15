"""
agt_tableQ.py
Q-tableを使ったQ学習アルゴリズム
"""
import sys
import pickle
import numpy as np
import core


class Agt(core.coreAgt):
    """
    Q-tableを使ったQ学習エージェント
    """
    def __init__(
        self,
        n_action=2,
        input_size=(7, ),
        init_val_Q=0,
        epsilon=0.1,
        alpha=0.1,
        gamma=0.9,
        max_memory=500,
        filepath=None,
    ):
        """
        Parameters
        ----------
        n_action: int
            行動の種類の数
        input_size: tuple of int 例 (7,), (5, 5)
            入力の次元
        init_val_Q: float
            Q値の初期値
        epsilon: float (0から1まで)
            Q学習のε、乱雑度
        gammma: float (0から1まで)
            Q学習のγ、割引率
        alpha: float (0から1まで)
            Q学習のα、学習率
        max_memory: int
            記憶する最大の観測数
        filepath: str
            セーブ用のファイル名
        """

        # パラメータ
        self.n_action = n_action
        self.input_size=input_size
        self.init_val_Q = init_val_Q
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.max_memory=max_memory
        self.filepath = filepath

        # 変数
        self.time = 0
        self.Q = {}
        self.len_Q = 0

        super().__init__()

    def select_action(self, obs):
        """
        観測obsに対して、行動actionを選ぶ
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
        観測observationに対するQ値を出力する
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
                print('新規の観測数が上限 %d に達しました。' % self.max_memory)
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
    