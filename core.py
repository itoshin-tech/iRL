"""
core.py
EnvとAgtの抽象クラス
"""
import numpy as np

class coreEnv():
    """
    Envの抽象クラス
    """
    def __init__(self):
        """
        インスタンス生成時の処理
        """
        self.obs_in_render = True
        self.sight_in_render = True

    def reset(self):
        """
        変数を初期化
        """

    def step(self, action: int):
        """
        action に従って、observationを更新

        Returns
        -------
        observation: np.ndarray
        reward: int
        done: bool
        """
        observation = np.ndarray([0, 0])
        reward = 0
        done = False
        return observation, reward, done

    def render(self):
        """
        環境の状態に対応したimg を作成

        Returns
        -------
        img: np.ndarray((h, w, 3), type=np.uint8)
        """
        img = np.ndarray((100, 200, 3), type=np.uint8)
        return img


class coreAgt():
    """
    Agtの抽象クラス
    """
    def __init__(self):
        """
        インスタンス生成時の処理
        """
        
    def select_action(self, observation):
        """
        observation に基づいてaction を出力

        Returns
        -------
        action: int
        """
        action = 0
        return action

    def get_Q(self, observation):
        """
        observationに対するQ値を出力
        """
        Q = [0, 0]  # action の種類数のQ値
        return Q

    def learn(self, observation, action, reward, next_observation, done):
        """
        学習
        """

    def save_weights(self, filepath):
        """
        Qtableやweightパラメータの保存
        """

    def load_weights(self, filepath):
        """
        Qtableやweightパラメータの読み込み
        """
    