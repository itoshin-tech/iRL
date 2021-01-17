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

    def reset(self):
        """
        変数を初期化
        """

    def step(self, action: int):  # pylint:disable=no-self-use
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

    def render(self):  # pylint:disable=no-self-use
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
        

    def build_model(self):
        """
        モデル構築(Tensorflow使用時)
        """

    def select_action(self, observation):  # pylint:disable=no-self-use
        """
        observation に基づいてaction を出力

        Returns
        -------
        action: int
        """
        action = 0
        return action

    def get_Q(self, observation):  # pylint:disable=no-self-use
        """
        observationに対するQ値を出力
        """
        Q = [0, 0]  # action の種類数のQ値
        return Q

    def learn(self, observation, action, reward, next_observation, done):
        """
        学習
        """

    def reset(self):
        """
        内部状態をリセット(lstmやgruで使用)
        """

    def save_state(self):
        """
        内部状態をメモリーに保存(lstmやgruで使用)
        """

    def load_state(self):
        """
        内部状態の復元(lstmやgruで使用)
        """

    def save_weights(self, filepath):
        """
        Qtableやweightパラメータの保存
        """

    def load_weights(self, filepath):
        """
        Qtableやweightパラメータの読み込み
        """
    