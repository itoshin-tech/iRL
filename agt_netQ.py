"""
agt_netQ.py
ニューラルネットを使ったQ学習アルゴリズム
"""
import numpy as np
import tensorflow as tf
import core


class Agt(core.coreAgt):
    """
    Q値をニューラルネットで近似するエージェント
    """
    def __init__(
        self,
        n_action=2,  # 行動の種類の数（ネットワークの出力数）
        input_size=(7, ),　# 入力サイズ
        n_dense=64,  # 中間層のユニット数
        epsilon=0.1,
        gamma=0.9,
        filepath=None,
    ):
        """
        Parameters
        ----------
        n_action: int
            行動の種類の数
        input_size: tuple of int 例 (7,), (5, 5)
            入力の次元
        epsilon: float (0から1まで)
            Q学習のε、乱雑度
        gammma: float (0から1まで)
            Q学習のγ、割引率
        n_dense: int
            中間層1のニューロン数
        n_dense2: int or None
            中間層2のニューロン数
            None の場合は中間層2はなし
        filepath: str
            セーブ用のファイル名
        """

        # パラメータ
        self.n_action = n_action
        self.input_size = input_size
        self.n_dense = n_dense
        self.epsilon = epsilon
        self.gamma = gamma
        self.filepath = filepath

        # 変数
        self.time = 0
        self.model = None

        super().__init__()

    def build_model(self):
        """
        指定したパラメータでモデルを構築する
        """
        # (A) 入力層
        inputs = tf.keras.Input(shape=(self.input_size))

        # (B) 入力層を1次元に展開する
        x = tf.keras.layers.Flatten()(inputs)

        # (C) 中間層を定義。活性化関数はreluに指定
        x = tf.keras.layers.Dense(self.n_dense, activation='relu')(x) # (C)

        # (D) 出力層を定義。活性化関数はlinear（線形）に指定
        outputs = tf.keras.layers.Dense(self.n_action, activation='linear')(x) # (D)

        # (E) ニューラルネットワークモデルのインスタンス生成
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs) # (E)

        # (F) 勾配法のパラメータの定義
        self.model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mse']
        )

    def select_action(self, observation):
        """
        観測値observationに対して、行動actionを選ぶ
        """

        Q = self.get_Q(observation)

        if self.epsilon < np.random.rand():
            action = np.argmax(Q)
        else:
            action = np.random.randint(0, self.n_action)
        return action

    def get_Q(self, observation):
        """
        観測値observationに対するQ値を出力する
        """
        obs = self._trans_code(observation)
        Q = self.model.predict(obs.reshape((1,) + self.input_size))[0]
        return Q

    def _trans_code(self, observation):
        """
        observationを変換する場合にはこの関数内を記述
        """
        return observation

    def learn(self, observation, action, reward, next_observation, done):
        """
        学習
        """
        obs = self._trans_code(observation)
        next_obs = self._trans_code(next_observation)
        Q = self.model.predict(obs.reshape((1,) + self.input_size))

        if done is False:
            next_Q = self.model.predict(next_obs.reshape((1,) + self.input_size))[0]
            target = reward + self.gamma * max(next_Q)
        else:
            target = reward

        Q[0][action] = target
        self.model.fit(obs.reshape((1,) + self.input_size), Q, verbose=0, epochs=1)

    def save_weights(self, filepath=None):
        """
        学習をファイルに保存する
        """
        if filepath is None:
            filepath = self.filepath
        self.model.save(filepath + '.h5', overwrite=True)

    def load_weights(self, filepath=None):
        """
        学習をファイルから読み込む
        """
        if filepath is None:
            filepath = self.filepath
        self.model = tf.keras.models.load_model(filepath + '.h5')


if __name__ == '__main__':
    agt = Agt()
    agt.build_model()
    agt.model.summary()
    agt.save_weights('agt_data/test.h5')
    agt.load_weights('agt_data/test.h5')
    