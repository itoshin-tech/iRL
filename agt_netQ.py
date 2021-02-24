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
            n_action=2,         # int: 行動の種類の数（ネットワークの出力数）
            input_size=(7, ),   # tuple of int: 入力サイズ
            n_dense=64,         # int: 中間層のニューロン数
            epsilon=0.1,        # float: 乱雑度
            gamma=0.9,          # float: 割引率
            filepath=None,      # str: 保存ファイル名
        ):
        """
        初期処理
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
        self.model = None   # networkモデルのインスタンス用

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
            metrics=['mse'],
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

    def get_Q(self, obs):
        """
        観測値observationに対するQ値を出力する
        """
        Q = self.model.predict(obs.reshape((1,) + self.input_size))[0]
        return Q

    def learn(self, obs, act, rwd, next_obs, done):
        """
        学習
        """
        Q = self.model.predict(obs.reshape((1,) + self.input_size))
        target = Q.copy()

        if done is False:
            next_Q = self.model.predict(next_obs.reshape((1,) + self.input_size))[0]
            target_act = rwd + self.gamma * max(next_Q)
        else:
            target_act = rwd

        target[0][act] = target_act
        self.model.fit(obs.reshape((1,) + self.input_size), target, verbose=0, epochs=1)

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
    