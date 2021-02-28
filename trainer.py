"""
trainer.py
環境とエージェントを組み合わせて動かす
"""
import time
import cv2
import numpy as np


class Trainer:
    """
    環境とエージェントをシミュレーションする
    """
    def __init__(
        self,
        agt=None,       # Agt class: エージェント
        env=None,       # Env class: 環境
        eval_env=None,  # Env class: 評価用環境
        ):
        """
        初期処理
        """
        # (A) Agt, Envのインスタンスをクラス内の変数（アトリビュート）として保持        # 
        self.agt = agt
        self.env = env
        self.eval_env = eval_env

        # 変数
        self.hist_eval_rwds_in_episode = []
        self.hist_eval_steps_in_episode = []
        self.hist_eval_x = []
        self.hist_start_x = 0

        # 学習履歴データ保存クラスのインスタンス生成
        self.eval_simhist = SimHistory()

    def simulate(self,
        n_step=1000,        # int: ステップ数(-1は終了条件にしない)
        n_episode=-1,       # int: エピソード数(-1は終了条件にしない)
        is_eval=True,       # bool: 評価を行うか
        is_learn=True,      # bool: 学習を行うか
        is_animation=False, # bool: アニメーションの表示をするか
        eval_interval=100,  # int: 評価を何ステップ毎にするか
        eval_n_step=-1,     # int: 評価のステップ数(-1は終了条件にしない)
        eval_n_episode=-1,  # int: 評価のエピソード数(-1は終了条件にしない)
        eval_epsilon=0.0,   # float: 乱雑度
        anime_delay=0.5,    # float: アニメーションのフレーム間の秒数
        obss=None,          # list: Q値チェック用の観測
        show_header=''      # str: コンソール表示のヘッダー情報
        ):
        """
        環境とエージェントをシミュレーションする
        """
        self.obss = obss
        self.show_header = show_header

        # 開始時間の記録
        stime = time.time()

        # 環境クラスと変数を初期化
        obs = self.env.reset()
        timestep = 0
        episode = 0
        done = False
        self.on_simulation = True

        # シミュレーションのループ開始
        while self.on_simulation:
            if done is False:
                # agtが行動を選ぶ
                act = self.agt.select_action(obs)
                next_obs, rwd, next_done = self.env.step(act)

                # agtが学習する
                if is_learn is True:
                    self.agt.learn(obs, act, rwd, next_obs, next_done)

            else:
                # 最終状態になったら、envを初期化する
                next_obs = self.env.reset()
                rwd = None
                next_done = False

            # next_obs, next_done を次の学習のために保持
            obs = next_obs
            done = next_done

            # アニメーション描画
            if is_animation:
                img = self.env.render()
                cv2.imshow('trainer', img)
                key = cv2.waitKey(int(anime_delay * 1000))
                if key == ord('q'):
                    self.off_simulation()
                    break
                if key == ord('o'):
                    self.env.obs_in_render = bool(1 - self.env.obs_in_render)
                if key == ord('v'):
                    self.env.sight_in_render = bool(1 - self.env.sight_in_render)

            # 一定のステップ数で記録と評価と表示を行う
            if is_eval is True:
                if (timestep % eval_interval == 0) and (timestep > 0):
                    # 記録用変数の初期化
                    eval_rwds_in_episode = None
                    eval_steps_in_episode = None
                    if (eval_n_step > 0) or (eval_n_episode > 0):
                        # 評価を行う
                        out = self.evaluation(
                            eval_n_step=eval_n_step,
                            eval_n_episode=eval_n_episode,
                            eval_epsilon=eval_epsilon,
                        )
                        # 記録
                        eval_rwds_in_episode = out.mean_rwds[0]
                        eval_steps_in_episode = out.mean_STEPs_in_episode[0]
                        self.hist_eval_rwds_in_episode.append(eval_rwds_in_episode)
                        self.hist_eval_steps_in_episode.append(eval_steps_in_episode)
                        self.hist_eval_x.append(self.hist_start_x + timestep)

                    # 途中経過表示
                    ptime = time.time() - stime
                    if eval_rwds_in_episode is not None:
                        print('%s %d steps, %d sec --- rwds/epsd % .2f, steps/epsd % .2f' % (
                                self.show_header,
                                timestep, ptime,
                                eval_rwds_in_episode,
                                eval_steps_in_episode,
                                )
                            )
                    else:
                        print('%s %d --- %d sec' % (
                                self.show_header,
                                timestep, ptime,
                                )
                            )

            # 指定したstepかepisode数に達したら終了
            if n_step > 0:
                if timestep >= n_step:
                    break
            if n_episode > 0:
                if episode > n_episode:
                    break

            timestep += 1
            episode += done

        # Q値を表示
        self._show_Q()
 
        return

    def off_simulation(self):
        """
        シミュレーションを終了する
        """
        self.on_simulation = False

    def evaluation(self,
            eval_n_step,
            eval_n_episode,
            eval_epsilon,
        ):
        """
        学習を止めてエージェントを環境で動作させ、
        パフォーマンスを評価する
        """
        epsilon_backup = self.agt.epsilon
        self.agt.epsilon = eval_epsilon
        self.eval_simhist.reset()
        obs = self.eval_env.reset()
        timestep = 0
        episode = 0
        done = False

        # シミュレーション開始
        while True:
            if done is False:
                act = self.agt.select_action(obs)
                next_obs, rwd, next_done = self.eval_env.step(act)
                self.eval_simhist.add(act, rwd, next_done) # 記録用
            else:
                next_obs = self.eval_env.reset()
                rwd = None
                next_done = False

            obs = next_obs
            done = next_done

            if eval_n_step > 0:
                if timestep >= eval_n_step:
                    break
            if eval_n_episode > 0:
                if episode > eval_n_episode:
                    break

            timestep += 1
            episode += done
        self.eval_simhist.record()

        self.agt.epsilon = epsilon_backup

        return self.eval_simhist

    def _show_Q(self):
        if self.obss is None:
            return

        print('')
        print('Q values')
        for obss in self.obss:
            for obs in obss:
                val = self.agt.get_Q(np.array(obs))
                if val[0] is not None:
                    valstr = [' % .2f' % v for v in val]
                    print('Obs:%s  Q:%s' % (str(np.array(obs)), ','.join(valstr)))
            print('')


    def save_history(self, pathname):
        """
        履歴をセーブ
        """
        np.savez(pathname + '.npz',
            eval_rwds=self.hist_eval_rwds_in_episode,
            eval_steps=self.hist_eval_steps_in_episode,
            eval_x=self.hist_eval_x,
        )

    def load_history(self, pathname):
        """
        履歴をロード
        """
        hist = np.load(pathname + '.npz')
        self.hist_eval_rwds_in_episode = hist['eval_rwds'].tolist()
        self.hist_eval_steps_in_episode = hist['eval_steps'].tolist()
        self.hist_eval_x = hist['eval_x'].tolist()
        self.hist_start_x = self.hist_eval_x[-1]


class SimHistory:
    """
    シミュレーションの履歴保存クラス
    """
    def __init__(self):
        self.reset()

    def reset(self, init_step = 0):
        """
        履歴をリセット
        """
        self.mean_rwds = []
        self.rwds = []
        self.rwd  = 0

        self.mean_STEPs_in_episode = []
        self.steps_in_episode = []
        self.step_in_episode = 0

        self.steps_for_mean = []
        self.stepcnt = init_step

    def add(self, act, rwd, done):  # pylint:disable=unused-argument
        """
        履歴の追加
        """
        self.rwd += rwd
        self.step_in_episode += 1
        self.stepcnt += 1
        if done is True:
            self.rwds.append(self.rwd)
            self.rwd = 0
            self.steps_in_episode.append(self.step_in_episode)
            self.step_in_episode = 0

    def record(self):
        """
        履歴の平均の計算と保存
        """
        self.mean_rwds.append(np.mean(self.rwds))
        self.rwds = []

        self.mean_STEPs_in_episode.append(np.mean(self.steps_in_episode))
        self.steps_in_episode = []
        self.steps_for_mean.append(self.stepcnt)
