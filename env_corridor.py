"""
env_corridor.py
廊下の環境
"""
import sys
from enum import Enum, auto
import numpy as np
import cv2
import core


class TaskType(Enum):
    """
    タスクタイプの列挙型
    """
    L4g23 = auto()
    L5g34 = auto()
    # mytask = auto() # オリジナルタスクタイプを追加

    @classmethod
    def Enum_of(cls, task_str):
        """
        タスクの文字列を列挙型に変換
        """
        for t in TaskType:
            if t.name == task_str:
                return t
        return None


class Env(core.coreEnv):
    """
    フィールドが1次元のシンプルな迷路
    """

    # 内部表現のID
    ID_brank = 0
    ID_agt = 1
    ID_goal = 3  # env_swanptour.py と合わせて3にしている

    def __init__(
            self,
            field_length=4,
            goal_candidate=(2, 3),
            pos_start=0,
            reward_fail=0,
            reward_move=0,
            reward_goal=1,
        ):
        """
        Parameters
        ----------

        field_length: int
            フィールドの長さ
        goal_candidate: list of int (0からfield_length-1まで)
            ゴールの出る場所, [2, 3], [2, 3, 5] などとする
        reward_fail: float
            脱出に失敗した時の報酬
        reward_move: float
            1歩進むことのコスト
        reward_goal: float
            成功した時の報酬
        pos_start: int (0からfield_length-1まで)
            スタート位置
        """
        # parameters
        self.field_length = field_length
        self.goal_candidate=goal_candidate
        self.pos_start = pos_start
        self.reward_fail = reward_fail
        self.reward_move = reward_move
        self.reward_goal = reward_goal
        super().__init__()

        # 行動数
        self.n_action = 2

        # 変数
        self.agt_pos = None
        self.goal_pos = None
        self.field = None
        self.is_first_step = None
        self.done = None
        self.reward = None
        self.action = None
        self.agt_state = None # render 用

    def set_task_type(self, task_type):
        """
        task_type を指定して、parameterを一括設定する
        """

        if task_type == TaskType.L4g23:
            self.field_length = 4
            self.goal_candidate = (2, 3)
            self.pos_start = 0
            self.reward_fail = 0
            self.reward_move = 0
            self.reward_goal = 1

        elif task_type == TaskType.L5g34:
            self.field_length = 5
            self.goal_candidate = (3, 4)
            self.pos_start = 0
            self.reward_fail = 0
            self.reward_move = 0
            self.reward_goal = 1

            """
        elif task_type == TaskType.mytask:  # オリジナルタスクタイプ追加
            self.field_length = 20
            self.goal_candidate = list(range(10, 20))
            self.pos_start = 0
            self.reward_fail = 0
            self.reward_move = 0
            self.reward_goal = 1
            """

        else:
            raise ValueError('task_type が間違っています')


    def reset(self):
        """
        内部状態をリセットする
        """
        self.done = False
        self.reward = None
        self.action = 1
        self.agt_state = 'move' # render 用

        self.agt_pos = self.pos_start
        self.field = np.ones(self.field_length, dtype=int) * Env.ID_brank
        idx = np.random.randint(len(self.goal_candidate))
        self.goal_pos = self.goal_candidate[idx]

        self.is_first_step = True
        observation = self._make_observation()
        return observation

    def step(self, action):
        """
        action にしたがって環境の状態を 1 step 進める
        """
        # next state
        if action == 1: # 進む
            next_pos = self.agt_pos + 1
            if next_pos >= self.field_length:
                reward = self.reward_fail
                done = True
                self.agt_state = 'fail'
            else:
                self.agt_pos = next_pos
                reward = self.reward_move
                done = False
                self.agt_state = 'move'
        elif action == 0: # try
            if self.agt_pos == self.goal_pos:
                reward = self.reward_goal
                done = True
                self.agt_state = 'goal'
            else:
                reward = self.reward_fail
                done = True
                self.agt_state = 'fail'

        observation = self._make_observation()

        # render 用
        self.done = done
        self.reward = reward
        self.action = action

        return observation, reward, done

    def _make_observation(self):
        """
        現在の状態から、エージェントが受け取る入力情報を生成
        """
        observation = self.field.copy()
        observation[self.goal_pos] = Env.ID_goal
        observation[self.agt_pos] = Env.ID_agt
        return observation

    def render(self):
        #  パラメータ
        unit = 50
        col_brank = (0, 255, 0)
        col_agt = (255, 255, 255)
        col_agt_miss = (0, 0, 255)
        col_agt_rwd = (50, 200, 50)
        col_agt_edge = (0, 0, 0)
        col_goal = (255, 100, 0)

        width = unit * self.field_length
        height = unit

        img = np.zeros((height, width, 3), dtype=np.uint8)

        # 背景の描画
        cv2.rectangle(img, (0, 0), (width-1, height-1), col_brank, -1)

        # ゴールの描画
        r0 = (unit * self.goal_pos, 0)
        r1 = (unit * (self.goal_pos + 1), height - 1)
        cv2.rectangle(img, r0, r1, col_goal, -1)

        # ロボットの色
        if self.agt_state == 'fail':
            col = col_agt_miss
        elif self.agt_state == 'goal':
            col = col_agt_rwd
        else:
            col = col_agt

        # ロボットの描画
        radius = int(unit * 0.35)
        r0 = (int(unit * self.agt_pos + unit/2), int(unit / 2))
        cv2.circle(img, r0, radius, col, -1)
        cv2.circle(img, r0, radius, col_agt_edge, 2)

        # ロボットの方向
        dr = np.array([0, 1])
        if self.action == 1:
            dr = np.array([1, 0])
        radius = int(unit * 0.2)
        r1 = np.array(r0) + unit * 0.25 * dr
        r1 = r1.astype(int)
        cv2.circle(img, tuple(r1), radius, col_agt_edge, -1)

        return img


def show_obs(obs, act, rwd, done):
    """
    変数を表示
    """
    if act is not None:
        print('%s act:%d, rwd:% .2f, done:%s' % (obs, act, rwd, done))
    else:
        print('start')
        print('%s' % (obs))


if __name__ == '__main__':

    argvs = sys.argv

    if len(argvs) < 2:
        MSG = '\n' + \
            '---- 操作方法 -------------------------------------\n' + \
            '[task type] を指定して実行します\n' + \
            '> python env_corridor.py [task_type]\n' + \
            '[task_type]\n' + \
            '%s\n' % ', '.join([t.name for t in TaskType])
        print(MSG)
        sys.exit()

    env = Env()
    ttype = TaskType.Enum_of(argvs[1])
    if ttype is None:
        MSG = '\n' + \
            '[task type] が異なります。以下から選んで指定してください。\n' + \
            '%s\n' % ', '.join([t.name for t in TaskType])
        print(MSG)
        sys.exit()

    env.set_task_type(ttype)
    MSG = '\n' + \
        '---- 操作方法 -------------------------------------\n' + \
        '[f] 右に進む\n' + \
        '[d] チャレンジ\n' + \
        '池の位置でチャレンジを押すと成功\n' + \
        '---------------------------------------------------'
    print(MSG)
    print('[task_type]: %s\n' % argvs[1])
    is_process = False
    done = False
    obs = env.reset()
    act = None
    rwd = None
    done = False
    show_obs(obs, act, rwd, done)
    while True:
        image = env.render()
        cv2.imshow('env', image)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
        if key == ord('d'):
            act = 0
            is_process = True

        if key == ord('f'):
            act = 1
            is_process = True

        if is_process is True:
            if done is True:
                obs = env.reset()
                act = None
                rwd = None
                done = False
            else:
                obs, rwd, done = env.step(act)

            show_obs(obs, act, rwd, done)

            is_process = False
