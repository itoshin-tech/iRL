"""
env_corridor.py
廊下の環境
"""
import sys
from enum import Enum, auto
import numpy as np
import cv2


PATH_ROBOT = 'rsc/robo_right.png'
PATH_CRYSTAL = 'rsc/crystal_small.png'
PATH_BRANK = 'rsc/brank.png'


class TaskType(Enum):
    """
    タスクタイプの列挙型
    """
    short_road = auto()
    long_road = auto()

    @classmethod
    def Enum_of(cls, task_str):
        """
        タスクの文字列を列挙型に変換
        """
        for t in TaskType:
            if t.name == task_str:
                return t
        return None


class CorridorEnv():
    """
    フィールドが1次元のシンプルな迷路
    """
    # 内部表現のID
    ID_brank = 0
    ID_agt = 1
    ID_goal = 2

    def __init__(
            self,
            field_length=4,         # int: フィールドの長さ
            goal_candidate=(2, 3),  # tuple of int: ゴールの位置
            pos_start=0,            # スタート位置
            reward_fail=-1,          # 失敗した時の報酬（ペナルティ）
            reward_move=-1,          # 進んだ時の報酬（コスト）
            reward_goal=5,          # クリスタルを得た時の報酬
        ):
        """
        初期処理
        """
        super().__init__()

        # パラメータ
        self.field_length = field_length
        self.goal_candidate=goal_candidate
        self.pos_start = pos_start
        self.reward_fail = reward_fail
        self.reward_move = reward_move
        self.reward_goal = reward_goal

        # 行動数
        self.n_action = 2

        # 変数
        self.agt_pos = None
        self.goal_pos = None
        self.is_first_step = None
        self.agt_state = None # render 用

        # 画像のロード
        self.img_robot = cv2.imread(PATH_ROBOT)
        self.img_crystal = cv2.imread(PATH_CRYSTAL)
        self.img_brank = cv2.imread(PATH_BRANK)
        self.unit = self.img_robot.shape[0]

    def set_task_type(self, task_type):
        """
        task_type を指定して、parameterを一括設定する
        """
        if task_type == TaskType.short_road:  # (1)
            self.field_length = 4           # 廊下の長さ
            self.goal_candidate = (2, 3)    # クリスタルの出る位置の候補
            self.pos_start = 0              # スタート地点
            self.reward_fail = -1           # 「拾う」の失敗時の報酬（ペナルティ）
            self.reward_move = -1           # 「進む」の報酬（コスト）
            self.reward_goal = 5            # 「拾う」の成功時の報酬

        elif task_type == TaskType.long_road:  # (2)
            self.field_length = 5
            self.goal_candidate = (1, 2, 3, 4)
            self.pos_start = 0
            self.reward_fail = -1
            self.reward_move = -1
            self.reward_goal = 5

        else:
            raise ValueError('task_type が間違っています')


    def reset(self):
        """
        内部状態をリセットする
        """
        self.agt_state = 'move' # render 用

        self.agt_pos = self.pos_start
        idx = np.random.randint(len(self.goal_candidate))
        self.goal_pos = self.goal_candidate[idx]

        self.is_first_step = True
        observation = self._make_observation()
        return observation

    def step(self, action):
        """
        action にしたがって環境の状態を 1 step 進める
        """

        # 次の状態を求める
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
        elif action == 0: # 試す
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

        return observation, reward, done

    def _make_observation(self):
        """
        現在の状態から、エージェントが受け取る入力情報を生成
        """
        observation = np.ones(self.field_length, dtype=int) * CorridorEnv.ID_brank
        observation[self.goal_pos] = CorridorEnv.ID_goal
        observation[self.agt_pos] = CorridorEnv.ID_agt
        return observation

    def render(self):
        # 画像サイズ
        unit = self.unit
        width = unit * self.field_length
        height = unit

        # 画像用の変数準備
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # ブロック各種の描画
        for i_x in range(self.field_length):
            img = self._copy_img(img, self.img_brank, unit * i_x, 0)

        # ゴールの描画
        if self.agt_state != 'goal':
            img = self._copy_img(img, self.img_crystal, unit * self.goal_pos, 0, isTrans=True)

        # ロボットの描画
        img = self._draw_robot(img)

        return img

    def _draw_robot(self, img):
        """
        ロボットを描く
        """
        col_good = (0, 0, 255)
        col_bad = (0, 200, 0)

        # ロボット画像の選択
        img_robot = self.img_robot.copy()

        idx = np.where(np.all(img_robot==224, axis=-1))
        if self.agt_state == 'fail':
            img_robot[idx] = col_good
        elif self.agt_state == 'goal':
             img_robot[idx] = col_bad

        # ロボット画像の貼り付け
        unit = self.unit
        x0 = np.array(self.agt_pos) * unit
        img = self._copy_img(img, img_robot, x0, 0, isTrans=True)

        return img        

    def _copy_img(self, img, img_obj, x, y, isTrans=False):
        """
        img にimg_objをコピーする

        Parameters
        ----------
        img: 3d numpy.ndarray
            張り付ける先の画像
        img_obj: 3d numpy.ndarray
            張り付ける画像
        x, y: int
            img 上でのり付ける座標
        isTrans: bool
            True: 白を透明にする
        
        Returns
        -------
        img: 3d numpy.ndarray
            コピー後の画像
        
        """
        img_out = img.copy()
        h, w = img_obj.shape[:2]
        x1 = x + w
        y1 = y + h
        if isTrans is True:
            idx = np.where(np.all(img_obj==255, axis=-1))
            img_back = img[y:y1, x:x1, :].copy()
            img_obj[idx] = img_back[idx]
        img_out[y:y1, x:x1, :] = img_obj
        
        return img_out


def _show_obs(act, rwd, obs, done):
    """
    強化学習の情報をコンソールに表示
    """
    if act is not None:
        print('act:%d, rwd:% .2f, obs:%s, done:%s' % (act, rwd, obs, done))
    else:
        print('')
        print('first obs:%s' % (obs))


if __name__ == '__main__':

    argvs = sys.argv

    if len(argvs) < 2:
        MSG = '\n' + \
            '---- 実行方法 -------------------------------------\n' + \
            '[task type] を指定して実行します\n' + \
            '> python env_corridor.py [task_type]\n' + \
            '[task_type]\n' + \
            '%s\n' % ', '.join([t.name for t in TaskType]) + \
            '---------------------------------------------------'

        print(MSG)
        sys.exit()

    env = CorridorEnv()
    ttype = TaskType.Enum_of(argvs[1])
    if ttype is None:
        MSG = '\n' + \
            '[task type] が異なります。以下から選んで指定してください。\n' + \
            '%s' % ', '.join([t.name for t in TaskType])
        print(MSG)
        sys.exit()

    env.set_task_type(ttype)
    MSG = '\n' + \
        '---- 操作方法 -------------------------------------\n' + \
        '[f] 右に進む\n' + \
        '[d] 拾う\n' + \
        '[q] 終了\n' + \
        'クリスタルを拾うと成功\n' + \
        '---------------------------------------------------'
    print(MSG)
    print('[task_type]: %s' % argvs[1])
    is_process = False
    done = False
    obs = env.reset()
    act = None
    rwd = None
    done = False
    _show_obs(act, rwd, obs, done)
    while True:
        image = env.render()
        cv2.imshow('env', image)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
        if key in [ord('d'), ord(' ')]:
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

            _show_obs(act, rwd, obs, done)

            is_process = False
