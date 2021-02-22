"""
env_field.py
クリスタルタスクの環境
"""
import sys
from enum import Enum, auto
import numpy as np
import cv2

# 自作モジュール
import core
import myutil


PATH_ROBOT = [
    'rsc/robo_back.png',
    'rsc/robo_left.png',
    'rsc/robo_front.png',
    'rsc/robo_right.png',
]

# PATH_CRYSTAL = 'rsc/crystal_big.png'
PATH_CRYSTAL = 'rsc/crystal.png'
PATH_WALL = 'rsc/wall.png'
PATH_BRANK = 'rsc/brank.png'


class TaskType(Enum):
    """
    タスクタイプの列挙型
    """
    open_field = auto()
    fixed_cave = auto()
    four_crystals = auto()
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
    複数のゴール（沼）と壁がある2D迷路問題
    """
    # 内部表現のID
    ID_brank = 0
    ID_agt = 1
    ID_wall = 2
    ID_goal = 3

    # 方向
    dr = np.array([
            [0, -1],
            [-1, 0],
            [0, 1],
            [1, 0],
        ]) 
  
    def __init__(  # pylint:disable=too-many-arguments, too-many-locals
            self,
            field_size=5,
            sight_size=3,
            max_time=30,
            n_wall=1,
            n_goal=2,
            start_pos=(3, 3),
            start_dir=0,
            reward_hit_wall=-0.2,
            reward_move=-0.1,
            reward_goal=1.0,
            maze_type='random',
            wall_observable=True,
            obs_in_render=True,
            sight_in_render=True,
        ):
        """
        Parameters
        ----------
        field_size :int
            フィールドの大きさ
        sight_size: int
            視野の大きさ(field_sizeよりも小さくする)
        max_time: int
            タイムリミット
        n_wall: int
            壁の数
        n_goal: int
            ゴールの数
        start_pos: (int, int)
            スタート地点
        start_dir: int (0, 1, 2, or 3)
            スタート時の方向
        reward_hit_wall: float
            壁に当たったときの報酬
        reward_move: float
            動きのコスト
        reward_goal: float
            ゴールに到達したときの報酬
        maze_type='random': str
            迷路タイプ
            'random', 'fixed_maze01'
        wall_observable: bool
            壁を観測に入れる
        obs_in_render: bool
            renderの出力にobsを入れる
        sight_in_render: bool
            renderでsight外を暗くする
        """
        self.field_size = field_size
        self.sight_size = sight_size
        self.max_time = max_time
        self.n_wall = n_wall
        self.n_goal = n_goal
        self.start_pos = start_pos
        self.start_dir = start_dir
        self.reward_hit_wall = reward_hit_wall
        self.reward_move = reward_move
        self.reward_goal = reward_goal
        self.maze_type = maze_type
        self.wall_observable = wall_observable
        self.obs_in_render = obs_in_render
        self.sight_in_render = sight_in_render

        super().__init__()

        # 行動数
        self.n_action = 3

        # 変数
        self.agt_pos = None
        self.agt_dir = None
        self.n_visited_goal = 0  # ゴールを訪れた回数
        self.field = None  # フィールド行列
        self.done = None  # 最終状態だったらTrue
        self.reward = None
        self.action = None
        self.time = None
        self.agt_state = None  # render 用
        self._truefield = None

        # 画像のロード
        self.img_robot = []
        for path in PATH_ROBOT:
            self.img_robot.append(cv2.imread(path))
        self.img_crystal = cv2.imread(PATH_CRYSTAL)
        self.img_brank = cv2.imread(PATH_BRANK)
        self.img_wall = cv2.imread(PATH_WALL)
        self.unit = self.img_robot[0].shape[0]

    def set_task_type(self, task_type):
        """
        task_type を指定して、parameterを一括設定する
        """
        if task_type == TaskType.open_field:
            self.field_size = 5
            self.sight_size = 4
            self.max_time = 15
            self.n_wall = 0
            self.n_goal = 1
            self.start_pos = (2, 2)
            self.reward_hit_wall = -0.2
            self.reward_move = -0.1
            self.reward_goal = 1
            self.maze_type = 'random'
            self.wall_observable = False

        elif task_type == TaskType.fixed_cave:
            self.field_size = 5
            self.sight_size = 2
            self.max_time = 25
            self.n_wall = None
            self.n_goal = None
            self.start_pos = None
            self.reward_hit_wall = -0.2
            self.reward_move = -0.1
            self.reward_goal = 1
            self.maze_type = 'fixed_maze01'
            self.wall_observable = True

        elif task_type == TaskType.four_crystals:
            self.field_size = 7
            self.sight_size = 2
            self.max_time = 30
            self.n_wall = 4
            self.n_goal = 4
            self.start_pos = (3, 3)
            self.reward_hit_wall = -0.2
            self.reward_move = -0.1
            self.reward_goal = 1
            self.maze_type = 'random'
            self.wall_observable = True
        else:
            raise ValueError('task_type が間違っています')

    def reset(self):
        self.done = False
        self.reward = None
        self.action = None
        self.agt_state = 'move'  # render 用
        self.time = 0
        self.n_visited_goal = 0

        if self.maze_type == 'random':
            # 迷路をランダム生成
            for i in range(100):
                self._make_maze()
                # 解けないパターンが生成される場合があるのでチェックをする
                possible_goal = self._maze_check()
                if possible_goal == self.n_goal:
                    break
                if i == 99:
                    raise ValueError('迷路が生成できません。壁の数を減らしてください')

        elif self.maze_type == 'fixed_maze01':
            """
            '-': brank
            'g': goal (crystal)
            'w': wall
            """

            # 以下の部分で、fiexed_field の迷路を変えることができます
            maze = [
                '--ww-',
                '-----',
                '-w--g',
                '-wgww',
                'ww--w',
                ]
            # ゴール（クリスタルの数）
            self.n_goal = 2
            # スタート地点と方向の指定
            self._my_maze(maze, start_pos=(0, 1), start_dir=0)

        else:
            raise ValueError('maze_type が間違っています')

        observation = self._make_observation()
        return observation


    def _my_maze(self, maze, start_pos=(0, 0), start_dir=0):
        """
        文字で表した迷路を行列に変換

        '-': brank
        'g': goal(crystal)
        'w': wall
        """
        myfield = []
        for mline in maze:
            line = []
            id_val = None
            for i in mline:
                if i == 'w':
                    id_val = Env.ID_wall
                elif i == '-':
                    id_val = Env.ID_brank
                elif i == 'g':
                    id_val = Env.ID_goal
                else:
                    raise ValueError('迷路のコードに解釈できない文字が含まれています')
                line.append(id_val)
            myfield.append(line)
        self._truefield = np.array(myfield, dtype=int)
        self.field_size = self._truefield.shape[0]
        self.start_pos = start_pos
        self.start_dir = start_dir

        # start
        self.agt_pos = self.start_pos
        self.agt_dir = self.start_dir


    def _make_maze(self):
        """
        ランダムな迷路を生成
        """
        # start
        self.agt_pos = self.start_pos
        self.agt_dir = self.start_dir

        # field
        self._truefield = np.ones((self.field_size,) * 2, dtype=int) * Env.ID_brank

        # goal
        for _ in range(self.n_goal):
            while True:
                x_val = np.random.randint(0, self.field_size)
                y_val = np.random.randint(0, self.field_size)
                if not(x_val == self.start_pos[0] and y_val == self.start_pos[1]) \
                    and self._truefield[y_val, x_val] == Env.ID_brank:
                    self._truefield[y_val, x_val] = Env.ID_goal
                    break

        # wall
        for _ in range(self.n_wall):
            for i in range(100):
                x_val = np.random.randint(0, self.field_size)
                y_val = np.random.randint(0, self.field_size)
                if not(self.agt_pos[0] == x_val and self.agt_pos[1] ==y_val) and \
                        self._truefield[y_val, x_val] == Env.ID_brank:
                    self._truefield[y_val, x_val] = Env.ID_wall
                    break
            if i == 99:
                print('壁の数が多すぎて迷路が作れません')
                sys.exit()

    def _maze_check(self):
        """
        スタート地点から到達できるゴールの数を出力
        """
        field = self._truefield
        f_h, f_w = field.shape
        x_agt, y_agt = self.agt_pos

        f_val = np.zeros((f_h + 2, f_w + 2), dtype=int)
        f_val[1:-1, 1:-1] = field
        enable = 99
        f_val[y_agt + 1, x_agt + 1] = enable
        possible_goal = 0
        while True:  # pylint:disable=too-many-nested-blocks
            is_change = False
            for i_y in range(1, f_h + 1):
                for i_x in range(1, f_w + 1):
                    if f_val[i_y, i_x] == enable:
                        f_val, is_change, reached_goal = \
                            self._count_update(f_val, i_x, i_y, enable)
                        possible_goal += reached_goal
            if is_change is False:
                break

        return possible_goal

    def _count_update(self, f_val, i_x, i_y, enable):
        d_agt = np.array([
                        [ 0, -1],
                        [-1,  0],
                        [ 1,  0],
                        [ 0,  1],
                        ]
            )
        is_change = False
        possible_goal = 0
        for i in range(d_agt.shape[0]):
            if f_val[i_y + d_agt[i, 0], i_x + d_agt[i, 1]] == Env.ID_brank or \
                f_val[i_y + d_agt[i, 0], i_x + d_agt[i, 1]] == Env.ID_goal:
                if f_val[i_y + d_agt[i, 0], i_x + d_agt[i, 1]] == Env.ID_goal:
                    possible_goal += 1
                    f_val[i_y + d_agt[i, 0], i_x + d_agt[i, 1]] = enable
                    is_change = True
                elif f_val[i_y + d_agt[i, 0], i_x + d_agt[i, 1]] == Env.ID_brank:
                    f_val[i_y + d_agt[i, 0], i_x + d_agt[i, 1]] = enable
                    is_change = True
                else:
                    raise ValueError('Err!')

        return f_val, is_change, possible_goal


    def step(self, action):
        """
        action にしたがって環境の状態を 1 step 進める
        """
        self.agt_state = 'move'  # render 用
        done = None

        # 次の状態を求める
        if action == 0:
            # 前進
            pos = self.agt_pos + Env.dr[self.agt_dir]
            if pos[0] < 0 or self.field_size <= pos[0] or \
                pos[1] < 0 or self.field_size <= pos[1]:
                # 範囲外に行こうとした
                self.agt_state = 'hit_wall'
                reward = self.reward_hit_wall
                done = False

            elif self._truefield[pos[1], pos[0]] == Env.ID_goal:
                # ゴールに訪れた場合
                self.agt_state = 'goal'
                self._truefield[pos[1], pos[0]] = Env.ID_brank
                reward = self.reward_goal
                self.n_visited_goal += 1
                if self.n_visited_goal == self.n_goal:
                    # 全てのゴールに訪れた
                    done = True
                    self.agt_pos = pos
                else:
                    # まだ全てではない
                    done = False
                    self.agt_pos = pos

            elif self._truefield[pos[1], pos[0]] == Env.ID_brank:
                # 何もないので進める
                self.agt_state = 'brank'
                self.agt_pos = pos
                reward = self.reward_move
                done = False

            elif self._truefield[pos[1], pos[0]] == Env.ID_wall:
                # 壁に進もうとした
                self.agt_state = 'hit_wall'
                reward = self.reward_hit_wall
                done = False

            else:
                raise ValueError('Err!')

        elif action == 1:
            # 右に向く
            self.agt_dir = (self.agt_dir + 1) % 4
            reward = self.reward_move
            done = False

        elif action == 2:
            # 左に向く
            self.agt_dir = (self.agt_dir - 1) % 4
            reward = self.reward_move
            done = False

        else:
            raise ValueError('Err!')

        # 時間切れ
        self.time += 1
        if self.time >= self.max_time:
            reward = self.reward_hit_wall
            done = True
            self.agt_state = 'timeover'

        observation = self._make_observation()

        # render 用
        self.done = done
        self.reward = reward
        self.action = action

        return observation, reward, done

    def _make_observation(self):
        """
        現在の状態から、エージェントが受け取る入力情報を生成
        入力情報は自己を中心としたゴールと壁の位置
        """
        # 周囲の壁を1で表す行列を生成
        around_wall = self._truefield.copy()
        around_wall[self._truefield == Env.ID_wall] = 1
        around_wall[self._truefield != Env.ID_wall] = 0

        # 周囲のクリスタルを1で表す行列を生成
        around_goal = self._truefield.copy()
        around_goal[self._truefield == Env.ID_goal] = 1
        around_goal[self._truefield != Env.ID_goal] = 0

        # goal 観測用、まずフィールドの3倍の大きさのobs_goalを作る
        f_s = self.field_size
        size = f_s * 3
        obs_goal = np.zeros((size, size), dtype=int)

        # agt_posを中心とした観測行列obs_goalを作成
        obs_goal[f_s:f_s * 2, f_s:f_s * 2] = around_goal
        s_s = self.sight_size
        x_val = f_s + self.agt_pos[0]
        y_val = f_s + self.agt_pos[1]
        obs_goal = obs_goal[y_val-s_s:y_val+s_s+1, x_val-s_s:x_val+s_s+1]

        # ロボットの方向に合わせて観測行列を回転
        if self.agt_dir == 3:
            obs_goal = np.rot90(obs_goal)
        elif self.agt_dir == 2:
            for _ in range(2):
                obs_goal = np.rot90(obs_goal)
        elif self.agt_dir == 1:
            for _ in range(3):
                obs_goal = np.rot90(obs_goal)

        # 同様に壁の観測行列を作成
        if self.wall_observable is True:
            obs_wall = np.ones((size, size), dtype=int)
            obs_wall[f_s:f_s * 2, f_s:f_s * 2] = around_wall
            obs_wall = obs_wall[y_val-s_s:y_val+s_s+1, x_val-s_s:x_val+s_s+1]
            if self.agt_dir == 3:
                obs_wall = np.rot90(obs_wall)
            elif self.agt_dir == 2:
                for _ in range(2):
                    obs_wall = np.rot90(obs_wall)
            elif self.agt_dir == 1:
                for _ in range(3):
                    obs_wall = np.rot90(obs_wall)

            obs = np.c_[obs_goal, obs_wall]
        else:
            obs = obs_goal

        return obs

    def render(self):
        """
        画面表示用の画像を生成
        ※エージェントの入力情報ではなくユーザー用
        """
        # フィールドの描画 --------------------

        # 画像サイズ
        unit = self.unit
        width = unit * self.field_size
        height = unit * self.field_size

        # 画像用の変数準備
        img_out = np.zeros((height, width, 3), dtype=np.uint8)

        # ブロック各種の描画
        for i_x in range(self.field_size):
            for i_y in range(self.field_size):
                # ブロックの描画開始座標
                r0 = (unit * i_x, unit * i_y)

                if self._truefield[i_y, i_x] == Env.ID_wall:
                    # 壁
                    myutil.copy_img(img_out, self.img_wall, r0[0], r0[1])
                else:
                    # ブランク
                    myutil.copy_img(img_out, self.img_brank, r0[0], r0[1])

                if self._truefield[i_y, i_x] == Env.ID_goal:
                    # ゴール
                    myutil.copy_img(img_out, self.img_crystal, r0[0], r0[1], isTrans=True)

        # ロボットの描画
        self._draw_robot(img_out)

        if self.sight_in_render is True:
            # 観測範囲の外側を暗くする
            img_out = self._draw_sight_effect(img_out)

        if self.obs_in_render is True:
            # 観測の描画
            img_obs = self._draw_observation(unit, height)

            # 画像の統合
            mgn_w = 10  # フィールドと観測値の境界線の太さ
            mgn_col = (200, 200, 0)
            img_mgn = np.zeros((height, mgn_w, 3), dtype=np.uint8)
            cv2.rectangle(
                img_mgn,
                (0, 0), (mgn_w, height), mgn_col, -1)

            img_out = cv2.hconcat([img_out, img_mgn, img_obs])


        return img_out

    def _draw_sight_effect(self, img):
        img_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        unit = self.unit
        ss = self.sight_size
        x0, y0 = (np.array(self.agt_pos) - np.array([ss, ss]))* unit
        x1, y1 = (np.array(self.agt_pos) + np.array([ss, ss]))* unit + unit
        img_mask = cv2.rectangle(img_mask, (x0, y0), (x1, y1), 255, -1)
        idx = img_mask == 0
        img[idx] = img[idx] * 0.5
        return img

    def _draw_robot(self, img):
        """
        ロボットを描く
        """
        col_good = (0, 0, 255)
        col_bad = (0, 200, 0)
        # ロボット画像の選択
        img_robot = self.img_robot[self.agt_dir].copy()

        idx = np.where(np.all(img_robot==224, axis=-1))
        if self.agt_state == 'hit_wall':
            img_robot[idx] = col_good
        elif self.agt_state == 'goal':
             img_robot[idx] = col_bad

        # ロボット画像の貼り付け        
        unit = self.unit
        x0, y0 = np.array(self.agt_pos) * unit
        img = myutil.copy_img(img, img_robot, x0, y0, isTrans=True)

        return img


    def _draw_observation(self, unit, height):
        """
        観測情報を描く
        """
        unit = self.unit
        col_agt_obs = (50, 50, 255)
        col_back = (159, 183, 207)
        col_obj = (100, 50, 50)

        # 画像の大きさを決めて画像用変数を作成
        observation = self._make_observation()
        obs_ih, obs_iw = observation.shape
        obs_unit = height / obs_ih  # 画像の縦の長さがフィールドと同じになるように obs_unitを決める
        obs_width = int(obs_unit * obs_iw)
        img_obs = np.zeros((height, obs_width, 3), dtype=np.uint8)

        # 背景色で塗りつぶし
        cv2.rectangle(
            img_obs, (0, 0), (obs_width, height),
            col_back, -1,
        )

        # 物体を描画
        rate = 0.8  # 四角を小さくして隙間が見えるようにする、その割合
        for i_y in range(observation.shape[0]):
            for i_x in range(observation.shape[1]):
                if observation[i_y, i_x] > 0:
                    r0 = (int(obs_unit * i_x), int(obs_unit * i_y))
                    r1 = (int(r0[0] + obs_unit * rate),  int(r0[1] + obs_unit * rate))
                    cv2.rectangle(img_obs, r0, r1, col_obj, -1)

        # 中心にマークを描画
        cy = int((obs_ih - 1) / 2)
        if obs_iw == obs_ih * 2:
            cxs = (cy, obs_ih + cy)
        else:
            cxs = (cy, )

        col = col_agt_obs  # ロボットの色
        for cx in cxs:
            r0 = (int(obs_unit * cx), int(obs_unit * cy))
            r1 = (int(r0[0] + obs_unit * rate),  int(r0[1] + obs_unit * rate))
            cv2.rectangle(img_obs, r0, r1, col, 2)
            
        return img_obs

def _show_obs(act, rwd, obs, done):
    """
    変数を表示
    """
    if act is not None:
        print('')
        print('act:%d, rwd:% .2f, done:%s' % (act, rwd, done))
        print(obs)
    else:
        print('')
        print('first obs:')
        print(obs)


if __name__ == '__main__':

    argvs = sys.argv

    if len(argvs) < 2:
        MSG = '\n' + \
            '---- 操作方法 -------------------------------------\n' + \
            '[task type] を指定して実行します\n' + \
            '> python env_field.py [task_type]\n' + \
            '[task_type]\n' + \
            '%s\n' % ', '.join([t.name for t in TaskType]) + \
            '---------------------------------------------------'

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
    MSG =  '---- 操作方法 -------------------------------------\n' + \
           '[e] 前に進む [s] 左に90度回る [f] 右に90度回る\n' + \
           '[o] 観測の表示のON/OFF\n' + \
           '[v] 視野の表示のON/OFF\n' + \
           '[q] 終了\n' + \
           '全てのクリスタルを回収するとクリア、次のエピソードが開始\n' + \
           '---------------------------------------------------'
    print(MSG)
    print('[task_type]: %s\n' % argvs[1])
    is_process = False
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
        if key == ord('e'):
            act = 0
            is_process = True

        if key == ord('f'):
            act = 2
            is_process = True

        if key == ord('s'):
            act = 1
            is_process = True

        if key == ord('o'):
            env.obs_in_render = bool(1 - env.obs_in_render)

        if key == ord('v'):
            env.sight_in_render = bool(1 - env.sight_in_render)

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
