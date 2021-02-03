"""
共通で使う関数
"""
import numpy as np
import matplotlib.pyplot as plt


def copy_img(img, img_obj, x, y, isTrans=False):
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
    ※ img を受け取らなくても、入力したimg が変更される
    
    """
    h, w = img_obj.shape[:2]
    x1 = x + w
    y1 = y + h
    if isTrans is True:
        idx = np.where(np.all(img_obj==255, axis=-1))
        img_back = img[y:y1, x:x1, :].copy()
        img_obj[idx] = img_back[idx]
    img[y:y1, x:x1, :] = img_obj
    
    return img
  


def show_graph(pathname, target_reward=None, target_step=None):
    """
    学習曲線の表示
    """
    hist = np.load(pathname + '.npz')
    eval_rwd = hist['eval_rwds'].tolist()
    eval_step = hist['eval_steps'].tolist()
    eval_x = hist['eval_x'].tolist()

    plt.figure(figsize=(10,5))
    plt.subplots_adjust(hspace=0.6)

    # reward / episode
    plt.subplot(211)
    plt.plot(eval_x, eval_rwd, 'b.-')
    if target_reward is not None:
        plt.plot(
            [eval_x[0], eval_x[-1]],
            [target_reward, target_reward],
            'r:')

    plt.title('reward / episode')
    plt.grid(True)

    # steps / episode
    plt.subplot(212)
    plt.plot(eval_x, eval_step, 'b.-')
    if target_step is not None:
        plt.plot(
            [eval_x[0], eval_x[-1]],
            [target_step, target_step],
            'r:')
    plt.title('steps / episode')
    plt.grid(True)

    plt.show()
