"""
便利な関数
"""
import numpy as np


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
  
