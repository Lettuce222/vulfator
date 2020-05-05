import cv2 as cv
import numpy as np
from PIL import Image

BLUE = [255,0,0]        # rectangle color
RED = [0,0,255]         # possible BG
GREEN = [0,255,0]       # possible FG
BLACK = [0,0,0]         # obvious BG
WHITE = [255,255,255]   # obvious FG

DRAW_BG = {'color' : BLACK, 'val' : 0}
DRAW_FG = {'color' : WHITE, 'val' : 1}
DRAW_PR_FG = {'color' : GREEN, 'val' : 3}
DRAW_PR_BG = {'color' : RED, 'val' : 2}


# 現在の描画する線の色
value = DRAW_FG


# 描画する線の太さ
thickness = 3


# ウィンドウ名
win_name = "WINDOW"


# grabcut結果表示の際、alphaが大きいほど黒背景が濃ゆくなる
alpha = 0.7
beta = 1 - alpha


# grabcutイテレーション回数
itr = 1


# 描画フラグ
drawing = False

display = None
mask = None


def mouse_event(event, x, y, flags, param):
    global display, mask, thickness, value, drawing, bgdModel, fgdModel

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        cv.circle(display, (x, y), thickness, value['color'], -1)
        cv.circle(mask, (x, y), thickness, value['val'], -1)


    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            cv.circle(display, (x, y), thickness, value['color'], -1)
            cv.circle(mask, (x, y), thickness, value['val'], -1)


    elif event == cv.EVENT_LBUTTONUP:
        if drawing == True:
            drawing = False
            cv.circle(display, (x, y), thickness, value['color'], -1)
            cv.circle(mask, (x, y), thickness, value['val'], -1)


def cut_main_obj(filename):
    global display, mask, thickness, value, drawing, bgdModel, fgdModel
    # 出力イメージ番号
    pic_num = 1

    img = cv.imread(filename)
    display = img.copy()

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    bgdModel = np.zeros((1,65), dtype=np.float64)
    fgdModel = np.zeros((1,65), dtype=np.float64)

    cv.namedWindow(win_name, cv.WINDOW_NORMAL)

    roi = cv.selectROI(win_name, img)

    cv.grabCut(img, mask, roi, bgdModel, fgdModel, itr, cv.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    display = img * mask2[:,:,np.newaxis]
    display = cv.addWeighted(display, alpha, img, beta, 0)

    cv.setMouseCallback(win_name, mouse_event)

    while (True):

        cv.imshow(win_name, display)
        key = cv.waitKey(1)

        if key == ord('e'):
            cv.destroyAllWindows()
            break


        elif key == ord("n"):

            drawing = False

            cv.grabCut(img, mask, roi, bgdModel, fgdModel, itr, cv.GC_INIT_WITH_MASK)
            mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
            display = img * mask2[:,:,np.newaxis]
            display = cv.addWeighted(display, alpha, img, beta, 0)


        elif key == ord("s"):
            cv.grabCut(img, mask, roi, bgdModel, fgdModel, itr, cv.GC_INIT_WITH_MASK)
            mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
            img_rgba = cv.cvtColor(display, cv.COLOR_RGB2RGBA)
            output = img_rgba * mask2[:,:,np.newaxis]

            # Pillow形式に変換
            output = cv.cvtColor(output, cv.COLOR_BGRA2RGBA)
            output = Image.fromarray(output)
            output = output.crop(output.getbbox())

            return output.convert("RGBA")

        elif key == ord('0'): # BG drawing
            print(" mark background regions with left mouse button \n")
            value = DRAW_BG
            print("Obvious BGD")


        elif key == ord('1'): # FG drawing
            print(" mark foreground regions with left mouse button \n")
            value = DRAW_FG
            print("Obvious FGD")


        elif key == ord('2'): # PR BG drawing
            value = DRAW_PR_BG
            print("Possible BGD")


        elif key == ord('3'): # PR FG drawing
            value = DRAW_PR_FG
            print("Possible FGD")

def main():
  filename = "data/input/shoes.jpg"
  main_obj = cut_main_obj(filename)
  image = Image.new("RGB", (900, 900), (156, 191, 223)).convert("RGBA")

  max_length = max(main_obj.width, main_obj.height)
  resize_rate = 500 / max_length
  new_width = int(main_obj.width * resize_rate)
  new_height = int(main_obj.height * resize_rate)
  main_obj = main_obj.resize((new_width, new_height))

  image.paste(main_obj, (int(450-new_width/2), int(450-new_height/2)), main_obj)
  image.save(filename.replace('input', 'output').replace('jpg', 'png'))
  main_obj.save('temp.png')

if __name__ == '__main__':
  main()
