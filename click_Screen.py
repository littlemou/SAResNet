#colab训练时防止中断，模拟点击屏幕
import pyautogui
import time
pyautogui.click(1000,1000)
while(True):
    time.sleep(600)
    pyautogui.click(1000,1000)