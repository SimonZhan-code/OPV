import array
import multiprocessing
# from multiprocessing import Process, Value, Queue
from multiprocessing.queues import Queue
from labjack import ljm
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import datetime
import os
import sys
import time
from scipy.signal import butter, lfilter, freqz
import math
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct
import pywt
from scipy import signal as sig
import function_wei
import pyqtgraph as pg
import array
import serial
import threading

class SharedCounter(object):
    """ A synchronized shared counter.
    The locking done by multiprocessing.Value ensures that only a single
    process or thread may read or write the in-memory ctypes object. However,
    in order to do n += 1, Python performs a read followed by a write, so a
    second process may read the old value before the new one is written by the
    first process. The solution is to use a multiprocessing.Lock to guarantee
    the atomicity of the modifications to Value.
    This class comes almost entirely from Eli Bendersky's blog:
    http://eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing/
    """

    def __init__(self, n=0):

        self.count = multiprocessing.Value('i', n)

    def increment(self, n=1):
        """ Increment the counter by n (default = 1) """
        with self.count.get_lock():
            self.count.value += n

    @property
    def value(self):
        """ Return the value of the counter """
        return self.count.value


class Queue(multiprocessing.queues.Queue):
    """ A portable implementation of multiprocessing.Queue.
    Because of multithreading / multiprocessing semantics, Queue.qsize() may
    raise the NotImplementedError exception on Unix platforms like Mac OS X
    where sem_getvalue() is not implemented. This subclass addresses this
    problem by using a synchronized shared counter (initialized to zero) and
    increasing / decreasing its value every time the put() and get() methods
    are called, respectively. This not only prevents NotImplementedError from
    being raised, but also allows us to implement a reliable version of both
    qsize() and empty().
    """

    def __init__(self, *args, **kwargs):
        ctx = multiprocessing.get_context()
        super(Queue, self).__init__(*args, **kwargs, ctx=ctx)
        self.size = SharedCounter(0)

    def put(self, *args, **kwargs):
        self.size.increment(1)
        super(Queue, self).put(*args, **kwargs)

    def get(self, *args, **kwargs):
        self.size.increment(-1)
        return super(Queue, self).get(*args, **kwargs)

    def qsize(self):
        """ Reliable implementation of multiprocessing.Queue.qsize() """
        return self.size.value

    def empty(self):
        """ Reliable implementation of multiprocessing.Queue.empty() """
        return not self.qsize()

    def clear(self):
        """ Remove all elements from the Queue. """
        while not self.empty():
            self.get()


def switch_pin(ljm, handle, numFrames,
               names_switch_1_io, names_switch_2_io, aValues, switch_chip_1_pin, switch_chip_2_pin):
    ljm.eWriteNames(handle, numFrames, names_switch_1_io, aValues[switch_chip_1_pin])
    ljm.eWriteNames(handle, numFrames, names_switch_2_io, aValues[switch_chip_2_pin])
    return switch_chip_1_pin, switch_chip_2_pin

def switch_one_pin(ljm, handle, numFrames, names_switch_1_io, aValues, switch_chip_1_pin):
    ljm.eWriteNames(handle, numFrames, names_switch_1_io, aValues[switch_chip_1_pin])
#     print(aValues[switch_chip_1_pin])
#     ljm.eWriteNames(handle, numFrames, names_switch_2_io, aValues[switch_chip_2_pin])
    return switch_chip_1_pin


def fft_result_logarithm(seg_data):
    y = np.squeeze(seg_data)
    n = len(y)  # length of the signal one second
    Y = np.fft.fft(y) / n  # fft computing and normalization
    Y1 = abs(Y[range(int(n / 2))])
    # logarithm output
    Y2 = 20 * np.log10(np.clip(np.abs(Y1), 1e-20, 1e100))
    return Y1, Y2

def fft_result_logarithm(seg_data):
    y = np.squeeze(seg_data)
    n = len(y)  # length of the signal one second
    Y = np.fft.fft(y) / n  # fft computing and normalization
    Y1 = abs(Y[range(int(n / 2))])
    # logarithm output
    Y2 = 20 * np.log10(np.clip(np.abs(Y1), 1e-20, 1e100))
    return Y1, Y2


def butter_bandpass_filter(data, lowcut, highcut, scan_rate, order=1):
    nyq = 0.5 * scan_rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y


def butter_lowpass(cutOff, fs, order=5):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog = True)
    return b, a


def butter_lowpass_filter(data, cutOff, fs, order=4):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def plot_waveletdata(plot_channel, rawdata_wavelet_np):
    lev = 5
    print("校验第 %i 通道的数据" % plot_channel)
    plt.subplot(211)
    plt.plot(rawdata_wavelet_np)
    plt.subplot(212)
    plt.plot(function_wei.wden(rawdata_wavelet_np, 'heursure', 'soft', 'one', lev, 'sym8'))


def get_rawdata_from_LJ(file_name, numIterations):

    s_timer = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
    fileName = file_name + s_timer + ".csv"
    # if os.path.exists(fileName):
    #     print("Please change the file_name")
    #     import sys
    #     sys.exit(1)
    f = open(fileName, 'a')

    handle = ljm.openS("T7", "ANY", "ANY")

    curIteration = 0
    rawdata_list = []

    while curIteration < numIterations:
        try:
            curTime = time.time()
            counter_sample = 0
            rawdata_list_one_second = []
            while time.time() - curTime < 1:
                counter_sample = counter_sample + 1
                rawdata_list_sample = []
                for i in range(8):  # channels
                    switch_pin_type = i
                    channel_num = switch_one_pin(ljm, handle, numFrames, names_switch_1_io, aValues, switch_pin_type)
                    #                     print(channel_num)
                    time.sleep(200 / 1000000)
                    result_0 = ljm.eReadName(handle, name_Ain_0)
                    rawdata_list_sample.append(counter_sample)
                    rawdata_list_sample.append(time.time())
                    rawdata_list_sample.append(result_0)
                    result_1 = ljm.eReadName(handle, name_Ain_1)
                    rawdata_list_sample.append(counter_sample)
                    rawdata_list_sample.append(time.time())
                    rawdata_list_sample.append(result_1)
                rawdata_list_one_second.append(rawdata_list_sample)
                rawdata_list.append(rawdata_list_sample)
            print("\n The time is: %s, The sample rate is: %s " % (
            datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"), counter_sample))
            np.savetxt(f, np.array(rawdata_list_one_second), delimiter=',', fmt='%.6f')
            curIteration = curIteration + 1
        except KeyboardInterrupt:
            break
        except Exception:
            import sys
            print(sys.exc_info()[1])
            break

    print("\nFinished!")
    return np.array(rawdata_list)


# def collect_data(queue_rawdata, file_name, timer_seconds):
def collect_data():
    global queue_rawdata;
    timer_seconds = 500

    file_name = "./Data/Env_250_Exp_250_1"

    s_timer = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
    file_name = file_name + s_timer + ".csv"
    # if os.path.exists(fileName):
    #     print("Please change the file_name")
    #     import sys
    #     sys.exit(1)
    f = open(file_name, 'a')

    handle = ljm.openS("T7", "ANY", "ANY")

    cur_iteration = 0


    channels_n = 4
    lev = 1

    while cur_iteration < timer_seconds:
        try:
            curTime = time.time()
            counter_sample = 0
            rawdata_list_one_second = []
            rawdata_list_one_second_queen = []
            while time.time() - curTime < 0.1:
                counter_sample = counter_sample + 1
                rawdata_list_sample_save = []
                rawdata_list_sample_queen = []
                for i in range(channels_n):  # channels
                    switch_pin_type = i
                    channel_num = switch_one_pin(ljm, handle, numFrames, names_switch_1_io, aValues, switch_pin_type)
                    #                     print(channel_num)
                    time.sleep(200 / 1000000)
                    result_0 = ljm.eReadName(handle, name_Ain_0)
                    rawdata_list_sample_save.append(counter_sample)
                    rawdata_list_sample_save.append(time.time())
                    rawdata_list_sample_save.append(result_0)
                    rawdata_list_sample_queen.append(result_0)

                    result_1 = ljm.eReadName(handle, name_Ain_1)
                    rawdata_list_sample_save.append(counter_sample)
                    rawdata_list_sample_save.append(time.time())
                    rawdata_list_sample_save.append(result_1)
                    rawdata_list_sample_queen.append(result_1)

                rawdata_list_one_second.append(rawdata_list_sample_save)
                rawdata_list_one_second_queen.append(rawdata_list_sample_queen)

            # print("\n The time is: %s, The sample rate is: %s " % (
            #     datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"), counter_sample))

            np.savetxt(f, np.array(rawdata_list_one_second), delimiter=',', fmt='%.6f')
            cur_iteration = cur_iteration + 1

            rawdata_np_one_second = np.array(rawdata_list_one_second_queen)
            rawdata_np_one_second_wavelet = rawdata_np_one_second.copy()

            # for i in range(channels_n * 2):
            #     rawdata_np_one_second_wavelet[:, i] = function_wei.wden(
            #         rawdata_np_one_second[:, i].reshape(-1,), 'heursure', 'soft', 'one', lev, 'sym8')
            # print(rawdata_np_one_second_wavelet.shaoe)
            queue_rawdata.put(rawdata_np_one_second_wavelet)
            # print(queue_rawdata.qsize())

        except KeyboardInterrupt:
            break
        except Exception:
            import sys
            print(sys.exc_info()[1])
            break

    print("\nFinished!")
    # return np.array(rawdata_list)


def plotData():
    global index_i;
    plot_data = queue_rawdata.get()
    index_set = plot_data.shape[0]
    for i in range(index_set):
        # print('index_set: ', index_set, ' i: ', i)
        if index_i < historyLength:
            data[index_i] = plot_data[i, 0]
            data2[index_i] = plot_data[i, 1]
            index_i = index_i+1
        else:
            data[:-1] = data[1:]
            data[index_i-1] = plot_data[i, 0]
            data2[:-1] = data2[1:]
            data2[index_i - 1] = plot_data[i, 1]
    curve.setData(data)
    curve2.setData(data2)

index_i = 0
queue_rawdata = Queue()
if __name__ == '__main__':
    # set config of labjack hardware
    name_Ain_0 = "AIN0"
    name_Ain_1 = "AIN2"
    names_switch_1_io = ["FIO0", "FIO1", "FIO2"]
    names_switch_2_io = ["FIO4", "FIO5", "FIO6"]
    aValues = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
               [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1], ]
    numFrames = len(names_switch_1_io)

    # timer_seconds = 150
    #
    # file_name = "./Data/Env_250_Exp_250_1"
    # 环境光强度、被测光强度、第几条数据

    # if not os.path.exists(file_name):
    #     os.makedirs(file_name)

    # rawdata_np = get_rawdata_from_LJ(file_name, timer_seconds)

    # print("The shape of the raw data : ", rawdata_np.shape)

    # channels_n = rawdata_np.shape[1] // 3
    #
    # for i in range(channels_n):
    #     plot_channel = i
    #     plt.figure(plot_channel)
    #     rawdata_wavelet_np = rawdata_np[:, 2 + 3 * plot_channel].copy().reshape(-1, )
    #     plot_waveletdata(plot_channel, rawdata_wavelet_np)


    #
    # # # set process
    # process_collect = Process(target=collect_data, args=(queue_rawdata, file_name, timer_seconds))
    # process_analyze = Process(target=analyze_data, args=(queue_rawdata,))
    # # # start process
    # process_collect.start()
    # process_analyze.start()
    # # # # finish process
    # process_collect.join()
    # process_analyze.terminate()

    app = pg.mkQApp()  # 建立app
    win = pg.GraphicsWindow()  # 建立窗口
    win.setWindowTitle(u'pyqtgraph逐点画波形图')
    win.resize(800, 500)  # 小窗口大小
    data = array.array('i')  # 可动态改变数组的大小,double型数组
    data2 = array.array('i')
    historyLength = 100  # 横坐标长度
    a = 0
    data = np.zeros(historyLength).__array__('d')  # 把数组长度定下来
    data2 = np.zeros(historyLength).__array__('d')
    p = win.addPlot()  # 把图p加入到窗口中
    p.showGrid(x=True, y=True)  # 把X和Y的表格打开
    p.setRange(xRange=[0, historyLength], yRange=[-2, 2], padding=0)
    p.setLabel(axis='left', text='y / V')  # 靠左
    p.setLabel(axis='bottom', text='x / point')
    p.setTitle('semg')  # 表格的名字
    curve = p.plot(pen='r')  # 绘制一个图形
    curve.setData(data)
    curve2 = p.plot(pen='g')  # 绘制一个图形
    curve2.setData(data2)

    th1 = threading.Thread(target=collect_data)
    th1.start()
    timer = pg.QtCore.QTimer()
    timer.timeout.connect(plotData)  # 定时刷新数据显示
    timer.start(1)  # 多少ms调用一次
    app.exec_()

