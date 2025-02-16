import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Суммарная интенсивность входного потока
l = 25
# Количество генерируемых данных
Num = 2000


def write2file(filename, data):
    df = pd.DataFrame(data)
    df.to_csv(filename, sep='\t', header=False, index=False)


def sim_time(func, *args):
    sum = -1
    t = 0
    while t < 1:
        t += func(*args)
        sum += 1
    return sum


def gen_burst():
    filename = "burst_data.csv"
    s = 0
    t = 0
    lambda_01 = 0.15
    lambda_10 = 0.3
    lambda_0 = 5
    lambda_1 = (l * (lambda_01 + lambda_10) - lambda_0 * lambda_10) / lambda_01
    data = []
    arr = []
    while t < Num:
        tau = 0
        if s % 2 == 0:
            tau = random.expovariate(lambda_01)
            time = t
            while time < tau + t:
                r = random.expovariate(lambda_0)
                time += r
                if time < tau + t:
                    arr.append(time)
            s += 1
        else:
            tau = random.expovariate(lambda_10)
            time = t
            while time < tau + t:
                r = random.expovariate(lambda_1)
                time += r
                if time < tau + t:
                    arr.append(time)
            s += 1
        t += tau

    for i in range(1, Num):
        data.append(sum(1 for x in arr if i - 1 < x <= i))

    print(np.mean(data))
    write2file(filename, data)


def gen_exp():
    filename = "exp_data.csv"
    data = []
    for i in range(0, Num):
        r = sim_time(random.expovariate, l)
        data.append(r)
    print(np.mean(data))
    write2file(filename, data)
    # plt.plot(data)
    # plt.show()


def gen_lognormal():
    filename = "lognormal_data.csv"
    data = []
    sigma = 1
    mu = np.log(1 / l) - (sigma ** 2) / 2
    for i in range(0, Num):
        r = sim_time(random.lognormvariate, mu, sigma)
        data.append(r)
    print(np.mean(data))
    write2file(filename, data)
    # plt.plot(data)
    # plt.show()


gen_exp()
gen_lognormal()
gen_burst()
