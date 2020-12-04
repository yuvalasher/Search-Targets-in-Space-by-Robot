from bayesian_updating import bayesian_updating_iteration
from utils import Location, CONFIG_PATH
from configparser import ConfigParser
from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import random
from typing import Dict, Any

config = ConfigParser()
config.read(CONFIG_PATH)

def sensitivity(param_name, values):
    info_gain_res = {}
    ent_res = {}
    convergence_iter = {}

    for alpha in values:
        config.set("PARAMS", param_name, str(alpha))
        params = config["PARAMS"]
        ent_upd, info_gain_upd, conv_iter, t_pron, non_t_prob = bayesian_updating_iteration(params, until_convergence=False, sensitivity_analysis=True)
        info_gain_res[alpha] = info_gain_upd
        ent_res[alpha] = ent_upd
        convergence_iter[alpha] = [conv_iter]

    return info_gain_res, ent_res, convergence_iter


def plot_results(info_gain_res, ent_res, convergence_iter, param_name):
    fig, ax = plt.subplots()
    for k, v in info_gain_res.items():
        ax.plot(list(range(len(v))), v, label=k, markevery=convergence_iter[k], marker='D', markerfacecolor='black')
    ax.legend(title=param_name)
    ax.set_title(
        'Information gain change over time - for different values of {} - (Black dot - time of convergence)'.format(
            param_name))
    plt.xlabel('time')
    plt.ylabel('Information gain')
    plt.show()

    fig, ax = plt.subplots()
    for k, v in ent_res.items():
        ax.plot(list(range(len(v))), v, label=k, markevery=convergence_iter[k], marker='D', markerfacecolor='black')
    ax.legend(title=param_name)
    ax.set_title(
        'Entropy change over time - for different values of {} - (Black dot - time of convergence)'.format(param_name))
    plt.xlabel('time')
    plt.ylabel('Entropy')
    plt.show()


def alpha_sensitivity():
    info_gain, ent, convergence_iter = sensitivity("alpha", [0.1, 0.25, 0.5, 0.75])
    plot_results(info_gain, ent, convergence_iter, "Alpha")


def sensor_power_sens():
    info_gain, ent, convergence_iter = sensitivity("LAMBDA_SENSOR", [5, 10, 15, 20])
    plot_results(info_gain, ent, convergence_iter, "Sensor Power")


def pta_sens():
    info_gain, ent, convergence_iter = sensitivity("pta", [0.25, 0.5, 0.75, 1])
    plot_results(info_gain, ent, convergence_iter, "Pta")


def distance_from_center_sens():
    list_of_loc = [(0, 0), (2, 2), (5, 5), (9, 9)]
    info_gain, ent, convergence_iter = sensitivity("AGENT_POSITION", list_of_loc)

    center = (9, 9)
    for loc in list_of_loc:
        dist = np.round(np.sqrt(np.power(loc[0] - center[0], 2) + np.power(loc[1] - center[1], 2)))
        info_gain[dist] = info_gain.pop(loc)
        ent[dist] = ent.pop(loc)
        convergence_iter[dist] = convergence_iter.pop(loc)

    plot_results(info_gain, ent, convergence_iter, "Distance from center")


def plot_targ_non_targ_prob():
    params = config["PARAMS"]
    ent_upd, info_gain_upd, conv_iter, t_pron, non_t_prob = bayesian_updating_iteration(params, until_convergence=False, sensitivity_analysis=True)

    fig, ax = plt.subplots()
    ax.plot(list(range(len(t_pron))), t_pron)
    ax.set_title('Target probability change until convergence')
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.show()

    fig, ax = plt.subplots()
    for dis, loc_his in non_t_prob.items():
        ax.plot(list(range(len(loc_his))), loc_his, label=dis)
    ax.set_title('Non Target probability change until convergence')
    plt.legend(title="Distance from agent")
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.show()


def plot_info_gain_and_entropy():
    params = config["PARAMS"]
    ent_upd, info_gain_upd, conv_iter, t_pron, non_t_prob = bayesian_updating_iteration(params, sensitivity_analysis=True)

    fig, ax = plt.subplots()
    ax.plot(list(range(len(ent_upd))), ent_upd)
    ax.set_title('Entropy change until convergence')
    plt.xlabel('Time')
    plt.ylabel('Entropy')
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(list(range(len(info_gain_upd))), info_gain_upd)
    ax.set_title('Information gain change until convergence')
    plt.xlabel('Time')
    plt.ylabel('Information gain')
    plt.show()


if __name__ == '__main__':
    params = config["PARAMS"]
    alpha_sensitivity()
    # sensor_power_sens()
    # pta_sens()
    # distance_from_center_sens()
    # plot_targ_non_targ_prob()
    # plot_info_gain_and_entropy()