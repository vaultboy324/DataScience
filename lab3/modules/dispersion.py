from constants.fields import *

import numpy
from scipy import stats


class Dispersion:
    _X = []
    _levels_count = 0
    _objects_count = 0
    _k_1 = 0
    _k_2 = 0
    _full_avg = 0
    _levels_avg = []
    _alpha = 0.01
    _report = ''
    _q_1 = 0
    _q_2 = 0
    _fisher = 0
    _probability = 0

    @staticmethod
    def _init(dataset):
        for row in dataset:
            Dispersion._X.append(row[LIST])

        Dispersion._levels_count = len(Dispersion._X)
        Dispersion._objects_count = len(Dispersion._X[0])

    @staticmethod
    def _create_avgs():
        Dispersion._full_avg = numpy.mean(Dispersion._X)

        for row in Dispersion._X:
            Dispersion._levels_avg.append(numpy.mean(row))

    @staticmethod
    def _create_k():
        Dispersion._k_1 = Dispersion._levels_count - 1
        Dispersion._k_2 = Dispersion._levels_count * (Dispersion._objects_count - 1)

    @staticmethod
    def _create_q():
        result = 0
        for current_avg in Dispersion._levels_avg:
            result += numpy.power((current_avg - Dispersion._full_avg), 2)
        result *= Dispersion._objects_count
        Dispersion._q_1 = result

        result = 0
        for index in range(0, len(Dispersion._X)):
            for current_x in Dispersion._X[index]:
                result += numpy.power((current_x - Dispersion._levels_avg[index]), 2)
        Dispersion._q_2 = result

    @staticmethod
    def _create_fisher():
        Dispersion._fisher = (Dispersion._q_1 * (1 / Dispersion._k_1)) / (Dispersion._q_2 * (1 / Dispersion._k_2))

    @staticmethod
    def _create_critical():
        Dispersion._probability = stats.f.sf(Dispersion._fisher, Dispersion._k_1, Dispersion._k_2)

    @staticmethod
    def _create_report():
        Dispersion._report += f'X = {Dispersion._X}\n\n'
        Dispersion._report += f'n = {Dispersion._levels_count}\n\n'
        Dispersion._report += f'm = {Dispersion._objects_count}\n\n'
        Dispersion._report += f'k[1] = {Dispersion._k_1}\n\n'
        Dispersion._report += f'k[2] = {Dispersion._k_2}\n\n'
        Dispersion._report += f'avg(X) = {Dispersion._full_avg}\n\n'
        Dispersion._report += f'avg(x[i]) = {Dispersion._levels_avg}\n\n'
        Dispersion._report += f'Q[1] = {Dispersion._q_1}\n\n'
        Dispersion._report += f'Q[2] = {Dispersion._q_2}\n\n'
        Dispersion._report += f'Значение критерия Фишера = {Dispersion._fisher}\n\n'
        Dispersion._report += f'alpha = {Dispersion._alpha}\n\n'
        Dispersion._report += f'Вероятность события, где верна нулевая гипотеза = {Dispersion._probability}\n\n'
        if Dispersion._alpha < Dispersion._probability:
            Dispersion._report += 'Нет оснований отвергать нулевую гипотезу\n\n'
        else:
            Dispersion._report += 'Есть основания отвергнуть нулевую гипотезу\n\n'

    @staticmethod
    def get_result(dataset):
        Dispersion._init(dataset)
        Dispersion._create_avgs()
        Dispersion._create_k()
        Dispersion._create_q()
        Dispersion._create_fisher()
        Dispersion._create_critical()
        Dispersion._create_report()
        return Dispersion._report
