import numpy as np
from numpy import linalg
from scipy.stats import t as student


class Regression:
    Y = []
    X = []
    BACK_PLAN_MATRIX = []
    m = 0,
    n = 0,
    k = 0,
    beta = [],
    E = [],
    S = 0,
    S_VECTOR = [],
    t = [],
    alpha = 0.05,
    critical = 0,
    func = ''
    report = ''

    @staticmethod
    def _init(matrix: list):
        for row in matrix:
            Regression.Y.append(row[0])

            current_row = [1]
            for index in range(1, len(row)):
                current_row.append(row[index])

            Regression.X.append(current_row)

        Regression.m = len(Regression.Y)
        Regression.n = len(Regression.X[0])
        Regression.k = Regression.m - Regression.n - 1

        Regression._create_back_plan_matrix()
        Regression._create_regression_function()
        Regression._create_e()
        Regression._create_s()
        Regression._create_s_vector()
        Regression._create_t()
        Regression._create_critical()
        Regression._create_report()

    @staticmethod
    def _create_back_plan_matrix():
        x = Regression.X
        x_transpose = np.transpose(x)
        plan_matrix = np.dot(x_transpose, x)
        Regression.BACK_PLAN_MATRIX = linalg.matrix_power(plan_matrix, -1)

    @staticmethod
    def _create_regression_function():
        x = Regression.X
        y = Regression.Y
        x_transpose = np.transpose(x)
        back_plan_matrix = Regression.BACK_PLAN_MATRIX
        Regression.beta = np.dot(np.dot(back_plan_matrix, x_transpose), y).tolist()
        result = Regression.beta

        func = f'{result[0]} + '
        for index in range(1, len(result)):
            func += f'({result[index]})x{index} + '

        Regression.func = func + 'e'

    @staticmethod
    def _create_e():
        Regression.E = np.subtract(Regression.Y, np.dot(Regression.X, Regression.beta)).tolist()
        return Regression.E

    @staticmethod
    def _create_s():
        factor = 1 / Regression.k
        e = Regression.E

        sum_e = 0
        for element in e:
            sum_e += element ** 2

        Regression.S = np.sqrt(factor * sum_e)

    @staticmethod
    def _create_s_vector():
        back_plan_matrix = Regression.BACK_PLAN_MATRIX
        s_vector = []

        for index in range(0, len(back_plan_matrix)):
            s_vector.append(np.sqrt(back_plan_matrix[index][index]))

        Regression.S_VECTOR = s_vector

    @staticmethod
    def _create_t():
        t = []
        for index in range(0, len(Regression.beta)):
            t.append(np.abs(Regression.beta[index] / Regression.S_VECTOR[index]))

        Regression.t = t

    @staticmethod
    def _create_critical():
        corrected_alpha = 1 - Regression.alpha[0] / 2
        Regression.critical = student.ppf([corrected_alpha], Regression.k)

    @staticmethod
    def _create_report():
        critical = Regression.critical[0]
        alpha = Regression.alpha[0]
        report = f'Регрессионная модель: {Regression.func}\n'
        report += (f'Критическое значение из таблицы Стьюдента для k={Regression.k}'
                   f' и alpha={alpha}: {critical}\n')

        for index in range(0, len(Regression.t)):
            current_t = Regression.t[index]
            report += f't[{index}] = {current_t}, '
            if current_t > Regression.critical:
                report += f'{current_t} > {critical}, => критерий является значимым \n'
            else:
                report += f'{current_t} < {critical}, => критерий не является значимым \n'

        Regression.report = report

    @staticmethod
    def get_report(matrix: list):
        Regression._init(matrix)

        return Regression.report
