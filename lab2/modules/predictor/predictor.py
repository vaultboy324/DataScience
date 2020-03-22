import numpy


class Predictor:
    _X = []
    _Y = []
    _M_X = []
    _M_Y = 0
    _a_transpose = []
    _S_inv = []
    _subtract_result = []
    _a_transpose_s_inv = []
    _a_transpose_s_inv_m_x = []
    _x_functional = ''
    _report = ''

    @staticmethod
    def _init(dataset):
        x = []
        y = []

        for row in dataset:
            y.append(row[0])
            for index in range(1, len(row)):
                try:
                    x[index - 1].append(row[index])
                except IndexError:
                    x.append([row[index]])

        Predictor._X = x
        Predictor._Y = y

    @staticmethod
    def _create_m_y():
        Predictor._M_Y = numpy.mean(Predictor._Y)

    @staticmethod
    def _create_m_x():
        for current_x in Predictor._X:
            Predictor._M_X.append(numpy.mean(current_x))

    @staticmethod
    def _create_inv_s():
        s = numpy.cov(Predictor._X)
        Predictor._S_inv = numpy.linalg.inv(s)

    @staticmethod
    def _create_a_transpose():
        a = []
        for current_x in Predictor._X:
            current_cov = numpy.cov(Predictor._Y, current_x)
            a.append(current_cov[0][1])
        Predictor._a_transpose = numpy.transpose(a)

    @staticmethod
    def _create_multiplication_result():
        Predictor._a_transpose_s_inv = numpy.dot(Predictor._a_transpose, Predictor._S_inv)
        Predictor._a_transpose_s_inv_m_x = numpy.dot(Predictor._a_transpose_s_inv, Predictor._M_X)

    @staticmethod
    def __create_functional():
        for index in range(0, len(Predictor._a_transpose_s_inv)):
            Predictor._x_functional += f'({Predictor._a_transpose_s_inv[index]})x[{index + 1}] + '
        Predictor._x_functional = Predictor._x_functional[0:-2]
        return Predictor._x_functional

    @staticmethod
    def _create_report():
        Predictor._report += f'X = {Predictor._X}\n\n'
        Predictor._report += f'Y = {Predictor._Y}\n\n'
        Predictor._report += f'M(X) = {Predictor._M_X}\n\n'
        Predictor._report += f'M(Y) = {Predictor._M_Y}\n\n'

        Predictor.__create_functional()
        Predictor._report += (f'P(X) = {Predictor._M_Y} + '
                              f'{Predictor._x_functional} - '
                              f'({Predictor._a_transpose_s_inv_m_x})\n')
        Predictor._report += f'P(X) = {Predictor._M_Y - Predictor._a_transpose_s_inv_m_x} + {Predictor._x_functional}'

    @staticmethod
    def get_result(dataset):
        Predictor._init(dataset)
        Predictor._create_m_y()
        Predictor._create_m_x()
        Predictor._create_a_transpose()
        Predictor._create_inv_s()
        Predictor._create_multiplication_result()
        Predictor._create_report()
        return Predictor._report

