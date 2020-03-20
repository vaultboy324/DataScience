import numpy

class Factor:
    N = 0
    n = 0
    Z = []
    R = []
    V = []
    T = []
    A = []
    W = []
    eps = 0.2
    report = ''
    z_funcs = ''
    F = []

    @staticmethod
    def _init(dataset):
        columns = []
        for row in dataset:
            for index in range(0, len(row)):
                try:
                    columns[index].append(row[index])
                except(IndexError):
                    columns.append([row[index]])

        avgs = []
        stds = []
        for column in columns:
            avgs.append(numpy.mean(column))
            stds.append(numpy.std(column))

        Factor.N = len(dataset) + 1
        Factor.n = len(dataset[0])

        for index in range(0, len(columns)):
            z_row = []
            for element in columns[index]:
                z_row.append((element - avgs[index]) / stds[index])
            Factor.Z.append(z_row)

    @staticmethod
    def _create_r():
        z_transpose = numpy.transpose(Factor.Z)
        matrix_multiplication = numpy.dot(Factor.Z, z_transpose)
        Factor.R = numpy.multiply(1 / (Factor.N - 1), matrix_multiplication)

    @staticmethod
    def _create_t_and_v():
        Factor.V, Factor.T = numpy.linalg.eig(Factor.R)

    @staticmethod
    def _sort_t_and_v():
        rows = []

        for index in range(0, len(Factor.V)):
            rows.append({
                'index': index,
                'value': Factor.V[index]
            })
        rows.sort(key=lambda i: i['value'], reverse=True)

        new_t = []
        new_v = []

        for index in range(0, len(rows)):
            new_v.append(rows[index]['value'])
            new_t.append(Factor.T[rows[index]['index']])

        Factor.T = new_t
        Factor.V = new_v

    @staticmethod
    def _normalize_t():
        sums = []

        for t_row in Factor.T:
            for index in range(0, len(t_row)):
                try:
                    sums[index] += numpy.power(t_row[index], 2)
                except(IndexError):
                    sums.append(numpy.power(t_row[index], 2))

        for index in range(0, len(sums)):
            sums[index] = numpy.sqrt(sums[index])

        for t_row in Factor.T:
            for index in range(0, len(t_row)):
                t_row[index] = t_row[index] / sums[index]

    @staticmethod
    def _v_to_diag():
        sqrt_v = []

        for element in Factor.V:
            sqrt_v.append(numpy.sqrt(element))
        Factor.V = numpy.diag(sqrt_v)

    @staticmethod
    def _create_a():
        Factor.A = numpy.dot(Factor.T, Factor.V)

    @staticmethod
    def _create_w():
        delta = []

        for row in Factor.A:
            for index in range(0, len(Factor.A)):
                try:
                    delta[index] += numpy.power(row[index], 2)
                except(IndexError):
                    delta.append(numpy.power(row[index], 2))

        sum_delta = numpy.sum(delta)

        current_w = 0
        Factor.W.append(delta[current_w] / sum_delta)

        while 1 - Factor.W[current_w] > Factor.eps:
            numerator = 0
            current_w += 1
            for index in range(0, current_w + 1):
                numerator += delta[index]
            Factor.W.append(numerator / sum_delta)

    @staticmethod
    def _create_z_funcs():
        current_z = 1
        for row in Factor.A:
            Factor.z_funcs += f'Z[{current_z}] = '
            for index in range(0, len(Factor.W)):
                Factor.z_funcs += f'({row[index]}) * F[{index + 1}] + '
            Factor.z_funcs = Factor.z_funcs[0:-2]
            Factor.z_funcs += '\n'
            current_z += 1

    @staticmethod
    def _create_f():
        a_back = numpy.linalg.inv(Factor.A)

        for i in range(0, len(Factor.W)):
            current_factor = []
            for j in range(0, Factor.n):
                try:
                    current_factor += numpy.dot(a_back[i][j], Factor.Z[j])
                except(ValueError):
                    current_factor = numpy.dot(a_back[i][j], Factor.Z[j])
            Factor.F.append(list(current_factor))

    @staticmethod
    def _create_report():
        Factor.report += f'Z = {Factor.Z}\n\n'
        Factor.report += f'R = {Factor.R}\n\n'
        Factor.report += f'V = {Factor.V}\n\n'
        Factor.report += f'T = {Factor.T}\n\n'
        Factor.report += f'A = {Factor.A}\n\n'
        Factor.report += f'W = {Factor.W}\n\n'
        Factor.report += f'{Factor.z_funcs}\n'
        Factor.report += f'F = {Factor.F}\n\n'
        if Factor.eps == 0:
            Factor.report += f'A * F = {numpy.dot(Factor.A, Factor.F)}'

    @staticmethod
    def get_result(dataset):
        Factor._init(dataset)
        Factor._create_r()
        Factor._create_t_and_v()
        Factor._sort_t_and_v()
        Factor._normalize_t()
        Factor._v_to_diag()
        Factor._create_a()
        Factor._create_w()
        Factor._create_z_funcs()
        Factor._create_f()
        Factor._create_report()
        return Factor.report



