import numpy as np
import matplotlib.pyplot as plt


def get_means(data_x, data_y):
    return zip(*sorted((xVal, np.mean([yVal for a, yVal in zip(data_x, data_y) if xVal == a]))
                       for xVal in set(data_x)))

if __name__ == '__main__':
    data_cuda = np.genfromtxt('out_cuda.csv', delimiter=',', names=['n', 'time'])
    data_opencl = np.genfromtxt('out_opencl.csv', delimiter=',', names=['n', 'time'])
    data_cpp = np.genfromtxt('out_cpp.csv', delimiter=',', names=['n', 'time'])

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    x_cuda, y_cuda = get_means(data_cuda['n'], data_cuda['time'])
    x_opencl, y_opencl = get_means(data_opencl['n'], data_opencl['time'])
    x_cpp, y_cpp = get_means(data_cpp['n'], data_cpp['time'])

    #ax1.plot(x_cpp, np.array(y_cpp)/1000, label='Secuencial')
    ax1.plot(x_cuda, y_cuda, label='CUDA')
    ax1.plot(x_opencl, y_opencl, label='OpenCL')

    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.xlabel('Tamaño matriz (n x n)')
    plt.ylabel('Tiempo (ms)')
    plt.subplots_adjust(right=0.75)
    plt.savefig('cuda-opencl.png')
    plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    y_seq = np.array(y_cpp[1:]) / 1000
    y = np.divide(y_seq, y_cuda)
    ax1.plot(x_cuda, np.divide(y_seq, y_cuda))
    plt.xlabel('Tamaño matriz (n x n)')
    plt.ylabel('Speedup')
    plt.savefig('speedup.png')
    plt.show()
