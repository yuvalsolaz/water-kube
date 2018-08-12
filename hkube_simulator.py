import sys
import catboost_task
import os
import getopt

from socketIO_client import SocketIO

def hkube_simulator(ip, port):
    print('runing hkube simulator pipeline on: {}:{}...'.format(ip, port))
    socket_io = SocketIO(ip, port)
    socket_io.emit('connect')
    socket_io.emit('initialize')
    socket_io.emit('start')
    socket_io.on('started',print('simulator pipeline started'))
    socket_io.wait()


def main(argv):
    port = os.getenv('WORKER_SOCKET_PORT', 5678)
    print('starting hkube simulator on port {}'.format(port))
    print('run task in dubug mode')
    hkube_simulator('127.0.0.1', port)
    print('done')
'''

    try:
        opts, args = getopt.getopt(argv, "h:d:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print(getopt.GetoptError.msg)
        print('python hkube_simulator.py -d <debug>')
        sys.exit(2)
    if len(opts) < 1:
        print('python hkube_simulator.py -d <debug>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('python hkube_simulator.py -d <debug>')
            sys.exit()
        elif opt in ("-d", "--debug"):
            print('starting hkube simulator on port {}'.format(port))
            print('run task in dubug mode')
            hkube_simulator('127.0.0.1', port)
            print('done')
'''

if __name__ == '__main__':
    main(sys.argv[1:])
