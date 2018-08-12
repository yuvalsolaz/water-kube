import sys
import catboost_task
import os
import getopt

def main(argv):
    port = os.getenv('WORKER_SOCKET_PORT', 5678)
    t1 = catboost_task.catboostTask(port=port)
    print ('start tasking on port {}'.format(port))


if __name__ == '__main__':
    main(sys.argv[1:])
