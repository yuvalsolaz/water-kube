import sys
import catboost_task
import os

def main():
    port = os.getenv('WORKER_SOCKET_PORT', 5678)
    t1 = catboost_task.catboostTask(port=port)
    print ('start tasking on port {}'.format(port))


if __name__ == '__main__':
    main()
