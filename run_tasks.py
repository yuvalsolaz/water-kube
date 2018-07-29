import sys
import htask


def main(argv):
    if len(argv) < 2:
        print(f'usage: {argv[0]} <port> ')
        exit(1)

    port = argv[1]
    t1 = htask.HTask(port=port)
    print (f'start tasking on port {port}')


if __name__ == '__main__':
    main(sys.argv)
