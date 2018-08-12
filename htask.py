
'''
Task socket IO interface implementation for hkube interface
'''

from socketIO_client import SocketIO

class HTask:

    def __init__(self, ip='127.0.0.1', port=5000):

        self.init_params = None
        print('listening on: {}:{}...'.format(ip,port))
        self.socketIO = SocketIO(ip, port)

        self.socketIO.on('connect', self.on_connect)
        self.socketIO.on('disconnect', self.on_disconnect)
        self.socketIO.on('reconnect', self.on_reconnect)
        self.socketIO.on('initialize', self.on_initialize)
        self.socketIO.on('start', self.on_start)
        self.socketIO.on('stop', self.on_stop)
        self.socketIO.on('done', self.on_done)
        self.socketIO.wait()

    def on_connect(self):
        print('on_connect')


    def on_disconnect(self):
        print('disconnect')


    def on_reconnect(self):
        print('reconnect')


    def on_initialize(self, *args ):
        print('on_init')
        self.init_params = args[0]
        self.send_message('initialized',{'command': 'initialized'})


    def on_start(self, *dummy_args):
        print('on_start')
        self.send_message('started', {'command':'started'})
        input_msg = self.init_params['data']['input'][0]
        print('input message: {}'.format(input_msg))


    def on_stop(self):
        print('stop')


    def on_done(self):
        print('done')

    def send_message(self, topic, out_message):
        self.socketIO.emit(topic, out_message)

