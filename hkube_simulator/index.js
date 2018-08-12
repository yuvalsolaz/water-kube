var app = require('express')();
var http = require('http').Server(app);
var io = require('socket.io')(http);

port = 5678;

app.get('/', function(req, res){
  res.sendFile(__dirname + '/index.html');
});


io.on('connection', function(socket){
  console.log('a user connected');
    socket.on('control-message', function(msg){
      console.log('message: ' + msg);
      io.emit('control-message', msg);
    });
    socket.on('disconnect', function(){
      console.log('user disconnected');
    });
});

http.listen(port, function(){
  console.log('listening on *:'+port);
});

