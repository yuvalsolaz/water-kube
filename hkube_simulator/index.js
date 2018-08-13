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
      console.log('control message: ' + msg);
      if (msg == 'initialize') 
        io.emit(msg,{1:'todo : input data here '});
      else
        io.emit(msg); 
    });
    socket.on('initialized', function(msg){
      console.log('algorithm message: ' + JSON.stringify(msg));
    });
    socket.on('done', function(msg,data){
      console.log('algorithm message: ' + JSON.stringify(msg) + ' data: ' + JSON.stringify(data));
    });
    socket.on('disconnect', function(){
      console.log('user disconnected');
    });
});

http.listen(port, function(){
  console.log('listening on *:'+port);
});

