const http = require('http');
const fs = require('fs');
const child_process = require('child_process');
const port = 3000;

const index = fs.readFileSync('./index.html');

const requestHandler = (request, response) => {
  if (request.url.indexOf('/suggest') === 0) {
    // send suggestion to rnn_suggest.py
    let text = decodeURI(request.url.substring('/suggest?text='.length));
    console.log(text);
    child_process.exec('python3 rnn_suggest.py "' + text + '"', {}, (err, stdout, stderr) => {
      response.end(stdout);
    });
  } else {
    // static index page
    response.end(index);
  }
}

const server = http.createServer(requestHandler)

server.listen(port, (err) => {
  if (err) {
    return console.log('something bad happened', err)
  }

  console.log(`server is listening on ${port}`)
});
