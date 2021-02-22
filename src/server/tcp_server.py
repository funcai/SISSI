import socket
import sys
import struct
from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer('paraphrase-distilroberta-base-v1')

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_address = ('localhost', 10000)
print("Starting up on {} port {}".format(*server_address))
sock.bind(server_address)

sock.listen(1)

while True:
    print("Waiting for connection")
    connection, client_address = sock.accept()

    try:
        print('connection from {}'.format(client_address))
        received = b''
        while True:
            data = connection.recv(16)
            #print('received {!r}'.format(data))
            
            if data:
                #print('sending response to client')
                #response = 'stonks'
                #connection.sendall(data)
                received = received + data
            else:
                print('no more data from {}'.format(client_address))
                print('received {!r}'.format(received))

                embedded_text = embedder.encode(str(received), convert_to_tensor=True)
                
                print(embedded_text.shape)
                
                response_bytes = b''

                for txt in embedded_text:
                    ba = struct.pack("d", txt)
                    response_bytes = response_bytes + ba

                connection.sendall(response_bytes)

                received = b''
                break

    finally:
        connection.close()
