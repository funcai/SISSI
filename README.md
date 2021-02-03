## Running in docker
Execute the following command to build the docker container

    docker build -t sissi .

After that's done, start the docker container while making sure to add your gpus and sufficient shared memory

    docker run --gpus all --shm-size=1024m -it --rm -v $PWD:/tmp -w /tmp -p 8081:8081 sissi bash

To train the model run

    cd src/train && python test_train.py