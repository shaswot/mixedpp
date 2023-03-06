# START DOCKER CONTAINER
docker run --gpus all \
        -dit \
        -v ~/stash:/stash \
        --name bach-titanium \
        -p 1111:8888 \
        -p 1611:6006 \
        mixedpp:1.0 \
        screen -S jlab jupyter lab --no-browser --ip=0.0.0.0 --port 8888 --allow-root --notebook-dir='/repos' --token=''

