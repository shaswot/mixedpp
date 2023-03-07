# START DOCKER CONTAINER
docker run --gpus all \
        -dit \
        -v ~/stash:/stash \
        --name mozart-titanium \
        -p 1111:8888 \
        -p 1611:6006 \
        bhootmali/mixprecpack:1.1 \
        screen -S jlab jupyter lab \
                    --no-browser \
                    --ip=0.0.0.0 \
                    --port 8888 \
                    --allow-root \
                    --notebook-dir='/repos' \
                    --token=''

