docker run -u 0 \
            -v /home/nzolman/projects/sindy-rl:/home/firedrake/sindy-rl:z \
            -v /local/nzolman:/local/nzolman:z \
            -p 8888:8888 \
            -it hydrogym-sindy:latest \
            /bin/bash