# Setup for Jupyter Notebook

There are several steps to use the jupyter notebook:
1. Build the jupyter notebook docker image
2. Run the jupyter notebook docker image (which will start jupyter hub and you can access the notebook at `http://<your_hostname>:8888`)

```bash
# build the jupyter notebook image
docker build -t dgl:jupyter \
    --build-arg BASE_IMAGE=<desired_dgl_image> \
    -f Dockerfile.dgl_jupyter \
    .

# run the jupyter notebook image (this will start jupyter hub
# and you can access the notebook at http://<your_hostname>:8888)
docker run -it --rm --privileged \
    --cap-add=SYS_PTRACE \
    --ipc=host \
    --privileged=true \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --security-opt seccomp=unconfined \
    dgl:jupyter
```

