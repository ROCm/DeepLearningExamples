# Setup for Jupyter Notebook

There are several steps to use the jupyter notebook:
1. Build the jupyter notebook docker image
<<<<<<< HEAD
2. Run the jupyter notebook docker image (which will start jupyter hub and you can access the notebook at `http://<your_hostname>:8888`)
=======
2. [Optional] Download the pre-trained model checkpoint
3. Run the jupyter notebook docker image (which will start jupyter hub and you can access the notebook at `http://<your_hostname>:8888`)
>>>>>>> 3cb79959cec05de5fa4e576b95e4db8b5b3215b0

```bash
# build the jupyter notebook image
docker build -t dgl:jupyter \
    --build-arg BASE_IMAGE=<desired_dgl_image> \
    -f Dockerfile.dgl_jupyter \
    .

<<<<<<< HEAD
=======
# [Optional] Download the pre-trained model checkpoint
git lfs install
git lfs pull "model_qm9_100.pth"

>>>>>>> 3cb79959cec05de5fa4e576b95e4db8b5b3215b0
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

