python -m pip install --upgrade pip
<<<<<<< HEAD
python -m pip install numpy
=======
python -m pip install setuptools wheel pybind11 numpy
python -m pip install cuda-python nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12
python -m pip install --only-binary=:all: "cuda-python>=12,<13"
>>>>>>> prosperity-cu_and_python
python -m pip freeze > reqs.txt
