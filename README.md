# Neuron Detection in Calcium Imaging Data

Guy Teichman & Moran Neuhof, Winter 2019



### The data

The images we worked on should be in the "images/" folder. Unfortunately we cannot provide the images due to github's size limitations, so please make sure to acquire the images and place them in the project folder. The ground truth regions are provided in the regions/ folder. Please make sure you download the repository as-is in order for the code to run properly.

### Installing

1. The code relies on a few external packages, including cv2 and neurofinder. In order for the code to work properly, please install the `requirements.txt` file as follows:

```shell
> pip install -r requirements.txt
```

2. The code depends on the folder images/ (the files which we are analyzing in this project), so make sure the images/ folder is in the project's folder before running the code.

### Running the code

In order to run the code, simply run the following command from within the project folder.

```shell
> python NeuronDetectionProject.py
```

Make sure that the Segmentation/ folder is inside the project folder - it's part of the code!



A demonstration of the analysis, and the code we ran, can be found in the following notebooks:

- [Calcium Imaging - Neuron Detection Project.ipynb](https://github.com/neuhofmo/calcium-imaging-neuron-detection/blob/master/Calcium%20Imaging%20-%20Neuron%20Detection%20Project.ipynb)
- [Calcium Imaging - Plot Fluorescence.ipynb](https://github.com/neuhofmo/calcium-imaging-neuron-detection/blob/master/Calcium%20Imaging%20-%20Plot%20Fluorescence.ipynb)

A comprehensive tutorial of the methods used can be found in the document Neuron Detection in [Calcium Imaging Data.md](https://github.com/neuhofmo/calcium-imaging-neuron-detection/blob/master/Neuron%20Detection%20in%20Calcium%20Imaging%20Data.md).

