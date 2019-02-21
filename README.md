# Neuron Detection in Calcium Imaging Data

Guy Teichman & Moran Neuhof, Winter 2019



### The data

The images we worked on are provided in the images/ folder. The ground truth regions are provided in the regions/ folder. Please make sure you download the repository as-is in order for the code to run properly.

### Installing

The code relies on a few external packages, including cv2 and neurofinder. In order for the code to work properly, please install the `requirements.txt` file as follows:

```shell
> pip install -r requirements.txt
```

### Running the code

In order to run the code, simply run the following command from within the project folder.

```shell
> python NeuronDetectionProject.py
```

Make sure that the Segmentation/ folder is inside the project folder - it's part of the code!



A demonstration of the analysis, and the code we ran, can be found in the following notebooks:

- Calcium Imaging - Neuron Detection Project.ipynb
- Calcium Imaging - Plot Fluorescence.ipynb

A comprehensive tutorial of the methods used can be found in the document Neuron Detection in Calcium Imaging Data.md.

