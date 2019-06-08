# This git repository contains the code for my Master thesis, Automatic Verification of UI Tests.
## Setup
The dependencies for this project can be installed by running the following in the project root:
```bash
pip install -r requirements.txt
```
If python or TensorFlow complains about missing DLL files, remove the tensorflow-gpu package

## Structure
Files in this repository uses a naming content consisting of the name of the object and the following suffixes:
* \*train.py files contain code to create and train models.
* \*test.py files contain code to produce classification reports and confusion matrices for models
* \*demo.py files contain code to classify images.

The repository itself is structured as follows:
* The bounding folder contains code for detecting a bounding-box in RMS. This was ultimately abandoned in favour of the Horizon and Well tests because it had limited real-world applicability.
* The colour folder contains the code that was used to detect 
* The multitest folder contains the code that was used to detect Horizons and wells in RMS.
  * convolutionvis and filtersvis contain code that can visualise the filters of a model

## A note on the current state of the code
Keep in mind that the code shown here is experimental, and could have been formatted better.

For instance, the model definition and training could have been moved to a seperate class/file. However, this was not done as the primary focus at the time was to finish the thesis.
