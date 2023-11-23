# Season-invariant GNSS-denied visual localization for UAVs

This repository contains data and code for manuscript J. Kinnari, F. Verdoja, V. Kyrki ["Season-invariant GNSS-denied visual localization for UAVs"](https://doi.org/10.1109/LRA.2022.3191038).

## Setting up
Clone this repository on your computer.

Update on Nov 23, 2023: Pretrained checkpoint has been removed. Please contact author for acccess to it.

Install [Anaconda](https://www.anaconda.com/). Create a new environment using the provided yml file:

    conda env create -f environment.yml

Remark: The ```environment.yml``` file currently contains more packages than necessary.

## Simple example
A simple example for computing similarity between two 96 by 96 pixel RGB images is provided in ```computeSimilarity.py```. A few sample images can be found in folder ```sample_images```. To check operation of the trained model with pairs of images, run

    python3 computeSimilarity.py ../sample_images/sample1_anc.png ../sample_images/sample1_pos.png

## Configuration
The file ```configs/configuration.yml``` contains various configuration details with comments for each parameter.

## Training
After downloading the data to folder ```data``` as instructed at the end of this file, you may train the network yourself using the script ```trainSimilarityModel.py```.

## Citing
If this code is useful in your research, please consider citing:

```
@ARTICLE{9830867,
  author={Kinnari, Jouko and Verdoja, Francesco and Kyrki, Ville},
  journal={IEEE Robotics and Automation Letters}, 
  title={Season-Invariant GNSS-Denied Visual Localization for UAVs}, 
  year={2022},
  volume={7},
  number={4},
  pages={10232-10239},
  doi={10.1109/LRA.2022.3191038}}
```

## Downloading data

### What does one dataset contain?

Each dataset contains a number of aligned images, acquired at different times, from the same area, at a resolution of approximately 1 m/pixel.

### Where to store images

Place each image in the corresponding folder under folder ```data``` or adjust paths in ```configuration.yml``` to match your configuration.

### Acquiring Woodbridge and Fountainhead datasets

Rows 1 and 2 in the table below are from [this paper by Goforth and Lucey](https://doi.org/10.1109/ICRA.2019.8793558).

### Acquiring ge_area datasets

Rows 3 through 18 in the table below are acquired using [Google Earth Pro desktop](https://www.google.com/earth/versions/#earth-pro). Google Earth license does not allow distribution of data acquired with Google Earth Pro and therefore the dataset cannot be distributed as images directly. Also, Google Earth Pro does not appear to have a suitable API for exporting these kinds of images using a script.

However, in order to acquire the same dataset, you can download the same images with a bit of manual work according to these instructions:

- Import each placemark file (.kmz) found in folder ```ge_placemarks``` into Google Earth Pro
- Click View/Historical Imagery to enable historical imagery.
- From Layers panel, remove tickmarks next to each layer selection.

For each placemark:
- Double-click the placemark to center the view in the correct position, at the correct altitude
- From Places tab, remove the tickmark for that placemark in order to remove it from view.
- Slide the slider to one of the month/date combinations listed in table below.
- Click the "Save Image" button. From Map Options, disable the items "Title and Description", "Legend", "Scale", "Compass" and "HTML Area"
- From "Resolution", select "Maximum (4800 x 2987)"
- Click "Save Image". Save the image with filename "\<year\>\_\<month\>\_\<number\>.jpg" in the folder defined in the table below.

These instructions were written for Google Earth Pro 7.3.4.8248 (64-bit) on Windows.

### Listing of datasets

<table>
<tr><th>Row</th><th>Dataset name</th><th>Imaging months</th><th>Dataset use</th><th>Path</th></tr>
<td>1</td><td>woodbridge</td><td> 2006/8 2008/8 2010/8 2013/8 2015/8 2017/8</td><td>tra</td><td>../data/USGS/sat_data/woodbridge/images</td></tr>
<td>2</td><td>fountainhead</td><td> 2009/6 2012/5 2016/7</td><td>tra</td><td>../data/USGS/sat_data/fountainhead/images</td></tr>
<td>3</td><td>ge_area1</td><td> 2011/3,4,6 2015/8,8 2018/5</td><td>tra</td><td>../data/google_earth_exports/area1</td></tr>
<td>4</td><td>ge_area2</td><td> 2011/3,4,6 2013/8 2014/7 2015/8 2017/8 2018/5</td><td>tra</td><td>../data/google_earth_exports/area2</td></tr>
<td>5</td><td>ge_area4</td><td> 2013/8 2014/7 2015/8 2017/7 2018/5,5 2020/3,3</td><td>tra</td><td>../data/google_earth_exports/area4</td></tr>
<td>6</td><td>ge_area5</td><td> 2011/3 2013/3,8 2014/7 2015/8 2017/7,8 2018/5 2020/6</td><td>tra</td><td>../data/google_earth_exports/area5</td></tr>
<td>7</td><td>ge_area6</td><td> 2002/7 2011/3 2013/3,8 2014/7 2015/8 2017/7 2018/5 2020/6</td><td>tra</td><td>../data/google_earth_exports/area6</td></tr>
<td>8</td><td>ge_area12</td><td> 2013/4 2014/7 2017/7 2018/5 2020/6</td><td>tra</td><td>../data/google_earth_exports/area12</td></tr>
<td>9</td><td>ge_area13</td><td> 2011/3,9 2012/5 2013/4 2020/3,6</td><td>tra</td><td>../data/google_earth_exports/area13</td></tr>
<td>10</td><td>ge_area14</td><td> 2011/6 2012/5 2013/4,8 2015/8</td><td>tra</td><td>../data/google_earth_exports/area14</td></tr>
<td>11</td><td>ge_area15</td><td> 2015/8 2016/7 2018/5 2020/4</td><td>tra</td><td>../data/google_earth_exports/area15</td></tr>
<td>12</td><td>ge_area3</td><td> 2006/5 2008/4 2011/9 2012/5 2013/4,8 2014/5,7 2015/8 2017/7 2018/5,5 2020/3,3,6</td><td>tes</td><td>../data/google_earth_exports/area3</td></tr>
<td>13</td><td>ge_area3</td><td> 2006/5 2008/4 2011/9 2012/5 2013/4,8 2014/5,7 2015/8 2017/7 2018/5,5 2020/3,3,6</td><td>mcl</td><td>../data/google_earth_exports/area3</td></tr>
<td>14</td><td>ge_area7</td><td> 2002/7 2011/3 2013/3,8 2014/7 2015/8 2017/7 2018/5 2020/6</td><td>mcl</td><td>../data/google_earth_exports/area7</td></tr>
<td>15</td><td>ge_area8</td><td> 2008/5 2010/7 2011/8 2013/4 2014/4 2017/5,8 2019/8 2020/3,6</td><td>lik</td><td>../data/google_earth_exports/area8</td></tr>
<td>16</td><td>ge_area9</td><td> 2011/3,6 2017/8 2019/7 2021/7</td><td>lik</td><td>../data/google_earth_exports/area9</td></tr>
<td>17</td><td>ge_area10</td><td> 2012/5 2014/8 2020/6 2021/4</td><td>lik</td><td>../data/google_earth_exports/area10</td></tr>
<td>18</td><td>ge_area11</td><td> 2005/2 2013/6 2014/8 2019/4 2020/6 2021/4</td><td>lik</td><td>../data/google_earth_exports/area11</td></tr>
</table>

"Dataset use" column specifies the use of this dataset:
- tra: used in training the network
- tes: used in testing the network
- mcl: used for Monte Carlo localization tests
- lik: used for estimating likelihood from similarity score
