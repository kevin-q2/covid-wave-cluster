# Covid Wave Clustering :ocean:

In this repository is an implementation of a infection wave clustering methodology which has code to 

1. Split infection time-series into wave-like segments
2. Cluster those waves in a way that respects their timing within a global time period

We consider a wave to be a short period of pronounced infection activity, and argue that modeling and clustering waves 
gives a good way to understand the evolving spatio-temporal patterns of a disease or virus.

## Installation 
To build a minimal installation, first ensure that poetry is installed 
as a package manager. If you do not have poetry installed, 
instructions and basic usage  may be found [here](https://python-poetry.org/docs/). 
Importantly, this uses package is built with python version 3.12. 

Once poetry is installed, clone the repository
and run:

```
poetry install
```

I'd recommend simply cloning the entire repository in order to get all of the data and example notebooks, but installing just the 
code as a library is possible as well. 

Wherever we use the wavefinder or wav segmentation method, we are comparing our segmentations to 
another good segmentation method implemented [here](https://github.com/covid19db/epidemiological-waves/).
Please refer to their paper/code for details on how to use it.

## Data
Integral to our analyses and experiments is the data collected and provided in `data/` with attribution given to the following sources:
  1. [Google's Covid-19 Covid Repository](https://github.com/GoogleCloudPlatform/covid-19-open-data) records daily infection records, demographics,
     and geographical information from locations around the world. 
  2. [Oxford's Covid Government Response Tracker](https://github.com/OxCGRT/covid-policy-dataset) records daily information about each locations
     stringency or proactive responses to containment of the virus. We include their average
     containment health index measurements for both US states and a set of European countries.
  3. For visualization we use state[] (2022) and county (2021) shape files collected from the US [census](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.2022.html#list-tab-790442341), as well as shape files for countries provided by [OpenDataSoft](https://public.opendatasoft.com/explore/dataset/world-administrative-boundaries/information/?flg=en-us&location=10,22.37175,114.10565&basemap=jawg.light)

NOTE: County data is not included in this repository since the files are too large. Please manually download from the sources above to use this dataset. All tools for cleaning the data are provided in `data/county/cleaning/`.

## Instructions

Most of our analyses and experiments can be found in the jupyter notebooks in `examples/`. There you'll 
find most of the code needed to run the segmentation and clustering algorithms, and produce 
nice figures to view the results. 

We note that these are often computationally intensive processes which cannot be confined to a notebook. 
So often in the notebooks you'll see that we import data from the `batch/` folder. In `batch/` we include 
sets of python scripts which can be run to collect that data. Again, those are often computationally intensive to run,
so we also include folders `batch/state/data/` or `batch/country/data` with data for precomputed segmentations, pairwise 
distance matrices, and cluster analysis information. 
     
