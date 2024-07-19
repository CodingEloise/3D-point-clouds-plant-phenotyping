# 3D-point-clouds-plant-phenotyping
Automated segmentation and annotation of 3D point clouds for plant phenotyping



## File Structure

.
├── colored_annotation_processor.py   # Code for visualizing class and instance annotations
├── data                              # LAST-Straw dataset
├── output
│   ├── parts_and_instances           # Visualisation of the class and instance annotations
│   └── parts_only                    # Visualisation of the class annotations
└── README.md                         # This README file



## Data

The dataset used is LAST-Straw, which can be found at [LAST-Straw Dataset](https://lcas.lincoln.ac.uk/nextcloud/index.php/s/omQY9ciP3Wr43GH).



## Running the file

Download the LAST-Straw dataset and place it under the `./data` directory.

Make sure to install necessary libraries:

```sh
pip install numpy open3d colour-science
```

Then run

```sh
python colored_annotation_processor.py
```



## Example

Example of an original `.xyz` file:  ”A2_20220512_a.xyz“

![image-20240719190830198](E:\intern\3D-point-clouds-plant-phenotyping\image-20240719190830198.png)

Class Annotation

![image-20240719190128083](E:\intern\3D-point-clouds-plant-phenotyping\image-20240719190128083.png)

Instance Annotation

![image-20240719191352940](E:\intern\3D-point-clouds-plant-phenotyping\image-20240719191352940.png)





