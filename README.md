# 3D-point-clouds-plant-phenotyping
Automated segmentation and annotation of 3D point clouds for plant phenotyping



## File Structure
```
.
├── colored_annotation_processor.py   # Code for visualizing class and instance annotations                         
├── sample_output                     # Sample visualisation of class and instance annotation for ”A2_20220512_a.xyz“
└── README.md                         # This README file
```

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

![image-20240719190830198](image-20240719190830198.png)

Class Annotation

![image-20240719190128083](image-20240719190128083.png)

Instance Annotation

![image-20240719191352940](image-20240719191352940.png)

## PointNet++ Visualisation
Avg IoU reaches over 70%
![image](https://github.com/user-attachments/assets/baf4d27f-8726-460d-ae8a-df1594a949cd)

![image](https://github.com/user-attachments/assets/75fc362f-1da9-447d-8e22-9020e5a51082)

![image](https://github.com/user-attachments/assets/6d7f69fd-4606-46d8-a4d1-e64c3832324b)




