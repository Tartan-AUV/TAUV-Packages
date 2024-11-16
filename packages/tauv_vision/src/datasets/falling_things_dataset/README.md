# FAT dataset overview

The "Falling Things" (FAT) dataset is a collection of synthetic images with ground truth annotations for research in object detection and 3D pose estimation.  The dataset combines object models with complex backgrounds of high graphical quality to yield photorealistic images with accurate 3D pose annotations for all objects in all images.  The dataset contains 61,500 unique annotated frames of 21 household objects from the [YCB dataset](http://www.ycbbenchmarks.com/).  Each frame consists of a stereo pair of RGBD images (i.e., RGB stereo images with ground truth depth for both cameras) and 3D poses, per-pixel semantic segmentation, and 2D/3D bounding box coordinates for all object instances.  The images show the objects falling onto different surfaces in three different scenes (living room, sun temple, and kite demo), captured by a [custom plug-in](https://github.com/NVIDIA/Dataset_Synthesizer) for Unreal Engine 4.  The dataset can be used for research in pose estimation, depth estimation from a single or stereo pair of cameras, semantic segmentation, and other applications within computer vision and robotics. 

## Paper

The paper describing the dataset can be found [here](https://arxiv.org/abs/1804.06534).

If you use this dataset, please cite as follows:

```
@INPROCEEDINGS{tremblay2018arx:fat,
  AUTHOR = "Jonathan Tremblay and Thang To and Stan Birchfield",
  TITLE = "Falling Things: {A} Synthetic Dataset for {3D} Object Detection and Pose Estimation",
  BOOKTITLE = "CVPR Workshop on Real World Challenges and New Benchmarks for Deep Learning in Robotic Vision",
  MONTH = jun,
  YEAR = 2018}
```

## License

The license can be found [here](http://research.nvidia.com/publication/2018-06_Falling-Things).

## Downloading

The dataset can be downloaded from [here](http://research.nvidia.com/publication/2018-06_Falling-Things).

## Visualizing

The YCB object models can be downloaded and visualized using our [NVDU tool](https://github.com/NVIDIA/Dataset_Utilities).  Note that the [publicly available YCB object models](http://www.ycbbenchmarks.com/) are not necessarily centered or aligned with respect to their coordinate system.  The NDDS tool downloads these models and transforms them so that the origin of the coordinate system is at the centroid of the point cloud, and the object is (approximately) aligned with the coordinate axes.  (The alignment is not perfect, since the objects are not pure geometric shapes but rather noisy scans.)  The NDDS tool can then be used to visualize the overlay of these models on the FAT images according to ground truth.

## Folder structure

When the dataset is extracted from the `.zip` file, the folder tree structure is as follows:
```
data
+-- single
|   +-- 002_master_chef_can_16k
|   |   +-- kitchen_X    
|   |   |   +-- _object_settings.json   
|   |   |   +-- _camera_settings.json
|   |   |   +-- XXXXXX.left.depth.png
|   |   |   +-- XXXXXX.left.json
|   |   |   +-- XXXXXX.left.seg.png
|   |   |   +-- XXXXXX.left.jpg
|   |   |   +-- XXXXXX.right.depth.png
|   |   |   +-- XXXXXX.right.json
|   |   |   +-- XXXXXX.right.seg.png
|   |   |   +-- XXXXXX.right.jpg
|   |   |   +-- ...
|   |   +-- ...
|   |   +-- kitedemo_X
|   |   +-- ...
|   |   +-- temple_X
|   |   +-- ...
|   +-- ...
+-- mixed
|   +-- kitchen_X
|   |   +-- _object_settings.json   
|   |   +-- _camera_settings.json
|   |   +-- XXXXXX.left.depth.png
|   |   +-- XXXXXX.left.json
|   |   +-- XXXXXX.left.seg.png
|   |   +-- XXXXXX.left.jpg
|   |   +-- XXXXXX.right.depth.png
|   |   +-- XXXXXX.right.json
|   |   +-- XXXXXX.right.seg.png
|   |   +-- XXXXXX.right.jpg
|   +-- ...
|   +-- kitedemo_X
|   +-- ...
|   +-- temple_X
|   +-- ...    
```

At the root level, there are two folders representing the two types of scenes: 
- `single` (single falling object), and 
- `mixed` (2 to 10 falling objects).

### Single 

For `single` each of the 21 object types has its own folder: 
```sh
002_master_chef_can_16k  008_pudding_box_16k      024_bowl_16k          051_large_clamp_16k
003_cracker_box_16k      009_gelatin_box_16k      025_mug_16k           052_extra_large_clamp_16k
004_sugar_box_16k        010_potted_meat_can_16k  035_power_drill_16k   061_foam_brick_16k
005_tomato_soup_can_16k  011_banana_16k           036_wood_block_16k
006_mustard_bottle_16k   019_pitcher_base_16k     037_scissors_16k
007_tuna_fish_can_16k    021_bleach_cleanser_16k  040_large_marker_16k

```
Within each folder there are 3 different scenes (`kitchen`, `kitedemo`, and `temple`) and 5 independent locations (0 through 4) within each scene: 
```
kitchen_0  kitchen_2  kitchen_4   kitedemo_1  kitedemo_3  temple_0  temple_2  temple_4
kitchen_1  kitchen_3  kitedemo_0  kitedemo_2  kitedemo_4  temple_1  temple_3
```
Each of these subfolders contains a dataset of 100 images of a particular object within a particular scene location, thus leading to `21 x 3 x 5 x 100 = 31500` image frames for `single`.

### Mixed

For `mixed` the images are organized by the 3 scenes and 5 locations within each scene, similar to above.
```
kitchen_0  kitchen_2  kitchen_4   kitedemo_1  kitedemo_3  temple_0  temple_2  temple_4
kitchen_1  kitchen_3  kitedemo_0  kitedemo_2  kitedemo_4  temple_1  temple_3
```
Each of these subfolders contains a dataset of 2000 images of objects within a particular scene location, thus leading to `3 x 5 x 2000 = 30000` image frames for `mixed`.

## File details

The details of the files are as follows.

### Setting files

In each data folder containing frames (*e.g.,* `data/single/002_master_chef_can_16k/kitchen_0/`), there are two files describing the exported scene:
* `_object_settings.json` includes information about the objects exported.  This includes 
  - the names of the exported object classes (`exported_object_classes`) 
  - details about the exported object classes (`exported_objects`), including
    - name of the class (`class`)
    - numerical class ID for semantic segmentation (`segmentation_class_id`).  For `mixed`, this number uniquely identifies the object class, but for `single`, this number is always 255, since there is just one object.
    - 4x4 Euclidean transformation (`fixed_model_transform`).  This transformation is applied to the original publicly-available YCB object in order to center and align it (translation values are in centimeters) with the coordinate system (see the discussion above on the NDDS tool).  Note that this is actually the transpose of the matrix.
    - dimensions of the 3D bounding cuboid along the XYZ axes (`cuboid_dimensions`)
* `_camera_settings.json` includes the intrinsics of both cameras (`camera_settings`).

The baseline between cameras is 6.0 cm. 

### Captured frame files

Each frame export contains 
- left / right RGB images (`XXXXXX.left.jpg`, `XXXXXX.right.jpg`)
- left / right depth images (`XXXXXX.left.depth.png`, `XXXXXX.right.depth.png`)
- left / right segmentation images (`XXXXXX.left.seg.png`, `XXXXXX.right.seg.png`)
- left / right annotation files (`XXXXXX.left.json`, `XXXXXX.right.json`), 

#### Image files

The image files are
- RGB images: JPEG-compressed images from the virtual cameras
- depth images:  Depth along the optical axis (in 0.1 mm increments)
- segmentation images:  Each pixel indicates the numerical ID of the object whose surface is visible at that pixel

#### Annotation files

Each annotation file includes
- XYZ position and orientation of the camera in the world coordinate frame (`camera_data`)
- for each object,
  - class name (`class`)
  - visibility, defined as the percentage of the object that is not occluded (`visibility`).  (0 means fully occluded whereas 1 means fully visible)
  - XYZ position (in centimeters) and orientation (`location` and `quaternion_xyzw`)
  - 4x4 transformation (redundant, can be computed from previous) (`pose_transform_permuted`)
  - 3D position of the centroid of the bounding cuboid (in centimeters) (`cuboid_centroid`)
  - 2D projection of the previous onto the image (in pixels) (`projected_cuboid_centroid`)
  - 2D bounding box of the object in the image (in pixels) (`bounding_box`)
  - 3D coordinates of the vertices of the 3D bounding cuboid (in centimeters) (`cuboid`)
  - 2D coordinates of the projection of the above (in pixels (`projected_cuboid`) 

*Note:*  Like the `fixed_model_transform`, the `pose_transform_permuted` is actually the transpose of the matrix.  Moreover, after transposing, the columns are permuted, and there is a sign flip (due to UE4's use of a lefthand coordinate system).  Specifically, if `A` is the matrix given by `pose_transform_permuted`, then actual transform is given by `A^T * P`, where `^T` denotes transpose, `*` denotes matrix multiplication, and the permutation matrix `P` is given by
```
    [ 0  0  1]
P = [ 1  0  0]
    [ 0 -1  0]
```

#### Coordinate frames

The indexes of the 3D bounding cuboid are in the following order:  
- `FrontTopRight` [0]
- `FrontTopLeft` [1]
- `FrontBottomLeft` [2]
- `FrontBottomRight` [3]
- `RearTopRight` [4]
- `RearTopLeft` [5]
- `RearBottomLeft` [6]
- `RearBottomRight` [7]

The XYZ coordinate frames are attached to each object as if the object were a camera facing the world through the front.  In other words, from the point of view of viewing the front from inside the object, the X axis points to the right, the Y axis points down, and the Z axis points forward toward the world.  Alternatively, from the point of view of viewing the front of the object from the outside (shown below), the X axis points left, the Y axis points down, and the Z axis points out of the page toward the viewer (right-hand coordinate system).

```
      4 +-----------------+ 5
       /     TOP         /|
      /                 / |
   0 +-----------------+ 1|
     |      FRONT      |  |
     |                 |  |
     |  x <--+         |  |
     |       |         |  |
     |       v         |  + 6
     |        y        | /
     |                 |/
   3 +-----------------+ 2
```

## Uncompressed RGB images

In the official FAT dataset above, the RGB images are lossy-compressed `.jpg` images.  If you would prefer to work with uncompressed `.png` images (that is, not lossy-compressed), you may download the alternative version [here](https://drive.google.com/open?id=16fJNufhOHay-SU-JcpQy9JWME47zFDzg) (137 GB).  
