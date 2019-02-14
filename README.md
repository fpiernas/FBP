# FBP
Filtered Back Projection C++ by Fran Piernas Diaz

**Requirements**

OpenCV must be installed. To compile, link all OpenCV libraries.

**How to include it to your project**

Place "fbp.hpp" to the same directory as your source code and include it:
#include "fbp.hpp"

**How to use the code to reconstruct a single sinogram**

Use this code:

    Mat sinogram=imread("sinogram.jpg",IMREAD_COLOR); //load the sinogram image "sinogram.jpg"
    Mat filtered_sinogram=filter_sinogram(sinogram); //filter the sinogram
    Mat reconstruction=iradon(filtered_sinogram,false); //perform back projection. Change false to true if sinogram is a full turn
    renormalize255_frame(reconstruction); //normalize to 255
    imwrite("Reconstruction.jpg",reconstruction); //save reconstruction
    
**How to get data to use with the program**

Just record a video of the object spinning through X-Rays. You don't have to cut the video so the file consists of a loop of the object, the code will later ask you to cut it.
    
**How to use the code to reconstruct a 3D object from a CT scan video**

Call this function:

    fbp(video, extra_frames, pointcloud_threshold, normalizeByframe, full_turn);
    
Where:
  * video is a string containing the video name (example, led_test.mp4).
  * extra_frames is an integer greater or equal to 0. If your data consists of very few images (60 or less), try 1 or 2
      extra frames and the code will interpolate and increase your number of images available, reducing noise.
  * pointcloud_threshold is a 0-255 number wich will remove all reconstruction pixels whose intensity is less than this number.                             
      This only affects the point cloud file saved.
  * normalizeByframe is a boolean. If true, all slices will be normalized in a way that the brightest pixel of the slice gets 255.
      If false, all pixels of all slices will be normalized so that the brightest pixel of the entire object gets 255.
  * full_turn is a boolean. If true, the code assumes that the input data consists of a full turn of the object (so that the
      inverse Radon transform is calculated from 0º to 360º). If false, it's calculated from 0º to 180º (half turn).
      
**How to use the code to reconstruct a 3D object from a vector\<Mat\> object**     

Call this function:

    fbp(video, extra_frames, pointcloud_threshold, normalizeByframe, full_turn);
    
Where video is a vector\<Mat\> object. All other parameters are the same as explained above.

**What the code does**

The reconstruction from a sinogram image is straight forward. However, the reconstruction from a video requires more steps:

*The code will load the video and ask the user to select the exact frame of the video where the CT scan begins, this frame gets the angle 0º when performing the Radon transform. Navigate through frames with keys "+", "-" and "Enter". Don't use the numpad.

*The code will ask the user to select the end frame (angle 360º or 180º depending if full_turn is true or false). Press "+" until the object looks the same as in the begining frame you selected earlier (so it has completed a full turn) or press it until the object has completed half a turn.

*The code ask the user to select the object. It's critical to do this in a way that the central vertical line of the grid matches the axis of rotation.

*The code converts all frames to float, perform a gamma correction and converts to grayscale. If extra_frames is greater than 0, the code interpolates new images.

*The code ask the user to select an area of the background of the images where there is no object. It computes the mean intensity of the background for the next step. You can select no area and this background intensity is set to 1.

*The code calculates absorptances by dividing the intensities of each pixel by the mean background calculated before, then applies a -1.0*log10( ) to this value.

*The code makes the sinograms and filters them with an abs(sine()) filter

*The code gets the inverse Radon transform for all sinograms.

*Depending if normalizeBy frame is true or false, the code normalizes one way or another.

*The code converts all reconstructed slices to 8bit RGB, saves the video "Reconstruction.avi" and saves the point cloud with the desired pointcloud_threshold set.
