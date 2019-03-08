/**
    fbp.hpp
    Filtered Back Projection Algorithm

    MIT License
    Copyright (c) 2019 Fran Piernas Diaz
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    Created using OpenCV 4.0.0

    @author Fran Piernas Diaz (fran.piernas@gmail.com)
    @version 1.1

    Based on Peter Toft: "The Radon Transform - Theory and Implementation",
    Ph.D. thesis. Department of Mathematical Modelling, Technical University of Denmark, June 1996.



    ************CHANGELOG************
    V1.1:
        Changed the way the point cloud is saved. Values are now floating point normalized
        from 0 to 255, instead of 0 to 255 but integer, as in V1.0.



    Usage to reconstruct a single sinogram, use this code:

        Mat sinogram=imread("sinogram.jpg",IMREAD_COLOR); //load the sinogram image "sinogram.jpg"
        Mat filtered_sinogram=filter_sinogram(sinogram); //filter the sinogram
        Mat reconstruction=iradon(filtered_sinogram,false); //perform back projection. Change false to true if sinogram is a full turn.
        renormalize255_frame(reconstruction); //normalize to 255
        imwrite("Reconstruction.jpg",reconstruction); //save reconstruction

    Usage to reconstuct a CT scan from a video, call this function:

        fbp(video,
            extra_frames,
            pointcloud_threshold,
            normalizeByframe,
            full_turn)

        where:  (string) video is the name of the CT video.
                (int) extra_frames is the number of interpolated frames between pair of frames. Try 1 and
                    increase if your results are noisy due to very few images available.
                (int) pointcloud_threshold is a 0-255 number wich will remove all points whose intensity
                    is less than this number when saving the point cloud.
                (bool) normalizeByframe, if true, all slices will be normalized to the maximum value
                    of that slice. If false, the object is normalized to its absolute maximum.
                (bool) full_turn, if true, the code assumes that the input CT video is a full turn.
                    Set it false if the input is half a turn.

        Follow any instruction on the terminal

    Usage to reconstuct a CT scan from a vector<Mat>, call this function:

        fbp(frames,
            extra_frames,
            pointcloud_threshold,
            normalizeByframe,
            full_turn)

        where frames is a vector<Mat> containing all pictures from 0 to 360 (or 180) degrees. All other
        parameters are the same as explained above.

        Follow any instruction on the terminal

    Special thanks to Nicolas De Francesco (nevermore78@gmail.com) for the help on image processing.
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <fstream>
#include <cmath>
#define K_PLUS 43
#define K_MINUS 45
#define K_ENTER 13
using namespace cv;
using namespace std;

void gamma_correction_frame(Mat& frame)
{
    unsigned int f,c;
    for(f=0;f<frame.size().height;f++)
    {
        for(c=0;c<frame.size().width;c++)
        {
            frame.at<Vec3f>(f,c)[0]=pow(frame.at<Vec3f>(f,c)[0],2.2);
            frame.at<Vec3f>(f,c)[1]=pow(frame.at<Vec3f>(f,c)[1],2.2);
            frame.at<Vec3f>(f,c)[2]=pow(frame.at<Vec3f>(f,c)[2],2.2);
        }
    }
    return;
}

void gamma_correction_frames(vector<Mat>& frames)
{
    unsigned int i;
    for(i=0;i<frames.size();i++) gamma_correction_frame(frames.at(i));
    return;
}

vector<Mat> video_to_array(string video)
{
    VideoCapture video_input(video.c_str());
    Mat frame;
    vector<Mat> input_array;
    do
    {
        video_input>>frame;
        if(frame.empty()) break;
        input_array.push_back(frame.clone());
    }while(!frame.empty());
    return input_array;
}

unsigned int begin_frame(vector<Mat>& frame_array)
{
    namedWindow("Select begin frame of CT.",WINDOW_NORMAL);
    cout<<"Select begin frame of CT. Navigate through frames with + or - and press enter to select."<<endl;
    unsigned int frame_index=0, key_pressed;
    do
    {
        imshow("Select begin frame of CT.",frame_array.at(frame_index));
        key_pressed=waitKey(0);
        if(key_pressed==K_PLUS&&frame_index<frame_array.size()-1) frame_index++;
        if(key_pressed==K_MINUS&&frame_index>0) frame_index--;
        cout<<"Moving to frame: "<<frame_index+1<<"/"<<frame_array.size()<<endl;
        if(key_pressed==K_ENTER) break;

    }while(true);
    destroyWindow("Select begin frame of CT.");
    return frame_index;
}

unsigned int end_frame(vector<Mat>& frame_array, unsigned int initial_frame)
{
    namedWindow("Select end frame of CT.",WINDOW_NORMAL);
    cout<<"Select end frame of CT. Navigate through frames with + or - and press enter to select."<<endl;
    unsigned int frame_index=initial_frame, key_pressed;
    do
    {
        imshow("Select end frame of CT.",frame_array.at(frame_index));
        key_pressed=waitKey(0);
        if(key_pressed==K_PLUS&&frame_index<frame_array.size()-1) frame_index++;
        if(key_pressed==K_MINUS&&frame_index>initial_frame) frame_index--;
        cout<<"Moving to frame: "<<frame_index+1<<"/"<<frame_array.size()<<endl;
        if(key_pressed==K_ENTER) break;
    }while(true);
    destroyWindow("Select end frame of CT.");
    return frame_index;
}

void cut_area(vector<Mat>& frames)
{
    cout<<"Select the object. Align the grid to the axis of rotation."<<endl;
    Rect2d r=selectROI("Select the object. Align the grid to the axis of rotation.",frames.at(0));
    destroyAllWindows();
    if(r.empty()) return;
    int i=0;
    for(i=0;i<frames.size();i++) frames.at(i)=frames.at(i)(r);
    return;
}

Mat make_sinogram(vector<Mat> frames, unsigned int height)
{
    Mat sinogram(frames.at(0).size().width,frames.size(),frames.at(0).type());
    int frame_number=0, j;
    for(frame_number=0;frame_number<frames.size();frame_number++)
    {
        for(j=0;j<frames.at(0).size().width;j++)
        {
            sinogram.at<float>(j,frame_number)=frames.at(frame_number).at<float>(height,j);
        }
    }
    return sinogram;
}

vector<Mat> make_sinograms(vector<Mat>& frames)
{
    unsigned int i;
    vector<Mat> sinograms;
    for(i=0;i<frames.at(0).size().height;i++) sinograms.push_back(make_sinogram(frames,i).clone());
    return sinograms;
}

void remove_frames(vector<Mat>& frames, unsigned int A, unsigned int B)
{
    if(B<frames.size())frames.erase(frames.begin()+B+1, frames.begin()+frames.size());
    if(A>0)frames.erase(frames.begin(), frames.begin()+A);
    return;
}

Mat iradon(Mat& sinogram, bool full_turn) //Sinogram must be a 32bit single channel grayscale image normalized 0-1
{
    Mat reconstruction(sinogram.size().height,sinogram.size().height,CV_32FC1);
    float delta_t;
    if(full_turn) delta_t=2.0*M_PI/sinogram.size().width;
    else delta_t=1.0*M_PI/sinogram.size().width;
    unsigned int t,f,c,rho;
    for(f=0;f<reconstruction.size().height;f++)
    {
        for(c=0;c<reconstruction.size().width;c++)
        {
            reconstruction.at<float>(f,c)=0;
            for(t=0;t<sinogram.size().width;t++)
            {
                rho=((f-0.5*sinogram.size().height)*cos(delta_t*t)+(c-0.5*sinogram.size().height)*sin(delta_t*t)+0.5*sinogram.size().height);
                if((rho>0)&&(rho<sinogram.size().height)) reconstruction.at<float>(f,c)+=sinogram.at<float>(rho,t);
            }
            if(reconstruction.at<float>(f,c)<0)reconstruction.at<float>(f,c)=0;
        }
    }
    rotate(reconstruction,reconstruction,ROTATE_90_CLOCKWISE);
    return reconstruction;
}

void convert_frame2RGB8(Mat& frame)
{
    frame.convertTo(frame,CV_8UC3,1);
    cvtColor(frame,frame,COLOR_GRAY2RGB);
    return;
}

void convert_frames2RGB8(vector<Mat>& frames)
{
    unsigned int i;
    for(i=0;i<frames.size();i++) convert_frame2RGB8(frames.at(i));
    return;
}

void renormalize255_frame(Mat& frame)
{
    unsigned int f,c;
    float maxm=0;
    for(f=0;f<frame.size().height;f++)
    {
        for(c=0;c<frame.size().width;c++)
        {
            if(frame.at<float>(f,c)>maxm)maxm=frame.at<float>(f,c);
        }
    }

    for(f=0;f<frame.size().height;f++)
    {
        for(c=0;c<frame.size().width;c++)
        {
            frame.at<float>(f,c)=frame.at<float>(f,c)*255.0/maxm;
        }
    }
    return;
}

void renormalize255_frame(Mat& frame, float maxm)
{
    unsigned int f,c;
    for(f=0;f<frame.size().height;f++)
    {
        for(c=0;c<frame.size().width;c++)
        {
            frame.at<float>(f,c)=frame.at<float>(f,c)*255.0/maxm;
        }
    }
    return;
}

void renormalize255_frame_by_frame(vector<Mat>& frames)
{
    unsigned int i;
    for(i=0;i<frames.size();i++) renormalize255_frame(frames.at(i));
    return;
}

void renormalize255_frames(vector<Mat>& frames)
{
    unsigned int f,c,i;
    float maxm=0;
    for(i=0;i<frames.size();i++)
    {
        for(f=0;f<frames.at(i).size().height;f++)
        {
            for(c=0;c<frames.at(i).size().width;c++)
            {
                if(frames.at(i).at<float>(f,c)>maxm)maxm=frames.at(i).at<float>(f,c);
            }
        }
    }
    for(i=0;i<frames.size();i++)renormalize255_frame(frames.at(i),maxm);
    return;
}

void convert_frame2bw(Mat& frame)
{
    unsigned int f,c;
    Mat converted(frame.size().height,frame.size().width,CV_32FC1);
    for(f=0;f<frame.size().height;f++)
    {
        for(c=0;c<frame.size().width;c++)
        {
            converted.at<float>(f,c)=1.0/3.0*(frame.at<Vec3f>(f,c)[0]+frame.at<Vec3f>(f,c)[1]+frame.at<Vec3f>(f,c)[2]);
            //converted.at<float>(f,c)=frame.at<Vec3f>(f,c)[1];
        }
    }
    frame=converted.clone();
    return;
}

void convert_frames2bw(vector<Mat>& frames)
{
    unsigned int i;
    for(i=0;i<frames.size();i++) convert_frame2bw(frames.at(i));
    return;
}

void convert_frame2f(Mat& frame)
{
    frame.convertTo(frame,CV_32FC3,1.0/255.0);
    return;
}

void convert_frames2f(vector<Mat>& frames)
{
    unsigned int i;
    for(i=0;i<frames.size();i++)convert_frame2f(frames.at(i));
    return;
}

Mat filter_sinogram(Mat& sinogram)
{
    Mat filtered_sinogram;
    transpose(sinogram,filtered_sinogram);
    if(filtered_sinogram.type()==CV_8UC3) //Convert to gray scale and 32bit if the input is a 8bit RGB image
    {
        cout<<"Converting to 32bit grayscale..."<<endl;
        convert_frame2f(filtered_sinogram);
        convert_frame2bw(filtered_sinogram);
    }
    Mat dft_sinogram[2]={filtered_sinogram,Mat::zeros(filtered_sinogram.size(),CV_32F)};
    Mat dftReady;
    merge(dft_sinogram,2,dftReady);
    dft(dftReady,dftReady,DFT_ROWS|DFT_COMPLEX_OUTPUT,0);
    split(dftReady,dft_sinogram);
    unsigned int f,c;
    for(f=0;f<dft_sinogram[0].size().height;f++)
    {
        for(c=0;c<dft_sinogram[0].size().width;c++)
        {
            //Sine Filter
            dft_sinogram[0].at<float>(f,c)*=(1.0/(2.0*M_PI))*1.0*abs(sin(1.0*M_PI*(c)/dft_sinogram[0].size().width));
            dft_sinogram[1].at<float>(f,c)*=(1.0/(2.0*M_PI))*1.0*abs(sin(1.0*M_PI*(c)/dft_sinogram[0].size().width));
        }
    }
    merge(dft_sinogram,2,dftReady);
    dft(dftReady,filtered_sinogram,DFT_INVERSE|DFT_ROWS|DFT_REAL_OUTPUT,0);
    transpose(filtered_sinogram,filtered_sinogram);
    return filtered_sinogram;
}

void filter_all_sinograms(vector<Mat>& sinograms)
{
    unsigned int i;
    for(i=0;i<sinograms.size();i++) sinograms.at(i)=filter_sinogram(sinograms.at(i)).clone();
    return;
}

vector<Mat> video_iradon(vector<Mat>& sinograms, bool full_turn)
{
    vector<Mat> slices;
    unsigned int i, prev_perc;
    for(i=0;i<sinograms.size();i++)
    {
        slices.push_back(iradon(sinograms.at(i),full_turn).clone());
        if((int)(100.0*i/sinograms.size()-prev_perc)>0) cout<<(int)100.0*i/sinograms.size()<<"%"<<endl;
        prev_perc=100.0*i/sinograms.size();
    }
    return slices;
}

void save_scan(vector<Mat>& slices, unsigned int i_threshold)
{
    ofstream save_slices;
    save_slices.open("slices.xyz");
    save_slices.precision(6);
    unsigned int slice, f, c;
    for(slice=0;slice<slices.size();slice++)
    {
        for(f=0;f<slices.at(slice).size().height;f++)
        {
            for(c=0;c<slices.at(slice).size().width;c++)
            {
                if(slices.at(slice).at<float>(f,c)>i_threshold)
                {
                    save_slices<<f<<" "<<c<<" "<<slice<<" ";
                    save_slices<<scientific<<slices.at(slice).at<float>(f,c);
                    save_slices<<endl;
                }
            }
        }
    }
    save_slices.close();
    return;
}

float frames_get_bg_intensity(Mat& frame, Mat& display_image)
{
    float I0=0.0;
    cout<<"Select a background area."<<endl;
    Rect2d r=selectROI("Select a background area.",display_image);
    destroyAllWindows();
    if(r.empty())
    {
        cout<<"Background intensity set to 1"<<endl;
        return 1.0;
    }
    Mat selected=frame(r);
    unsigned int f,c;
    for(f=0;f<selected.size().height;f++)
    {
        for(c=0;c<selected.size().width;c++)
        {
            I0+=selected.at<float>(f,c);
        }
    }
    I0*=1.0/(selected.size().width*selected.size().height);
    cout<<"Background intensity set to "<<I0<<endl;
    return I0;
}

void frames_get_transmitance(vector<Mat>& frames, float I0)
{
    unsigned int i,f,c;
    for(i=0;i<frames.size();i++)
    {
        for(f=0;f<frames.at(i).size().height;f++)
        {
            for(c=0;c<frames.at(i).size().width;c++)
            {
                if(frames.at(i).at<float>(f,c)<I0&&frames.at(i).at<float>(f,c)>0.0)frames.at(i).at<float>(f,c)=-1.0*log10(frames.at(i).at<float>(f,c)/I0);
                else if(frames.at(i).at<float>(f,c)>I0) frames.at(i).at<float>(f,c)=0.0;
                else if(frames.at(i).at<float>(f,c)==0.0) frames.at(i).at<float>(f,c)=-1.0*log10((1.0/255.0)/I0);
            }
        }
    }
    return;
}

Mat create_interpolation_img(Mat& img1, Mat& img2, unsigned int extra_frames, unsigned int extra_frame_number) //expected CV_32FC1 images
{
    unsigned int i,f,c;
    float slope=0;
    Mat interpolated_frame(img1.size().height,img1.size().width,img1.type());
    for(f=0;f<img1.size().height;f++)
    {
        for(c=0;c<img1.size().width;c++)
        {
            slope=(1.0*(img2.at<float>(f,c)-img1.at<float>(f,c)))/(extra_frames+1);
            interpolated_frame.at<float>(f,c)=1.0*extra_frame_number*slope+img1.at<float>(f,c);
        }
    }
    return interpolated_frame;
}

vector<Mat> interpolate(vector<Mat>& frames, unsigned int extra_frames)
{
    unsigned int i, j;
    vector<Mat> interpolated;
    for(i=0;i<frames.size();i++)
    {
        interpolated.push_back(frames.at(i).clone());
        for(j=1;j<=extra_frames;j++)
        {
            if(i!=frames.size()-1)interpolated.push_back(create_interpolation_img(frames.at(i),frames.at(i+1),extra_frames,j).clone());
        }
    }
    return interpolated;
}

void save_video(vector<Mat>& frames, string name)
{
    VideoWriter video_reconstruction(name.c_str(),VideoWriter::fourcc('M','J','P','G'),25,frames.at(0).size(),true);
    unsigned int i;
    for(i=0;i<frames.size();i++) video_reconstruction.write(frames.at(i));
    video_reconstruction.release();
    return;
}

void fbp(vector<Mat>& frames, unsigned int extra_frames, unsigned int pointcloud_threshold, bool normalizeByframe, bool full_turn)
{
    vector<Mat> sinograms, slices;
    cut_area(frames);
    Mat display_image=frames.at(0).clone();
    convert_frames2f(frames);
    gamma_correction_frames(frames);
    convert_frames2bw(frames);
    if(extra_frames>0)
    {
        cout<<"Performing interpolation..."<<endl;
        frames=interpolate(frames,extra_frames);
        cout<<"Number of frames: "<<frames.size()<<endl;
    }
    frames_get_transmitance(frames,frames_get_bg_intensity(frames.at(0),display_image));

    cout<<"Building sinograms..."<<endl;
    sinograms=make_sinograms(frames);

    cout<<"Filtering sinograms..."<<endl;
    filter_all_sinograms(sinograms);

    cout<<"Performing Filtered Back Projection..."<<endl;
    slices=video_iradon(sinograms,full_turn);

    cout<<"Normalizing..."<<endl;
    if(!normalizeByframe)renormalize255_frames(slices);
    else renormalize255_frame_by_frame(slices);

    cout<<"Saving point cloud..."<<endl;
    save_scan(slices,pointcloud_threshold);

    cout<<"Converting to RGB..."<<endl;
    convert_frames2RGB8(slices);

    cout<<"Saving video..."<<endl;
    save_video(slices,"Reconstruction.avi");

    cout<<"Finished."<<endl;
    return;
}

void fbp(string input, unsigned int extra_frames, unsigned int pointcloud_threshold, bool normalizeByframe, bool full_turn)
{
    unsigned int A_frame, B_frame;
    vector<Mat> frames, sinograms, slices;
    frames=video_to_array(input.c_str());
    A_frame=begin_frame(frames);
    B_frame=end_frame(frames,A_frame);
    remove_frames(frames,A_frame,B_frame);
    cout<<"Number of frames: "<<frames.size()<<endl;
    cut_area(frames);
    Mat display_image=frames.at(0).clone();
    convert_frames2f(frames);
    gamma_correction_frames(frames);
    convert_frames2bw(frames);
    if(extra_frames>0)
    {
        cout<<"Performing interpolation..."<<endl;
        frames=interpolate(frames,extra_frames);
        cout<<"Number of frames: "<<frames.size()<<endl;
    }
    frames_get_transmitance(frames,frames_get_bg_intensity(frames.at(0),display_image));

    cout<<"Building sinograms..."<<endl;
    sinograms=make_sinograms(frames);

    cout<<"Filtering sinograms..."<<endl;
    filter_all_sinograms(sinograms);

    cout<<"Performing Filtered Back Projection..."<<endl;
    slices=video_iradon(sinograms,full_turn);

    cout<<"Normalizing..."<<endl;
    if(!normalizeByframe)renormalize255_frames(slices);
    else renormalize255_frame_by_frame(slices);

    cout<<"Saving point cloud..."<<endl;
    save_scan(slices,pointcloud_threshold);

    cout<<"Converting to RGB..."<<endl;
    convert_frames2RGB8(slices);

    cout<<"Saving video..."<<endl;
    save_video(slices,"Reconstruction.avi");

    cout<<"Finished."<<endl;
    return;
}





