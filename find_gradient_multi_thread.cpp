#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <thread>
#include <mutex>

int cell_ = 4;
int pattern_gradient_ = 3;
std::vector<cv::KeyPoint> hg_keys_;
static const int num_threads = 8;
int repeat_ = cell_ * cell_ / num_threads;
std::mutex mtx;

void ExtractHighGradient(int flag, int cell_part, const cv::Mat &im){
    //cell row and cell columns
    int rows = im.rows;
    int cols = im.cols;
    int c_row = im.rows/cell_;
    int c_col = im.cols/cell_;
    if(flag == 0){
        //no consider monocular case for now
    }
    else{
        int k = cell_part/cell_, l = cell_part%cell_;
        int count = 0;
        //divided image into cell by cell parts, in each part we'll extract high gradient points in a 7 by 7 pattern and calcuate one NID
        while(k<cell_){
            for(int i = c_row*k; i < c_row*(k+1); i += (2*pattern_gradient_+1)){
                if(i == 0 || i >= (rows - (2*pattern_gradient_ + 1)))
                    continue;
                for(int j = c_col*l; j < c_col*(l+1); j += (2*pattern_gradient_+1)){
                    if(j == 0 || j>=(cols - (2*pattern_gradient_ + 1)) )
                        continue;
                    //find the pixel that has max gradient in the `2*pattern_gradient+1` by ``2*pattern_gradient+1` pattern
                    float max_gradient_norm = 0.0;
                    float gradient_norm = 0.0;
                    cv::KeyPoint lmgp; //local_max_gradient_point
                    for(int y = i; y < i+2*pattern_gradient_+1; y++){
                        for(int x = j; x< j+2*pattern_gradient_+1; x++){
                                Eigen::Vector2d gradient(
                                    im.ptr<uchar>(y+1)[x] - im.ptr<uchar>(y-1)[x],
                                    im.ptr<uchar>(y)[x+1] - im.ptr<uchar>(y)[x-1]
                                );
                            gradient_norm = gradient.norm();
                            if(gradient_norm > max_gradient_norm){
                                lmgp.pt.x = x;
                                lmgp.pt.y = y;
                                max_gradient_norm = gradient_norm;
                            }
                        }
                    }
                    mtx.lock();//avoid push_back conflix problem
                    hg_keys_.push_back(lmgp);
                    mtx.unlock();
                }
            }
            k += repeat_;
        }
    }
}

//I use 8 threads and the speed is only 2 or 3 times faster = = .....
int main(int argc, char* argv[]){
    cv::Mat test;
    std::string add("../demo.png");
    if(argc != 2)
        std::cout<<"use default address "<<add.c_str()<<std::endl;
    else{
        add.clear();
        add.append(argv[1]);
    }
    
    test = cv::imread(add, 0);//0 for gray and 1 for rgb
    cv::Mat im_with_keypoints;

    std::thread t[num_threads];

    double extract_time = (double)cv::getTickCount();
    for (int i = 0; i < num_threads; ++i) {
        t[i] = std::thread(ExtractHighGradient, 1, i, test);
    }
    for (int i = 0; i < num_threads; ++i) {
        t[i].join();
    }
    //ExtractHighGradient(1, test);
    extract_time = ((double)cv::getTickCount() - extract_time)/cv::getTickFrequency();
    std::cout<<"extract time is "<<extract_time<<std::endl;

    std::cout<<"point size "<<hg_keys_.size()<<std::endl;

    cv::drawKeypoints(test, hg_keys_, im_with_keypoints);

    cv::namedWindow("show key points", cv::WINDOW_AUTOSIZE);
    cv::imshow("show key points", im_with_keypoints);

    cv::waitKey(0);
}
