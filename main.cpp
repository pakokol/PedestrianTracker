#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/video.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <vector>
#include <opencv2/bgsegm.hpp>
#include <opencv2/tracking.hpp>
#include <cmath>

void filterContoures(std::vector<std::vector<cv::Point>> &contours, double areaTresholdLowerBound, double areaTrehsoldUpperBound)
{
    size_t k = 0;
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area > areaTresholdLowerBound && area < areaTrehsoldUpperBound) {
            contours[k++] = contours[i];
        }
    }
    contours.resize(k);
}

std::vector<cv::Point2f> getGoodCorners(cv::Mat gray, std::vector<cv::Rect> boundingRects, int maxNumberOfCornersPerObject)
{
    std::vector<cv::Point2f> corners;
    for (size_t i = 0; i < boundingRects.size(); ++i) {
        cv::Mat mask(gray.size(), CV_8UC1, cv::Scalar(0));;
        mask(boundingRects[i]).setTo(cv::Scalar(255));

        std::vector<cv::Point2f> localROICorners;
        cv::goodFeaturesToTrack(gray, localROICorners, maxNumberOfCornersPerObject, 0.01, 10, mask, 3, 0, 0.04);
        corners.insert(corners.end(), localROICorners.begin(), localROICorners.end());
    }

    return corners;
}

std::vector<cv::Rect> getBoundingRects(std::vector<std::vector<cv::Point>> contours)
{
    std::vector<cv::Rect> boundingRects(contours.size());
    for (size_t i = 0; i < contours.size(); ++i) {
        boundingRects[i] = cv::boundingRect(contours[i]);
    }

    return boundingRects;
}

int main(/*int argc, char *argv[]*/)
{
    cv::VideoCapture cam;
    cam.open("roadWalk.avi");
    if (!cam.isOpened()) {
        std::cout << "Can't find the avi file!" << std::endl;
        return -1;
    }

    cv::Mat gray, prevGray, image, frame;
    std::vector<cv::Point2f> corners[2];
    //Thid tresholds need to be set experimentaly
    const double CONTOUR_AREA_TRESHOLD_LOWERBOUND = 250;
    const double CONTOUR_AREA_TRESHOLD_UPPERBOUND = 2500;
    const double VELOCITY_TRESHOLD_LOWERBOUND = 1.0;
    const double VELOCITY_TRESHOLD_UPPERBOUND = 8.0;
    const int MAX_CORNERS_AT_OBJECT = 5;
    //Frame rate
    const double fps = cam.get(cv::CAP_PROP_FPS);
    const double deltaT = 1.0 / fps; //Time between frames
    const double MAX_VELOCITY = 80;

    const cv::Size lkWindowSize = cv::Size(10,10);
    cv::Mat foreGround;
    cv::Ptr<cv::BackgroundSubtractorMOG2> bg = cv::createBackgroundSubtractorMOG2();
    bg->setHistory(500);
    bg->setVarThreshold(10);
    bg->setDetectShadows(true);

    cv::Mat morphKernel = cv::getStructuringElement(2, cv::Size(5,5));
    std::vector<cv::Mat> velocityMaps;
    std::vector<cv::Mat> orientationMaps;
    while(1) {
        cam >> frame;
        if (frame.size().height == 0 || frame.size().width == 0) {
            break;
        }
        frame.copyTo(image);
        cv::cvtColor(image, gray, CV_BGR2GRAY);

        //Background substraction using Gaussian mixture models
        bg->apply(image, foreGround, -1);
        cv::erode(foreGround, foreGround, morphKernel);
        cv::dilate(foreGround, foreGround, morphKernel);
        cv::threshold(foreGround, foreGround, 200, 255, 3);
        cv::imshow("Foreground", foreGround);

        std::vector<std::vector<cv::Point>> contours;
        cv::Mat contoureImage(image.size(), CV_8UC3, cv::Scalar(0));
        cv::findContours(foreGround, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        filterContoures(contours, CONTOUR_AREA_TRESHOLD_LOWERBOUND, CONTOUR_AREA_TRESHOLD_UPPERBOUND);
        cv::drawContours(contoureImage, contours, -1, cv::Scalar(0.0, 0.0, 255.0));
        cv::imshow("contoure image", contoureImage);

        std::vector<cv::Rect> boundingRects = getBoundingRects(contours);
        const std::string textToDisplay = "Number of pedestrian: " + std::to_string(boundingRects.size());
        cv::putText(image, textToDisplay, cv::Point(10, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255));

        //Optical flow calculation
        std::vector<uchar> status;
        std::vector<float> err;
        if (prevGray.empty()) {
            gray.copyTo(prevGray);
        }

        corners[1] = getGoodCorners(gray, boundingRects, MAX_CORNERS_AT_OBJECT);

        std::vector<double> sumVelocityInRect(boundingRects.size(), 0.0);
        std::vector<cv::Point> sumDisplacmentInRect(boundingRects.size(), cv::Point(0,0));
        std::vector<int> numberOfVectorsInRect(boundingRects.size(), 0);
        if (corners[1].size() != 0) {
            cv::calcOpticalFlowPyrLK(gray, prevGray, corners[1], corners[0], status, err, lkWindowSize);

            size_t i, k;
            for( i = k = 0; i < corners[1].size(); i++ )
            {
                if( status[i] ) {
                    cv::Point displasment = corners[0][i] - corners[1][i];
                    const double displacmentVecLenght = sqrt(pow(displasment.x, 2)+pow(displasment.y, 2));
                    if (displacmentVecLenght < VELOCITY_TRESHOLD_LOWERBOUND || displacmentVecLenght > VELOCITY_TRESHOLD_UPPERBOUND)
                        continue;

                    for (size_t j = 0; j < boundingRects.size(); ++j) {
                        if (boundingRects[j].contains(corners[1][i])) {
                            sumVelocityInRect[j] += (displacmentVecLenght / deltaT);
                            numberOfVectorsInRect[j] += 1;
                            sumDisplacmentInRect[j] += displasment;
                            break;
                        }
                    }
                    corners[1][k++] = corners[1][i];
                    cv::circle( image, corners[0][i], 2, cv::Scalar(0,0,255), 3, 8);
                    cv::line(image, corners[1][i], corners[0][i], cv::Scalar(0, 0, 255), 2);
                }
            }
            corners[1].resize(k);
        }

        //Calculate the velocity and orientation maps
        cv::Mat velocityMap(image.size(), CV_8UC3, cv::Scalar(0.0, 0.0, 0.0));
        cv::Mat orientationMap(image.size(), CV_8UC3, cv::Scalar(0.0, 0.0, 0.0));
        cv::cvtColor(velocityMap, velocityMap, CV_BGR2HSV);
        cv::cvtColor(orientationMap, orientationMap, CV_BGR2HSV);

        for (size_t i = 0; i < boundingRects.size(); ++i) {
            if (numberOfVectorsInRect[i] > 0) {
                const double vel = round((sumVelocityInRect[i] / numberOfVectorsInRect[i]));
                const double normVel = vel / MAX_VELOCITY;
                cv::drawContours(velocityMap, contours, i, cv::Scalar(240.0-(normVel*240.0), 255, 255), CV_FILLED);

                const cv::Point avgDisplacment = sumDisplacmentInRect[i] / numberOfVectorsInRect[i];
                std::vector<float> x(1), y(1), m, r;
                x[0] = avgDisplacment.x;
                y[0] = avgDisplacment.y;
                cv::cartToPolar(x, y, m, r, true);
                cv::drawContours(orientationMap, contours, i, cv::Scalar(r[0], 255.0, m[0]), CV_FILLED);
            }
            cv::rectangle(image, boundingRects[i], cv::Scalar(0.0, 255.0, 0.0));
        }

        cv::cvtColor(velocityMap, velocityMap, CV_HSV2BGR);
        cv::cvtColor(orientationMap, orientationMap, CV_HSV2BGR);

        velocityMaps.push_back(velocityMap);
        orientationMaps.push_back(orientationMap);

        cv::imshow("Orientation", orientationMap);
        cv::imshow("Optical flow", image);
        cv::swap(prevGray, gray);
        std::swap(corners[0], corners[1]);

        unsigned char key = cv::waitKey(50);
        if (key == 'E')
            break;
    }

    cv::Mat averageVelocity(image.size(), CV_32FC3, cv::Scalar(0.0));
    for (const auto& v : velocityMaps) {
        cv::Mat temp;
        v.convertTo(temp, CV_32FC3);
        averageVelocity += temp;
    }

    cv::Mat averageOrientation(image.size(), CV_32FC3, cv::Scalar(0.0));
    for (const auto& o : orientationMaps) {
        cv::Mat temp;
        o.convertTo(temp, CV_32FC3);
        averageOrientation += temp;
    }

    averageVelocity = averageVelocity / velocityMaps.size();
    cv::normalize(averageVelocity, averageVelocity, 0, 1, cv::NORM_MINMAX);

    averageOrientation = averageOrientation / orientationMaps.size();
    cv::normalize(averageOrientation, averageOrientation, 0, 1, cv::NORM_MINMAX);

    cv::imshow("Average velocity at pixels", averageVelocity);
    cv::imshow("Average orientation at pixels", averageOrientation);
    cv::waitKey();

    return 0;
}
