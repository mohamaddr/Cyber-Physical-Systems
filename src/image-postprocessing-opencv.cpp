/*
 * Copyright (C) 2019  Christian Berger
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the `Free Software Foundation, either version 3 of the License, or
 * (at your option) any later versio
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESjlkS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "cluon-complete.hpp"
#include "opendlv-standard-message-set.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "opencv2/features2d.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <cv.h>

using namespace cv;
using namespace std;

bool stopSignDetected(Mat img);
void detectYellowSign(Mat img);
bool cmpf(float A, float B);

bool atIntersection = false; // to know if the car has reached the intersection or not
bool turnRight = true;
bool toMoveForward = true;

RNG rng(12345);
float area = 0;

int32_t main(int32_t argc, char **argv)
{
    int32_t retCode{1};
    auto commandlineArguments = cluon::getCommandlineArguments(argc, argv);
    if ((0 == commandlineArguments.count("cid")) ||
        (0 == commandlineArguments.count("name")) ||
        (0 == commandlineArguments.count("width")) ||
        (0 == commandlineArguments.count("height")) ||
        (0 == commandlineArguments.count("stepsForward")))
    {
        std::cerr << argv[0] << " attaches to a shared memory area containing an ARGB image." << std::endl;
        std::cerr << "Usage:   " << argv[0] << " --cid=<OD4 session> --name=<name of shared memory area> [--verbose]" << std::endl;
        std::cerr << "         --cid:    CID of the OD4Session to send and receive messages" << std::endl;
        std::cerr << "         --name:   name of the shared memory area to attach" << std::endl;
        std::cerr << "         --width:  width of the frame" << std::endl;
        std::cerr << "         --height: height of the frame" << std::endl;
        std::cerr << "         --stepsForward: steps to move Forward on intersection" << std::endl;
        std::cerr << "Example: " << argv[0] << " --cid=112 --name=img.i420 --width=640 --height=480" << std::endl;
    }
    else
    {
        const std::string NAME{commandlineArguments["name"]};
        const uint32_t WIDTH{static_cast<uint32_t>(std::stoi(commandlineArguments["width"]))};
        const uint32_t HEIGHT{static_cast<uint32_t>(std::stoi(commandlineArguments["height"]))};
        const int STEPS_FORWARD{static_cast<int>(std::stoi(commandlineArguments["stepsForward"]))};
        const float RIGHT_TURN{(float)-0.38};
        const float LEFT_TURN{(float)0.20};
        int16_t delay = 0;
        float SPEED = 0;

        cout << "enter a speed" << endl;
        cin >> SPEED;
        cout << "enter a delay" << endl;
        cin >> delay;

        cluon::OD4Session od4{static_cast<uint16_t>(std::stoi(commandlineArguments["cid"]))};

        float tempDistReading{0.0f};
        auto onDistanceReading{[&tempDistReading](cluon::data::Envelope &&envelope)
                               // &<variables> will be captured by reference (instead of value only)
                               {
                                   MyTestMessage1 msg = cluon::extractMessage<MyTestMessage1>(std::move(envelope));
                                   tempDistReading = msg.myValue();
                                   // Local variables are not available outside the lambda function
                                   // Corresponds to odvd message set
                                   std::cout << "Received DistanceReading message  " << tempDistReading << std::endl;
                               }};

        od4.dataTrigger(2001, onDistanceReading);

        // Attach to the shared memory.
        std::unique_ptr<cluon::SharedMemory> sharedMemory{new cluon::SharedMemory{NAME}};
        if (sharedMemory && sharedMemory->valid())
        {
            std::clog << argv[0] << ": Attached to shared memory '" << sharedMemory->name() << " (" << sharedMemory->size() << " bytes)." << std::endl;

            // Interface to a running OpenDaVINCI session; here, you can send and receive messages.

            // Create an OpenCV image header using the data in the shared memory.
            IplImage *iplimage{nullptr};
            CvSize size;
            size.width = WIDTH;
            size.height = HEIGHT;

            iplimage = cvCreateImageHeader(size, IPL_DEPTH_8U, 4 /* four channels: ARGB */);
            sharedMemory->lock();
            {
                sleep(3);
                iplimage->imageData = sharedMemory->data();
                iplimage->imageDataOrigin = iplimage->imageData;
            }
            sharedMemory->unlock();

            int left = 0;    // 0 for initializing, -1 if there is no car on the left, 1 if there is a left car
            int maxCar = -1; // numbers of cars at the intersection
            // Max boundaries to detect the car on the left
            float maxLeftX = 100;
            float maxLeftY = 210;
            // used to assign the coordinates of corners around the found contour to be compared with maxLeftX and maxLeftY later on.
            float leftCornerX;
            float leftCornerY;
            //number of frames that have passed
            int frameCount = 0;

            while (od4.isRunning())
            {
                cv::Mat img;

                // Wait for a notification of a new frame.
                sharedMemory->wait();

                // Lock the shared memory.
                sharedMemory->lock();
                {
                    // Copy image into cvMat structure.
                    // Be aware of that any code between lock/unlock is blocking
                    // the camera to provide the next frame. Thus, any
                    // computationally heavy algorithms should be placed outside
                    // lock/unlock.
                    cv::Mat wrapped(HEIGHT, WIDTH, CV_8UC4, sharedMemory->data());
                    img = wrapped.clone();
                }

                sharedMemory->unlock();

                // TODO: Do something with the frame.

                cv::Mat imgHSV;
                cv::Mat green;

                cvtColor(img, imgHSV, cv::COLOR_BGR2HSV);
                imgHSV.convertTo(imgHSV, CV_8U);

                cvtColor(img, green, cv::COLOR_BGR2HSV);
                green.convertTo(green, CV_8U);
                // detecting blue color, which represents the leading car
                cv::Mat imgColorSpace;
                cv::inRange(imgHSV, cv::Scalar(90, 61, 133), cv::Scalar(115, 225, 255), imgColorSpace);
                // detecting green color, which represents the cars on the intersection
                cv::Mat imgColorSpaceGreen;
                cv::inRange(green, cv::Scalar(34, 50, 50), cv::Scalar(80, 185, 200), imgColorSpaceGreen);

                //Finding blue contours
                vector<vector<Point>> contours;
                findContours(imgColorSpace, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

                //finding green contours
                vector<vector<Point>> contoursGreen;
                findContours(imgColorSpaceGreen, contoursGreen, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

                /// Draw contours and find biggest contour (if there are other contours in the image, we assume the biggest one is the desired rect)
                // For blue color detection
                for (std::size_t i = 0, max = contours.size(); i != max; ++i)
                {
                    float tmp = (float)contourArea(contours[i]);
                    // Finding the biggest blue contour
                    if (tmp > area && tmp > 800)
                    {
                        area = tmp;
                    }
                } // loop ends here for blue color

                opendlv::proxy::PedalPositionRequest pedalReq;
                opendlv::proxy::WheelSpeedRequest wheelSpeedRequest;

                // for green color detection
                int count = 0;
                for (std::size_t i = 0, maxGreen = contoursGreen.size(); i != maxGreen; ++i)
                {
                    // Calculating the area of the green color
                    float ctArea = (float)cv::contourArea(contoursGreen[i]);
                    // if a contour is bigger than 400, it is counted as a car
                    if (ctArea >= 400)
                    {
                        count++;
                    }
                    // count cannot be more than 3 since at the intersection can only be 3 cars or less.
                    if (count > 3)
                    {
                        count = 3;
                    }
                    // Finding the coorinates of the surrounding box for the found contour
                    cv::RotatedRect boundingBox = cv::minAreaRect(contoursGreen[i]);

                    // draw the rotated rect
                    cv::Point2f corners[4];
                    boundingBox.points(corners);
                    // used for local testing to draw rectangle arround the detected green area.
                    cv::line(img, corners[0], corners[1], cv::Scalar(0, 0, 256));
                    cv::line(img, corners[1], corners[2], cv::Scalar(0, 0, 256));
                    cv::line(img, corners[2], corners[3], cv::Scalar(0, 0, 256));
                    cv::line(img, corners[3], corners[0], cv::Scalar(0, 0, 256));
                    leftCornerX = corners[1].x;
                    leftCornerY = corners[1].y;
                    // detecting a left car
                    if (leftCornerX <= maxLeftX && leftCornerX > 50 && leftCornerY < 250 && leftCornerY > maxLeftY && atIntersection == false)
                    {
                        left = 1;
                    }
                }
                // finding the number of cars at the intersection
                if (left == 1 && atIntersection == false && frameCount < 80)
                {
                    std::cout << "left car found :" << endl;
                    maxCar = count;
                    frameCount++;
                }
                else if (atIntersection == false && left != 1 && frameCount < 80)
                {
                    std::cout << "No left car found :" << endl;
                    maxCar = count;
                    frameCount++;
                    left = -1;
                }

                std::cout << "max cars:" << maxCar << endl;
                std::cout << "cars counter founds :" << count << endl;
                std::cout << "frames  founds :" << frameCount << endl;

                //cv::Point2f corners[4];

                if (atIntersection == false && area < 20000 /*&& sizeof(corners) > 0 */&& frameCount > 79)
                {
                    std::cout << "Now move forward ..." << endl;
                    pedalReq.position(SPEED);
                    od4.send(pedalReq);
                    detectYellowSign(img);
                    stopSignDetected(img);
                }
                else
                {
                    if (stopSignDetected(img))
                    {
                        if (area > 20000 /* && sizeof(corners) > 0*/)
                        {
                            std::cout << "Now stop ..." << endl;
                            pedalReq.position(0);
                            od4.send(pedalReq);
                        }
                        else
                        {
                            std::cout << "Now move forward ..." << endl;
                            pedalReq.position(SPEED);
                            od4.send(pedalReq);
                        }
                    }
                    else if (frameCount > 79)
                    {
                        // TODO add the intersection logic
                        std::cout << "and stop" << endl;
                        pedalReq.position(0);
                        wheelSpeedRequest.wheelSpeed(0);
                        od4.send(pedalReq);
                        od4.send(wheelSpeedRequest);

                        if (frameCount < 220)
                        {
                            frameCount++;
                        }

                        if (frameCount > 200)
                        {
                            if (maxCar != 0 && atIntersection == true)
                            {

                                if (count == maxCar && left == 1)
                                {
                                    maxCar = maxCar - 1;
                                    std::cout << "max cars in the end:" << maxCar << endl;
                                    left = -1;
                                }
                                else if (count == maxCar - 2 && left == 1)
                                {
                                    maxCar = maxCar - 1;
                                    std::cout << "max cars in the end:" << maxCar << endl;
                                }
                                else if (count < maxCar && left == -1)
                                {
                                    maxCar = maxCar - 1;
                                    std::cout << "max cars in the end:" << maxCar << endl;
                                }
                            }
                            else
                            {
                                opendlv::proxy::GroundSteeringRequest steerReq;
                                std::cout << "our turn to move, waiting for direction " << endl;
                                std::cout << "max cars in the end:" << maxCar << endl;
                                maxCar = 0;
                                //moving forward
                                if (cmpf(tempDistReading, 1.0))
                                {
                                    pedalReq.position(SPEED);
                                    od4.send(pedalReq);
                                    std::this_thread::sleep_for(std::chrono::milliseconds(8 * delay));
                                    pedalReq.position(0.0);
                                    od4.send(pedalReq);
                                }
                                // turning right
                                else if (cmpf(tempDistReading, 2.0) && turnRight == true)
                                {
                                    cout << "Moving some Steps!" << endl;
                                    if (toMoveForward == true)
                                    {
                                        for (int i = 0; i < STEPS_FORWARD; i++)
                                        {
                                            pedalReq.position(SPEED);
                                            od4.send(pedalReq);
                                        }
                                        toMoveForward = false;
                                    }
                                    steerReq.groundSteering(RIGHT_TURN);
                                    od4.send(steerReq);
                                    cout << "turning Right" << endl;
                                    pedalReq.position(SPEED);
                                    od4.send(pedalReq);
                                    std::this_thread::sleep_for(std::chrono::milliseconds(10 * delay));
                                    pedalReq.position(0.0);
                                    steerReq.groundSteering(0.0);
                                    od4.send(steerReq);
                                    od4.send(pedalReq);
                                }
                                // turning left
                                else if (cmpf(tempDistReading, 8.0) && turnRight == true)
                                {
                                    cout << "MOving some Steps!" << endl;
                                    if (toMoveForward == true)
                                    {
                                        for (int i = 0; i < STEPS_FORWARD; i++)
                                        {
                                            pedalReq.position(SPEED);
                                            od4.send(pedalReq);
                                        }
                                        toMoveForward = false;
                                    }
                                    steerReq.groundSteering(RIGHT_TURN);
                                    od4.send(steerReq);
                                    cout << "turning Right" << endl;
                                    pedalReq.position(SPEED);
                                    od4.send(pedalReq);
                                    std::this_thread::sleep_for(std::chrono::milliseconds(10 * delay));
                                    pedalReq.position(0.0);
                                    steerReq.groundSteering(0.0);
                                    od4.send(steerReq);
                                    od4.send(pedalReq);
                                    std::this_thread::sleep_for(std::chrono::milliseconds(delay));
                                }
                                // turning right
                                else if (cmpf(tempDistReading, 3.0))
                                {
                                    steerReq.groundSteering(LEFT_TURN);
                                    od4.send(steerReq);
                                    cout << "turning left" << endl;
                                    pedalReq.position(SPEED);
                                    od4.send(pedalReq);
                                    std::this_thread::sleep_for(std::chrono::milliseconds(8 * delay));
                                    pedalReq.position(0.0);
                                    steerReq.groundSteering(0.0);
                                    od4.send(steerReq);
                                    od4.send(pedalReq);
                                }
                                // to make the car stop
                                else if (cmpf(tempDistReading, 9.0))
                                {
                                    pedalReq.position(0.0);
                                    steerReq.groundSteering(0.0);
                                    od4.send(steerReq);
                                    od4.send(pedalReq);
                                }
                                else if (turnRight == false)
                                {
                                    std::cout << "yellow Sign Detected, Can not turn right!" << std::endl;
                                }
                            }
                        }
                    }
                }
                cv::waitKey(1);
                area = 0;
            }
        }
        retCode = 0;
    }
    return retCode;
}

// Detecting trafic sign (right turn is not allowed)
void detectYellowSign(Mat img)
{
    cv::Mat imgHSV;
    cvtColor(img, imgHSV, cv::COLOR_BGR2HSV);
    imgHSV.convertTo(imgHSV, CV_8U);
    cv::Mat imgColorSpace;
    cv::inRange(imgHSV, cv::Scalar(5, 164, 151), cv::Scalar(26, 225, 255), imgColorSpace);
    vector<vector<Point>> contours;
    findContours(imgColorSpace, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (std::size_t i = 0, max = contours.size(); i != max; ++i)
    {
        float tmp = (float)contourArea(contours[i]);
        if (tmp > 300)
        {
            turnRight = false;
            std::cout << "right turn is not allowed" << std::endl;
        }
    }
}
//detecting a stop sign
bool stopSignDetected(Mat img)
{
    bool detected = false;
    vector<Rect> stopsign;
    CascadeClassifier stopsign_cascade;
    stopsign_cascade.load("./stopSign.xml");

    stopsign_cascade.detectMultiScale(img, stopsign, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(140, 140));
    if (stopsign.size() > 0)
    {
        detected = true;
        atIntersection = true;
        cout << "stop sign detected" << endl;
    }
    else
    {
        detected = false;
        cout << "stop sign NOT detected" << endl;
    }
    return detected;
}
// a function used to compare floats.
bool cmpf(float A, float B)
{
    float epsilon = 0.005f;
    return (fabs(A - B) < epsilon);
}
