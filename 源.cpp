#include <opencv.hpp>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <direct.h>

using namespace std;
using namespace cv;

const int FACE_WIDTH = 120;
const int FACE_HEIGHT = FACE_WIDTH;
string trained_file_path = "C://Windows//System32//ProtectFile";
string trained_file_name = "//sysTrained.xml";
int pwd = 1;
const double DOOR = 0.4;

double getSimilarMat(Mat A, Mat B)
{
	double L2;
	if(A.size() == B.size())
		L2 = norm(A, B, CV_L2);
	double similarity = L2 / (double)(A.rows * A.cols);
	return similarity;
}

bool cascadeClassifierLoad(CascadeClassifier &detect, const char *file_name , const char *str)
{
	try {
		detect.load(file_name);
	}
	catch (cv::Exception e) {}
	if (detect.empty())
	{
		cout << str << endl;
		exit(0);
	}
}

bool getRoatData(double &dy, 
				double &dx, 
				double &angle, 
				double &len, 
				double &scale ,
				Point2f &center,
				Rect right_Rect,
				Rect left_Rect,
				Mat img
				)
{
	dy = right_Rect.y + right_Rect.height / 2 - left_Rect.y - left_Rect.height / 2;
	dx = right_Rect.x + right_Rect.width / 2 - left_Rect.x - left_Rect.width / 2;
	if ((abs(dx) < 10) || (abs(dy) > 10))
		return false;
	angle = atan2(dy, dx)*180.0 / CV_PI + 180;
	center.x = img.cols * 0.5f;
	center.y = img.rows * 0.5f;
	len = sqrt(dy*dy + dx*dx);
	scale = 0.58 * FACE_WIDTH / len;
	return true;
}

Mat equalFaceImg(Mat operateFaceImg)
{
	int w = operateFaceImg.cols;
	int h = operateFaceImg.rows;
	Mat wholeFace;
	equalizeHist(operateFaceImg, wholeFace);
	int midX = w / 2;
	Mat leftSide = operateFaceImg(Rect(0, 0, midX, h));
	Mat rightSide = operateFaceImg(Rect(midX, 0, w - midX, h));
	equalizeHist(leftSide, leftSide);
	equalizeHist(rightSide, rightSide);
	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
		{
			int v;
			if (x < w / 4)
			{
				v = leftSide.at<uchar>(y, x);
			}
			else if (x < w * 2 / 4)
			{
				int lv = leftSide.at<uchar>(y, x);
				int lw = wholeFace.at<uchar>(y, x);
				float f = (x - w / 4) / float(w / 4);
				v = cvRound((1.0 - f)*lv + (f)*lw);
			}
			else if (x < w * 3 / 4)
			{
				int lv = rightSide.at<uchar>(y, x - midX);
				int lw = wholeFace.at<uchar>(y, x);
				float f = (x - w * 2 / 4) / float(w / 4);
				v = cvRound((1.0 - f)*lv + (f)*lw);
			}
			else
			{
				v = rightSide.at<uchar>(y, x - midX);
			}
			wholeFace.at<uchar>(y, x) = v;
		}
	return wholeFace;
}

void paintEllipseFace(Mat &operateFaceImg)
{
	Mat mask = Mat(operateFaceImg.size(), CV_8UC1, Scalar(255));
	Point faceCenter = Point(cvRound(FACE_WIDTH*0.5), cvRound(FACE_HEIGHT*0.4));
	Size size = Size(cvRound(FACE_WIDTH*0.5), cvRound(FACE_HEIGHT*0.8));
	ellipse(mask, faceCenter, size, 0, 0, 360, Scalar(0), CV_FILLED);
	operateFaceImg.setTo(Scalar(128), mask);
}

bool faceOperateBeforeRecognize(std::vector<Rect> right_Rect, std::vector<Rect> left_Rect, Mat &operate_Face_Img , Mat largestFaceImg)
{
	double dy, dx, angle, len, scale;
	Point2f center;
	if (!getRoatData(dy, dx, angle, len, scale, center, right_Rect.at(0), left_Rect.at(0), operate_Face_Img))
		return false;
	Mat mapMatrix = getRotationMatrix2D(center, angle, scale);
	operate_Face_Img = Mat(FACE_WIDTH, FACE_HEIGHT, CV_32F, Scalar(128));
	warpAffine(largestFaceImg, operate_Face_Img, mapMatrix, largestFaceImg.size());
	Mat wholeFace;
	wholeFace = equalFaceImg(operate_Face_Img);
	operate_Face_Img = Mat(wholeFace.size(), CV_8U);
	bilateralFilter(wholeFace, operate_Face_Img, 0, 2.0, 1.5);
	paintEllipseFace(operate_Face_Img);
	return true;
}

void getTrainedVectorImg(Mat &old_mat , Mat operateFaceImg , vector<Mat> &preFaces, vector<int> &preNames , int &user_name , unsigned int &counts)
{
	old_mat = Mat(operateFaceImg);
	preFaces.push_back(old_mat);
	preNames.push_back(user_name++);
	cout << "Trained pics:   " << counts++ << endl;
}

void writeFaceFile(unsigned int counts , Mat operateFaceImg)
{
	string face_index;
	stringstream face_index_temp;
	face_index_temp.fill('0');
	face_index_temp << (counts - 1);
	face_index = face_index_temp.str();
	string mat_file_name = trained_file_path + "//Faces//" + "face_" + face_index + ".xml";
	FileStorage mat_file(mat_file_name.data(), FileStorage::WRITE);
	mat_file << "FACE" << operateFaceImg;
	mat_file.release();
}

bool trainedFace(vector<Mat> preFaces, vector<int> preNames)
{
	bool haveContrib = initModule_contrib();
	if (!haveContrib)
	{
		cerr << "Lack of contrib model" << endl;
		exit(0);
	}
	string faceAlName = "FaceRecognizer.Fisherfaces";
	Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
	if (model.empty())
	{
		cout << "The model is not available" << endl;
		exit(0);
	}
	model->train(preFaces, preNames);
	string trained_file_temp = trained_file_path + trained_file_name;
	model->save(trained_file_temp.data());
	return true;
}

double compareWhoYouAre(Mat operateFaceImg)
{
	Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
	if (model.empty())
	{
		cout << "The model is not available" << endl;
		exit(0);
	}
	string trained_file_temp = trained_file_path + trained_file_name;
	model->load(trained_file_temp.data());
	double identity;
	int id = model->predict(operateFaceImg);
	string face_index;
	stringstream face_index_temp;
	face_index_temp.fill('0');
	face_index_temp << (id + 1);
	face_index = face_index_temp.str();
	string mat_file_name = trained_file_path + "//Faces//" + "face_" + face_index + ".xml";
	FileStorage mat_file(mat_file_name.data(), FileStorage::READ);
	Mat comMat;
	mat_file["FACE"] >> comMat;
	identity = getSimilarMat(comMat, operateFaceImg);
	mat_file.release();
	return identity;
}

double funFaceDetect(int order_1 , char order_2)
{
	string cascadeFileName = "LBFace.xml", winName_1 = "faces", winName_2 = "face";
	string eyeDetectFileNameLeft = "HALEye.xml";
	string eyeDetectFileNameRight = "HAREye.xml";

	int loop_coun = 0;
	CascadeClassifier faceDetect;
	cascadeClassifierLoad(faceDetect, cascadeFileName.data(), "Can not detect any face");

	Mat frame, gray, old_mat;
	vector<Mat> images;
	vector<int> labels;
	vector<Mat> preFaces;
	vector<int> preNames;
	int user_name = 1;
	unsigned int counts = 1;

	int flags = CASCADE_SCALE_IMAGE;
	Size minFeature(20, 20);
	float searchScaleFactor = 1.1f;
	int minNeighbors = 4;

	VideoCapture camera;
	int choice_camera = order_1;
	camera.open(choice_camera);

	char work_method = order_2;

	while (1)
	{
		camera >> frame;
		cvtColor(frame, gray, CV_BGR2GRAY);
		Mat equalImg;
		equalizeHist(gray, equalImg);

		std::vector<Rect> faces;
		faceDetect.detectMultiScale(equalImg, faces, searchScaleFactor, minNeighbors, flags, minFeature);
		Rect largestFace;
		if (faces.size() > 0)
		{
			largestFace = faces.at(0);
			for (unsigned int i = 0; i < faces.size(); i++)
			{
				if (faces.at(i).area() > largestFace.area())
					largestFace = faces.at(i);
				rectangle(frame, faces.at(i), Scalar(0, 0, 255));
			}

			Mat largestFaceImg;
			Mat operateFaceImg;
			resize(gray(largestFace), largestFaceImg, Size(FACE_WIDTH, FACE_HEIGHT));
			resize(gray(largestFace), operateFaceImg, Size(FACE_WIDTH, FACE_HEIGHT));

			CascadeClassifier eyeDetectLeft;
			CascadeClassifier eyeDetectRight;
			cascadeClassifierLoad(eyeDetectLeft, eyeDetectFileNameLeft.data(), "Can not detect any left eye");
			cascadeClassifierLoad(eyeDetectRight, eyeDetectFileNameRight.data(), "Can not detect any right eye");

			std::vector<Rect> eyeLeftRect;
			std::vector<Rect> eyeRightRect;
			eyeDetectLeft.detectMultiScale(largestFaceImg, eyeLeftRect, 1.1, 3, CASCADE_FIND_BIGGEST_OBJECT, Size(10, 10));
			eyeDetectRight.detectMultiScale(largestFaceImg, eyeRightRect, 1.1, 3, CASCADE_FIND_BIGGEST_OBJECT, Size(10, 10));
			
			if ((eyeLeftRect.size() > 0) && (eyeRightRect.size() > 0))
			{
				if(!faceOperateBeforeRecognize(eyeRightRect, eyeLeftRect, operateFaceImg, largestFaceImg))
					continue;

				if ((work_method == 'r') || (work_method == 'R'))
				{
					if (old_mat.empty())
					{
						getTrainedVectorImg(old_mat, operateFaceImg, preFaces, preNames, user_name, counts);
					}
					else
					{
						if (getSimilarMat(old_mat, operateFaceImg) > 0.2)
						{
							getTrainedVectorImg(old_mat, operateFaceImg, preFaces, preNames, user_name, counts);
						}
					}
					writeFaceFile(counts, operateFaceImg);
					if (counts > 50)
					{
						if (trainedFace(preFaces, preNames))
							break;
					}
				}
				else if ((work_method == 'E') || (work_method == 'e'))
				{
					
					double identity = compareWhoYouAre(operateFaceImg);
					cout << "You have " << 20 - loop_coun << " chances" << endl;
					loop_coun++;
					if ((identity < DOOR) || (loop_coun > 20))
					{
						frame.release();
						destroyAllWindows();
						return identity;
					}
				}
				else
				{
					cout << "Cannot read your order!" << endl;
				}

			}
		}
		imshow(winName_1, frame);
		if (waitKey(10) == 27)
		{
			frame.release();
			destroyAllWindows();
			return 1;
		}
	}
	frame.release();
	destroyAllWindows();
}

void lockFile(const char *file_name)
{
	char file_name_temp[1000] = "";
	const unsigned int SIZE = 100;
	char head[200] = "cbc is great, he is a handsome man and a good man. he love coding and music....";
	char buffer[SIZE + 200];
	strcat(file_name_temp, file_name);
	strcat(file_name_temp, "temp");
	fstream file;
	file.open(file_name, ios_base::in | ios_base::out | ios_base::app | ios_base::binary);
	if (!file.is_open()) {
		cout << "Failed open the file:¡¡" << file_name << endl;
		return;
	}
	else
	{
		cout << "Open " << file_name << " Succeed!" << endl;
	}

	fstream file_temp;
	file_temp.open(file_name_temp, ios_base::in | ios_base::out | ios_base::app | ios_base::binary);
	if (!file_temp.is_open())
	{
		cout << "Failed create the file:¡¡" << file_name_temp << endl;
		return;
	}
	else
	{
		cout << "Create " << file_name_temp << " Succeed!" << endl;
	}

	file.read(buffer, strlen(head));
	if (strcmp(buffer, head) == 0)
	{
		cout << "Begin to crypt..." << endl;
		while (!(file.eof()))
		{
			file.read(buffer, sizeof(char));
			if ((file.eof()))
				break;
			buffer[0] -= pwd;
			file_temp.write(buffer, sizeof(char));
		}
		cout << "Crypt succeed!" << endl;
	}
	else
	{
		file.seekp(0, ios::beg);
		cout << "Begin to encrypt..." << endl;
		file_temp.write(head, strlen(head));
		while (!(file.eof()))
		{
			file.read(buffer, sizeof(char));
			if ((file.eof()))
				break;
			buffer[0] += pwd;
			file_temp.write(buffer, sizeof(char));
		}
		cout << "Encrypt succeed!" << endl;
	}
	file.close();
	remove(file_name);
	file_temp.close();
	rename(file_name_temp, file_name);
	return;
}



int main(int argc, const char *argv[])
{
	fstream file;
	string trained_file_temp = trained_file_path + trained_file_name;
	file.open(trained_file_temp.data(), ios_base::in | ios_base::out | ios_base::app | ios_base::binary);
	if (!file.is_open()) {
		string temp = trained_file_path;
		_mkdir(temp.data());
		temp += "//Faces";
		_mkdir(temp.data());
		trained_file_temp = trained_file_path + trained_file_name;
		file.open(trained_file_temp.data(), ios_base::in | ios_base::out | ios_base::app | ios_base::binary);
	}
	if (!file.is_open()) {
		trained_file_path = "D://ProtectFile";
		trained_file_temp = trained_file_path + trained_file_name;
		file.open(trained_file_temp.data(), ios_base::in | ios_base::out | ios_base::app | ios_base::binary);
	}
	if (!file.is_open()) {
		trained_file_path = "D://ProtectFile";
		string temp = trained_file_path;
		_mkdir(temp.data());
		temp += "//Faces";
		_mkdir(temp.data());
		trained_file_temp = trained_file_path + trained_file_name;
		file.open(trained_file_temp.data(), ios_base::in | ios_base::out | ios_base::app | ios_base::binary);
	}
	if (!file.is_open()) {
		cout << "Failed Check the file:¡¡" << trained_file_name << endl;
		cout << "Press Any Key to END my Work." << endl;
		getchar();
		return 0;
	}
	unsigned int count = 1;
	while ((!(file.eof())) && count < 100)
	{
		file.get();
		count++;
	}
	if (count < 90)
	{
		cout << "I need 50 photos which contain your face.\n"
			"So I need your PC camera.\n"
			"Don't worry , it is only used to get your face.\n"
			"Please move your head as you can.\n"
			"When a red box cover your face, it means I can get your face,\n"
			"and when the counts in the console part rises, it means I have get a good face.\n"
			"\n\nNow , let's begin. If you don't want to do this , type 'q'('Q') else type any other key.\n\n" << endl;
		char ch = getchar();
		if ((ch == 'q')|| ch == 'Q')
			return 1;
		else
		{
			cout << "\n\n\n\n\nPress Any Key Escape." << endl;
			funFaceDetect(0, 'r');
			cout << "Trainde Over, thanks for your support!" << endl;
		}
	}
	if (argc > 1)
	{
		for (int i = 1; i < argc; i++)
		{
			if (funFaceDetect(0, 'e') < DOOR)
			{
				cout << "Correct!" << endl;
				lockFile(argv[i]);
			}
			else
			{
				cout << "Cannot recognize who you are!" << endl;
			}
		}
	}
	else
	{
		cout << "You can move the file which you want to encrypt above this programm or type the name of it." << endl;
		cout << "Type the file name while you want to encrypt:" << endl;
		string file_name_type;
		getline(cin, file_name_type);
		if (funFaceDetect(0, 'e') < DOOR)
		{
			cout << "Correct! " << endl;
			lockFile(file_name_type.data());
		}
		else
		{
			cout << "Cannot recognize who you are!" << endl;
		}
	}
	cout << "\n\n\nPress Any Key to END my Work." << endl;
	getchar();
	return 1;
}