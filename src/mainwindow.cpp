#include "mainwindow.h"
#include "ui_mainwindow.h"

static void grayscale(cv::Mat &src, cv::Mat &dst);
static void gaussianBlur(cv::Mat &src, cv::Mat &dst, int kernel);
static QImage MatToQImage(const cv::Mat& mat);

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::measure_diameter()
/* This function uses a precise threshold over the cable jacket to measure
 * the cable diameter in pixels. Due to few pixel inconsistencies in the mask,
 * which are in the range of 1-3 pixels, it could be made better by averaging
 * the diameter above and below the line where the measurement is taken.
 *
 * Function will measure the diameter and display the result.
 */
{
    if (this->img.empty())
    {
        QMessageBox msgBox;
        msgBox.setText("No image loaded.");
        msgBox.exec();
        return;
    }

    // copy image to draw on
    cv::Mat measureImg = this->img.clone();
    // image containers
    cv::Mat gray;
    cv::Mat blur;
    cv::Mat thresh;

    ysize = measureImg.rows;

    // grayscale the image
    grayscale(measureImg, gray);

    // apply gaussian blur
    int blurKernel = 3;
    gaussianBlur(gray, blur, blurKernel);

    // apply a binary threshold mask tuned to cover the cable's length
    int threshold_low = 60;
    int threshold_high = 255;
    cv::threshold(blur, thresh, threshold_low, threshold_high, cv::THRESH_BINARY);

    // vector of splits required to measure the cable at 3 different points
    int split = ysize/4;
    std::vector<int32_t> splits = {split, split*2, split*3};

    // line container
    cv::Mat line;

    // loop through each split and get the diameter
    for (int s : splits)
    {
        // line splice at split s
        line = thresh.row(s);

        // obtain non zero values representing the cable's length
        cv::Mat nonZero;
        cv::findNonZero(line, nonZero);

        // get the indices of the first zero on the left and right side
        int cableLeft = nonZero.at<std::int32_t>(0, 0);
        int cableRight = nonZero.at<std::int32_t>(nonZero.rows - 1, 0);

        // subtract the right from left to obtain diameter
        int diamaeter = cableRight - cableLeft;

        // drawing on measureImg
        std::string text = QString("Diameter: %1").arg(diamaeter).toUtf8().constData();
        cv::line(measureImg, cv::Point(cableLeft, s), cv::Point(cableLeft - 15, s), cv::Scalar(0, 0, 255), 1);
        cv::line(measureImg, cv::Point(cableRight, s), cv::Point(cableRight + 15, s), cv::Scalar(0, 0, 255), 1);
        cv::putText(measureImg, text, cv::Point(cableRight + 25, s), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255));

        // image display
        QImage measureImage = MatToQImage(measureImg);
        this->measure_output = measureImg;
        this->ui->image_display->setPixmap(QPixmap::fromImage(measureImage));
    }
}

void MainWindow::on_measure_button_clicked()
{
    measure_diameter();
}

int MainWindow::classify_defect(cv::RotatedRect &rectangle, const cv::Mat &img)
/* Input
 *  ellipse: a rotated rectangle where the defect was detected.
 *  img: the original img Mat required for warping.
 *
 * Output
 *  int corrosponding to the defect classification.
 *
 * Function will warp the rotatedRect into an up-right rectangle.
 * This will allow us to average the sum of each column in the defect area
 * for classification.
 */
{
    cv::Mat matrix, warped, warpedGray;

    float h = rectangle.size.height;
    float w = rectangle.size.width;

    cv::Mat hist(cv::Mat::zeros(1,w,CV_32S));

    // points of image to be warped
    cv::Point2f src[4];
    rectangle.points(src);
    // warp destination
    cv::Point2f dst[4] = { {0.0f, h - 1},
                           {0.0f, 0.0f},
                           {w - 1.0f, 0.0f},
                           {w - 1.0f, h - 1.0f} };

    // warp matrix
    matrix = cv::getPerspectiveTransform(src, dst);
    cv::warpPerspective(img, warped, matrix, cv::Point(w,h));

    // grayscale warp so white pixel intensity can be summed
    grayscale(warped, warpedGray);

    // sum all the pixel values in each column then take the mean
    cv::reduce(warpedGray, hist, 0, 0, CV_32S);
    int avgIntensity = cv::sum(hist)[0]/w;

    // hyperparameter for scratches
    if (avgIntensity > 18000)
        return 2;  //scratch
    else
    {
        float ratio;
        if (h > w)
            ratio = w/h;
        if (w >= h)
            ratio = h/w;
        if (ratio <= 0.85)
            return 1; // cut
        else
            return 0; // pinhole
    }
}

void MainWindow::find_defect()
{
    if (this->img.empty())
    {
        QMessageBox msgBox;
        msgBox.setText("No image loaded.");
        msgBox.exec();
        return;
    }

    // copy image to draw on
    cv::Mat defectImg = this->img.clone();
    // image containers
    cv::Mat gray;
    cv::Mat blur;
    cv::Mat edges;

    int ysize = img.rows;
    int xsize = img.cols;

    // grayscale image
    grayscale(img, gray);

    // apply gaussian blur
    int blurKernel = 3;
    gaussianBlur(gray, blur, blurKernel);

    // use canny detection to find edges of defects
    int canny_low = 60;
    int canny_high = 255;
    cv::Canny(blur, edges, canny_low, canny_high);

    // apply morphology to accentuate edges
    cv::Mat im_erode;
    cv::Mat im_dilate;

    cv::Mat morphKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(edges, im_dilate, morphKernel);
    cv::erode(im_dilate, im_erode, morphKernel);

    // containers for cv::findContours() output
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    // find contours
    cv::findContours(im_erode, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    // defining a new black image to draw clustered rectangle on in order to combine them later
    cv::Mat groupImg = cv::Mat(ysize, xsize, CV_8UC3, cv::Scalar(0,0,0));

    // loop over the contours to find bounding boxes and draw rectangles
    for (int i = 0; i < contours.size(); i++)
    {
        // containers
        std::vector<std::vector<cv::Point>> contourPoly(contours.size());
        std::vector<cv::Rect> boundingRectangles(contours.size());

        // find arclength and approximate poly corners for contours
        float perimeter = cv::arcLength(contours[i], true);
        cv::approxPolyDP(contours[i], contourPoly[i], 0.02*perimeter, true);

        // find the bounding rectangles
        boundingRectangles[i] = boundingRect(contourPoly[i]);
        // calculate area of each box to filter out faulty detections
        float height = boundingRectangles[i].height;
        float width = boundingRectangles[i].width;
        float area = height * width;

        // draw only the good boxes
        if (area < 10000)
           cv::rectangle(groupImg, boundingRectangles[i].tl(), boundingRectangles[i].br(), cv::Scalar(255,255,255), -1);
    }

    /* In order to combine the multiple bounding boxes cluttered over one defect,
     * we threshold and find contours again through cv::RETER_EXTERNAL which will
     * simply combine all the boxes into one large contour around the fault.
     * Much of the process is repeated with slightly different parameters.
     * */

    // grouping image containers
    cv::Mat groupGray;
    cv::Mat groupBlur;
    cv::Mat groupThresh;

    // grayscale the rectangles image
    grayscale(groupImg, groupGray);

    // apply gaussian blur
    gaussianBlur(groupGray, groupBlur, blurKernel);

    // apply a binary threshold mask
    int groupThresholdLow = 50;
    int groupThresholdHigh = 255;
    cv::threshold(groupBlur, groupThresh, groupThresholdLow, groupThresholdHigh, cv::THRESH_BINARY);

    // containers for cv::findContours() output
    std::vector<std::vector<cv::Point>> groupContours;
    std::vector<cv::Vec4i> groupHierarchy;

    // find contours
    cv::findContours(groupThresh, groupContours, groupHierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    // vector to store ellipses data for final defect classification
    std::vector<cv::RotatedRect> minEllipse(contours.size());

    // loop over new contours
    for (int i=0; i < groupContours.size(); i++)
    {
        // containers
        std::vector<std::vector<cv::Point>> contourPoly(contours.size());
        std::vector<cv::Rect> boundingRectangles(contours.size());

        // approximate poly
        cv::approxPolyDP(groupContours[i], contourPoly[i], 1, true);

        boundingRectangles[i] = boundingRect(contourPoly[i]);

        // fit ellipses over the defects
        minEllipse[i] = fitEllipse(groupContours[i]);

        // draw ellipses on img
        ellipse(defectImg, minEllipse[i], cv::Scalar(0,0,255), 2);
        // classify which type of defect
        int defect = classify_defect(minEllipse[i], img);
        std::string text;
        switch (defect)
        {
        case 0:
            text = "Defect: Pin Hole";
            break;
        case 1:
            text = "Defect: Cut";
            break;
        case 2:
            text = "Defect: Scratch";
        }

        // drawing functions
        int center_x = minEllipse[i].center.x;
        int center_y = minEllipse[i].center.y;

        cv::putText(defectImg, text, cv::Point(center_x + 60, center_y), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255));
        QImage defectImage = MatToQImage(defectImg);
        this->ui->image_display->setPixmap(QPixmap::fromImage(defectImage));
        this->defect_outut = defectImg;
    }
}

void MainWindow::on_find_defect_button_clicked()
{
    find_defect();
}

void MainWindow::on_load_image_button_clicked()
{
    // image path
   QString filePath = QFileDialog::getOpenFileName(this,
        tr("Open Image"), "/Desktop", tr("Image Files (*.png *.jpg *.bmp)"));

    std::string imagePath = filePath.toLocal8Bit().constData();

    // read image
    this->img = imread(imagePath, cv::IMREAD_COLOR);

    // if image fails to load for any reason, display message
    if (img.empty())
    {
        QMessageBox msgBox;
        msgBox.setText("Could not load image.");
        msgBox.exec();
    }

    //display image
    QImage display;
    display.load(filePath);
    QPixmap displayMap;
    displayMap.fromImage(display);
    this->ui->image_display->setPixmap(QPixmap::fromImage(QImage(img.data, img.cols, img.rows, img.step, QImage::Format_BGR888)));
}

// Image functions
static void grayscale(cv::Mat& src, cv::Mat& dst)
{
    cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
    return;
}

static void gaussianBlur(cv::Mat& src, cv::Mat& dst, int kernel)
{
    cv::GaussianBlur(src, dst, cv::Size(kernel, kernel), 0);
    return;
}

static QImage MatToQImage(const cv::Mat& mat)
// credit to eyllanesc on stackoverflow for this great function!
{
    // 8-bits unsigned, NO. OF CHANNELS=1
    if(mat.type()==CV_8UC1)
    {
        // Set the color table (used to translate colour indexes to qRgb values)
        QVector<QRgb> colorTable;
        for (int i=0; i<256; i++)
            colorTable.push_back(qRgb(i,i,i));
        // Copy input Mat
        const uchar *qImageBuffer = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage img(qImageBuffer, mat.cols, mat.rows, mat.step, QImage::Format_Indexed8);
        img.setColorTable(colorTable);
        return img;
    }
    // 8-bits unsigned, NO. OF CHANNELS=3
    if(mat.type()==CV_8UC3)
    {
        // Copy input Mat
        const uchar *qImageBuffer = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage img(qImageBuffer, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return img.rgbSwapped();
    }
    return QImage();
}

void MainWindow::on_measure_save_button_clicked()
{
    if (this->measure_output.empty())
    {
        measure_diameter();
    }
    QString fileName = QFileDialog::getSaveFileName(this, "Save File",
                               "/home/measureoutput.jpg",
                               "Images (*.png *.xpm *.jpg)");
    cv::imwrite(fileName.toUtf8().constData(), measure_output);
}

void MainWindow::on_defect_save_button_clicked()
{
    if (this->defect_outut.empty())
    {
        find_defect();
    }
    QString fileName = QFileDialog::getSaveFileName(this, "Save File",
                               "/home/defectoutput.jpg",
                               "Images (*.png *.xpm *.jpg)");
    cv::imwrite(fileName.toUtf8().constData(), measure_output);
}

void MainWindow::on_exit_button_clicked()
{
    this->close();
}
