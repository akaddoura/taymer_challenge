#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QDebug>
#include <QFileDialog>
#include <QtGui>
#include <QMainWindow>
#include <QMessageBox>
#include <QPixmap>


#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:

    void on_measure_button_clicked();

    void measure_diameter();

    void on_find_defect_button_clicked();

    void find_defect();

    int classify_defect(cv::RotatedRect &ellipse, const cv::Mat &img);

    void on_load_image_button_clicked();

    void on_measure_save_button_clicked();

    void on_defect_save_button_clicked();

    void on_exit_button_clicked();

private:
    Ui::MainWindow *ui;
    cv::Mat img;

    int xsize = img.rows;
    int ysize = img.cols;

    cv::Mat measure_output;
    cv::Mat defect_outut;
};
#endif // MAINWINDOW_H
