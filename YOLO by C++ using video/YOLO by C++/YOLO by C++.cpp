
#include "yolo.h" // 包含YOLO类的头文件
#include <cstdlib>                     // atoi
#include <opencv2/videoio.hpp>         // VideoCapture / CAP_MSMF / CAP_DSHOW
#include <opencv2/core/utils/logger.hpp>  // 临时屏蔽 OpenCV 内部日志



// 在YOLO类中定义一个构造函数，接收一个Net_config类型的参数config
// 该构造函数用于初始化YOLO对象的各项属性，并加载预训练的YOLO模型

YOLO::YOLO(Net_config config)
{
	// 输出所使用的网络名称（来自config.netname）
	cout << "Net use " << config.netname << endl;

	// 将配置中的置信度阈值、NMS阈值、输入宽高分别赋值给YOLO对象的对应成员变量
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;
	this->inpWidth = config.inpWidth;
	this->inpHeight = config.inpHeight;

	// 将网络名称（config.netname）复制到YOLO对象的netname字符数组中
	strcpy_s(this->netname, config.netname.c_str());

	// 从配置文件中读取类别列表
	ifstream ifs(config.classesFile.c_str());  // 打开类别文件
	string line;
	while (getline(ifs, line))  // 逐行读取文件内容
	{
		// 将每行类别名称添加到YOLO对象的classes向量中
		this->classes.push_back(line);
	}

	// 加载预训练的YOLO模型
	// 参数分别为：模型配置文件路径、模型权重文件路径
	this->net = readNetFromDarknet(config.modelConfiguration, config.modelWeights);

	// 设置模型推理后端为OpenCV DNN Backend
	this->net.setPreferableBackend(DNN_BACKEND_OPENCV);

	// 设置模型推理目标设备为CPU
	this->net.setPreferableTarget(DNN_TARGET_CPU);
}

// 在YOLO类中定义一个名为postprocess的方法，该方法接受两个参数：
// 1. Mat类型的引用frame：表示当前待处理的图像帧
// 2. const vector<Mat>&类型的outs：表示YOLO模型输出的一系列特征图（可能包含多个预测层）

// 使用非最大值抑制（Non-Maximum Suppression, NMS）算法移除低置信度的边界框，

void YOLO::postprocess(Mat& frame, const vector<Mat>& outs)
{
	// 初始化存储类别ID、置信度和边界框的向量
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	// 遍历模型输出的所有特征图（outs）
	for (size_t i = 0; i < outs.size(); ++i)
	{
		// 获取当前特征图的浮点型数据指针
		float* data = (float*)outs[i].data;

		// 遍历当前特征图的每一行（即每一个预测单元）
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			// 提取当前预测单元对应的所有类别得分（scores），范围从第5列到最后一列
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);

			// 找出最高类别得分及其索引（classIdPoint）与置信度（confidence）
			Point classIdPoint;
			double confidence;
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

			// 如果当前预测单元的置信度高于预设阈值（this->confThreshold），则认为是有效检测
			if (confidence > this->confThreshold)
			{
				// 根据模型输出的相对坐标（data[0-3]）计算边界框在原图上的绝对坐标
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				// 将当前检测结果（类别ID、置信度、边界框）分别添加到相应向量中
				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	//  在这里应用非最大值抑制算法，根据置信度对所有边界框进行筛选，
	// 丢弃与高置信度框重叠过多的低置信度框，以减少重复检测

	vector<int> indices;
	NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		this->drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);
	}
}







// YOLO类的成员函数，用于在给定帧（frame）上绘制预测的边界框及其标签信息
// 参数：
//   classId：预测对象所属的类别ID
//   conf：预测对象的置信度分数
//   left, top, right, bottom：边界框的左上角坐标与右下角坐标
//   frame：待绘制的图像（Mat类型）

void YOLO::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	// 1. 绘制边界框
	// 使用rectangle函数在frame上绘制矩形，颜色为红色（Scalar(0, 0, 255)），线宽为3像素
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 3);

	// 2. 获取类别名称和置信度标签
	string label;  // 存储标签文本

	// 如果类别的向量不为空，则使用classId索引对应的类别名称
	if (!this->classes.empty())
	{
		// 确保classId在合法范围内
		CV_Assert(classId < (int)this->classes.size());

		// 构造标签文本：类别名称 + ":" + 置信度（保留两位小数）
		label = this->classes[classId] + ":" + format("%.2f", conf);
	}
	else
	{
		// 若无类别信息，仅显示置信度（保留两位小数）
		label = format("%.2f", conf);
	}

	// 3. 计算并调整标签位置
	int baseLine;  // 基线高度
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);  // 获取标签尺寸

	// 将top坐标向上调整，确保标签位于边界框上方，至少高出标签自身高度
	top = max(top, labelSize.height);

	// 注释掉的代码原本用于绘制背景填充的矩形（绿色，Scalar(0, 255, 0)），以突出标签区域
	// 现已改为直接使用putText绘制标签文本，无需背景矩形

	// 4. 绘制标签文本
	// 使用putText函数在frame上绘制标签文本，字体为FONT_HERSHEY_SIMPLEX，字号为0.75，颜色为绿色（Scalar(0, 255, 0)），线宽为1像素
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
}
// YOLO类的成员函数，用于在给定帧（frame）上执行对象检测，并绘制检测结果及性能指标
// 参数：
//   frame：待检测的图像（Mat类型）




void YOLO::detect(Mat& frame)
{
	Mat blob;  // 临时中间变量，用于存储预处理后的图像数据

	// 1. 预处理输入图像
	// 使用blobFromImage将frame转换为网络输入所需的Blob格式
	// 参数说明：
	//   frame：原始图像
	//   blob：输出的Blob对象
	//   scalefactor=1/255.0：对像素值进行归一化处理
	//   size=(this->inpWidth, this->inpHeight)：缩放图像至网络期望的输入尺寸
	//   mean=Scalar(0, 0, 0)：减去均值（此处为黑色）
	//   swapRB=true：交换BGR通道顺序为RGB（若网络需要）
	//   crop=false：不裁剪图像
	blobFromImage(frame, blob, 1 / 255.0, Size(this->inpWidth, this->inpHeight), Scalar(0, 0, 0), true, false);

	// 2. 设置网络输入
	this->net.setInput(blob);

	// 3. 执行前向传播，获取网络输出
	vector<Mat> outs;  // 存储网络输出的多层特征图
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());  // 输出指定的未连接输出层

	// 4. 后处理网络输出，提取并绘制检测结果
	this->postprocess(frame, outs);  // 调用成员函数postprocess处理网络输出并绘制边界框及标签

	// 5. 计算并绘制推理时间
	vector<double> layersTimes;  // 用于存储各层推理时间
	double freq = getTickFrequency() / 1000;  // 获取CPU时钟频率（单位：kHz）

	// 获取网络整体推理时间（单位：秒），并转换为毫秒（ms）
	double t = net.getPerfProfile(layersTimes) / freq;

	// 构建标签文本，包含网络名称和推理时间
	string label = format("%s Inference time : %.2f ms", this->netname, t);

	// 使用putText在frame图像左上角绘制推理时间标签
	putText(frame, label, Point(0, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);  // 蓝色，字号1，线宽2

	// 注释掉的代码原本用于保存处理后图像到文件（格式：netname_out.jpg）
	// imwrite(format("%s_out.jpg", this->netname), frame);
}




// 依次尝试不同后端与摄像头序号，成功则返回 true
// Windows 上 OpenCV 的 MSMF 后端在部分机型上会报 "Camera index out of range" 打不开默认摄像头，
// 因此这里按 MSMF -> DirectShow -> 自动 的顺序，配合 0~5 号设备逐个尝试
static bool openCamera(VideoCapture& cap, int wantIndex, int& usedIndex, string& usedBackend)
{
	struct BackendInfo { int id; const char* name; };
	static const BackendInfo backends[] = {
		{ CAP_MSMF,  "MSMF" },
		{ CAP_DSHOW, "DirectShow" },
		{ CAP_ANY,   "Auto" }
	};

	for (int k = 0; k < 3; ++k)
	{
		// i = -1 表示先试用户期望的序号，之后扫描 0~5 号设备
		for (int i = -1; i < 6; ++i)
		{
			int idx = (i < 0) ? wantIndex : i;
			if (i >= 0 && idx == wantIndex) continue;   // 已试过

			cap.release();
			if (cap.open(idx, backends[k].id))
			{
				usedIndex = idx;
				usedBackend = backends[k].name;
				return true;
			}
		}
	}
	return false;
}

int main(int argc, char** argv)
{
	// 摄像头序号可通过命令行参数指定，例如：YOLO by C++.exe 1
	int wantIndex = (argc > 1) ? atoi(argv[1]) : 0;

	// 创建YOLO对象（yolo_nets[2] = yolo-fastest，速度最快，适合实时视频检测）
	YOLO yolo_model(yolo_nets[2]);

	// 探测期间屏蔽 OpenCV 内部日志：失败的尝试会刷出一堆 ERROR/INFO，容易误导
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);

	VideoCapture cap;
	int usedIndex = -1;
	string usedBackend;
	bool ok = openCamera(cap, wantIndex, usedIndex, usedBackend);

	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_WARNING);

	if (!ok)
	{
		cerr << "Cannot open camera." << endl
		     << "已尝试：0~5 号设备 x MSMF / DirectShow / Auto 三种后端。" << endl
		     << "请依次检查：" << endl
		     << "  1. 摄像头是否被其他程序占用（相机 App、微信、腾讯会议、浏览器网页等）" << endl
		     << "  2. 设置 -> 隐私和安全性 -> 相机：允许桌面应用访问相机" << endl
		     << "  3. 设备管理器 -> 摄像头，确认设备存在且未被禁用" << endl
		     << "  4. 手动指定序号重试：本程序.exe 1" << endl;
		return -1;
	}

	cout << "Camera opened: index=" << usedIndex
	     << ", backend=" << usedBackend << endl
	     << "Press any key in the window to quit." << endl;

	// 创建窗口用于显示检测结果
	namedWindow("Object Detection", WINDOW_NORMAL);

	Mat frame;
	// 循环读取视频帧并进行YOLO检测
	while (true)
	{
		cap.read(frame);

		if (frame.empty())
		{
			cerr << "Failed to grab frame, camera disconnected?" << endl;
			break;
		}

		// 使用YOLO模型进行目标检测
		yolo_model.detect(frame);

		// 显示带有检测结果的图像
		imshow("Object Detection", frame);

		// 监听键盘，按任意键退出
		if (waitKey(1) >= 0)
			break;
	}

	cap.release();  // 释放捕获器
	destroyAllWindows();

	return 0;
}
