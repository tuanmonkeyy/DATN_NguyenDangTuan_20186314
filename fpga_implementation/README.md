# Triển khai mô hình học sâu trên Xilinx SoC FPGA với Vitis AI
Hướng dẫn triển khai các mô hình học sâu YOLO tuỳ chỉnh trên kit ZCU104 với Vitis AI. Hướng dẫn này cũng được áp dụng với các board MPSoC khác mà Vitis AI hỗ trợ như ZCU102, KV260...

---
<div id='requirements'/>

## Hướng dẫn cài đặt

**Yêu cầu tối thiểu về phần cứng và phần mềm để cài đặt và sử dụng Vitis AI:**
![yeucau](images/Annotation%202023-03-10%20153118.png)

**Sau khi đã đáp ứng được các yêu cầu trên, thực hiện cài đặt Vitis AI Docker image và set up board ZCU104 như sau:**
- **[Cài đặt Vitis AI Docker image](https://github.com/Xilinx/Vitis-AI/blob/1.4/docs/quick-start/install/install_docker/README.md "Install Vitis AI Docker")**
- **[Set up board ZCU104](https://docs.xilinx.com/r/1.4-English/ug1414-vitis-ai/Setting-Up-the-ZCU102/104-Evaluation-Board "ZCU104 setup")**

**Các phiên bản phần mềm được sử dụng trong đồ án:**
- Docker : 20.10.6
- Docker Vitis AI image : 1.4.916   
- Vitis AI : 1.4     
- TensorFlow : 1.15.2
- Python : 3.6.12
- Anaconda : 4.9.2

---
<div id='guide'/>

## Triển khai

### Mục lục
- [1) Khởi động môi trường Vitis AI Docker](#docker)
- [2) Khởi tạo dự án](#init)
- [3) CChuyển đổi định dạng mô hình từ Dakrnet sang TensorFlow](#convert)
- [4) Chuẩn bị test set, val set và calib set](#set)
- [5) Đánh giá mô hình TensorFlow](#danhgia)
- [6) Lượng tử hóa mô hình](#quantize)
- [7) Biên dịch mô hình](#compile)
- [8) Biên dịch chương trình thực thi](#build)
- [9)  Cấu hình board và kết nối](#cfg)
- [10) Chạy chương trình thực thi và đánh giá kết quả trên board](#runeval)
- [11) Demo với camera](#demo)
---
<div id='docker'/>

### 1) Khởi động môi trường Vitis AI Docker
Bước đầu tiên là mở Vitis AI Docker image để truy cập các công cụ và thư viện Vitis AI.
Em sử dụng Docker image cho CPU host: [xilinx/vitis-ai-cpu:1.4.916](https://hub.docker.com/r/xilinx/vitis-ai-cpu "Docker Vitis AI CPU").
Tuy nhiên, nếu có card đồ họa NVIDIA tương thích có hỗ trợ CUDA, cũng có thể sử dụng GPU docker.
```
./docker_run.sh xilinx/vitis-ai-cpu:1.4.916
```
![docker](images/Screenshot%20from%202023-03-08%2019-48-40.png)

---
<div id='init'/>

### 2) Kích hoạt môi trường ảo Conda Vitis AI với framework TensorFlow
Kích hoạt môi trường Conda cho Vitis AI TensorFlow framework để sử dụng các thư viện Python và các lệnh Vitis AI.

```
conda activate vitis-ai-tensorflow
```

---

<div id='convert'/>


### 3) Chuyển đổi định dạng mô hình từ Dakrnet sang TensorFlow
Cần chuẩn bị file config và weights của mô hình, ví dụ mô hình YOLOv4 với đường dẫn */model/darknet_model/yolov4.cfg* và */model/darknet_model/yolov4_best.weights*. Model Keras trung gian được đặt trong đường dẫn */model/keras_model/yolov4.h5*, chú ý là thư mục keras_model phải được tạo trước.
```
python keras-YOLOv3-model-set/tools/model_converter/convert.py \
	   model/darknet_model/yolov4.cfg \
	   model/darknet_model/yolov4_best.weights \
	   model/keras_model/yolov4.h5
```
![set](images/Screenshot%20from%202023-03-10%2016-56-10.png)

Sau đó chuyển đổi từ Keras sang TF, khi hoàn thành chuyển đổi frozen TF graph sẽ được lưu ở đường dẫn */model/frozen_tf_model/yolov4_frozen.pb*.
```
python keras-YOLOv3-model-set/tools/model_converter/keras_to_tensorflow.py \
	   --input_model model/keras_model/yolov4.h5 \
	   --output_model=model/frozen_tf_model/yolov4_frozen.pb
```
![set](images/Screenshot%20from%202023-03-10%2016-59-46.png)
---
<div id='set'/>

### 4) Chuẩn bị test set, val set và calib set
Để đánh giá các mô hình sau bước chuyển đổi định dạng cũng như sau khi biên dịch, cần chuẩn bị test set và val set để đánh giá, và trong bước này cũng chuẩn bị luôn calib set cho bước lượng tử hóa mô hình. Test set trong đường dẫn */data/test_set* và val set trong */data/val_set*, trong các thư mục này sẽ chứa ảnh và nhãn.
Để dễ dàng cho việc đánh giá mAP của các mô hình sau này thì tất cả nhãn và bouding box sẽ được tổng hợp vào 1 file text duy nhất, ví dụ với test set, nhãn sẽ được lưu trong */data/test_set_labels_anchors.txt*, với val set cũng tương tự.
```
python src/gather_labels_anchors.py \
	--dataset data/test_set \
	--image_format jpg \
	--output_file data/test_set_labels_anchors.txt
```
Tiếp theo là chuẩn bị calib set cho bước lượng tử hóa, calib set không cần nhãn do đó chỉ cần ảnh trong đường dẫn */data/calib_set* và cần 1 file text liệt kê các ảnh trong calib set, file này trong đường dẫn */data/calib.txt*

---


<div id='danhgia'/>

### 5) Đánh giá mô hình Tensorflow
Bước này đánh giá sự suy giảm độ chính xác của mô hình sau quá trình chuyển đổi từ định dạng Darknet sang Tensorflow. Ở đây ví dụ với mô hình YOLOv4. Các tham số đầu vào là đường dẫn đến TF frozen graph, tên của các node đầu vào và đầu ra (có thể xem bằng công cụ Netron), đường dẫn đến tập ảnh cần kiểm tra, ở đây là test set, định dạng của hình ảnh, file text chứa tên các lớp mục tiêu trong */model/classes.txt* và anchors của mô hình trong */model/yolov4_anchors.txt*, các giá trị ngưỡng conf (det_thresh) và ngưỡng nms (nms_thresh), định dạng ảnh và file text đầu ra chứa nhãn và tọa độ của các bounding box tương ứng với các kết quả phát hiện */result/tf_model/output.txt*.
```
python src/run_graph.py \
	--graph model/frozen_tf_model/yolov4_frozen.pb \
	--input image_input \
	--outputs conv2d_93/BiasAdd,conv2d_101/BiasAdd,conv2d_109/BiasAdd \
	--anchors model/yolov4_anchors.txt \
	--classes model/classes.txt \
	--det_thresh 0.15 \
	--nms_thresh 0.25 \
	--dataset data/test_set \
	--img_format jpg \
	--results result/tf_model/output.txt
```
Sau đó đánh giá mAP bằng lệnh sau:
```
python src/eval.py \
	--results_file result/tf_model/output.txt \
	--gt_file data/test_set_labels_anchors.txt \
	--detection_thresh 0.15 \
	--iou_thresh 0.25
```

---
<div id='quantize'/>

### 6) Lượng tử hóa mô hình
Tiếp theo là bước lượng tử hoá mô hình. Bước này bao gồm một giai đoạn hiệu chỉnh trong đó sử dụng hàm callback được định nghĩa trong một đoạn code Python ở *src/yolo_graph_input_keras_fn.py*. Hàm callback có tên *calib_input* được chạy ở mỗi lần lặp lại quy trình hiệu chuẩn. Để thực hiện hiệu chỉnh, nó sẽ mở ở mỗi lần lặp lại một loạt hình ảnh được liệt kê trong file */data/calib.txt*.
Đầu ra là mô hình lượng tử hóa, được lưu trong đường dẫn */model/quantized_model/quantize_eval_model.pb*. Các tham số đầu vào là: đường dẫn đến TF frozen graph, các nút đầu vào và đầu ra của graph và kích thước đầu vào, số lần lặp để hiệu chuẩn. Ở đây em đang cấu hình batch size = 16, calib set có 640 ảnh, do đó cần số lần lặp (iterations) là 640/16=40 lần.

```
vai_q_tensorflow quantize \
 			  --input_frozen_graph model/frozen_tf_model/yolov4_frozen.pb \
			  --input_fn yolo_graph_input_keras_fn.calib_input \
			  --output_dir model/quantized_model \
	          --input_nodes image_input \
			  --output_nodes conv2d_93/BiasAdd,conv2d_101/BiasAdd,conv2d_109/BiasAdd \
			  --input_shapes ?,640,480,3 \
			  --calib_iter 40
```
---
<div id='compile'/>

### 7) Biên dịch mô hình
Để biên dịch mô hình đã lượng tử hoá, cần phải cung cấp file cấu hình DPU tương ứng với phần cứng board, trong trường hợp của em là kit ZCU104 và kiến trúc DPUCZDX8G_ISA0_B4096_MAX_BG2, file này có trong đường dẫn */opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json*. Đầu vào là mô hình đã lượng tử hoá *quantize_eval_model.pb*, đường dẫn lưu mô hình được biên dịch và tên mô hình (mạng), ở đây tên mạng của em là dpu_yolov4, lưu ý là phải chỉ định kích thước đầu vào cho trình biên dịch.
```
vai_c_tensorflow --frozen_pb model/quantized_model/quantize_eval_model.pb \
                 --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json \
				 --output_dir model/compiled_model/ \
				 --net_name dpu_yolov4 \
				 --options "{'mode':'normal','save_kernel':'', 'input_shape':'1,640,480,3'}"
```

Mô hình đã biên dịch được lưu trong */model/compiled_model/*, tiếp theo cần phải tạo một file prototxt trùng tên với tên mô hình được biên dịch, ở đây là *dpu_yolov4.prototxt*, file này chứa các thông tin của mô hình để cung cấp cho Vitis AI Library khi chạy mô hình, cấu trúc của file này có thể tham khảo trong mẫu các mô hình đã biên dịch trong đồ án. Tất cả các thành phần này phải lưu trong một thư mục có tên của mô  hình và sao chép vào thẻ nhớ SD với đường dẫn */usr/share/vitis_ai_library/models/*.

Các mô hình đã biên dịch trong đồ án được lưu trữ tại [đây](https://drive.google.com/file/d/1fB3r0f7s7CjELueODlJto7fVzSmxcQRA/view?usp=share_link "compiled").

---
<div id='build'/>

### 8) Biên dịch chương trình thực thi

Bước tiếp theo là biên dịch chương trình thực thi, như đã trình bày trong đồ án xây dựng 2 chương trình thực thi tương ứng với xử lý và lưu ảnh, xử lý video hoặc frame từ camera và hiển thị lên màn hình được lưu trong đường dẫn tương ứng là *app/test_img_yolov4.cpp* và *app/test_video_yolov4.cpp*. Để biên dịch 2 chương trình này sử dụng cross-compiler đã được cài đặt, chương trình thực thi được biên dịch nằm trong cùng thư mục.
```
cd app;
source /home/tuan/petalinux_sdk_2021.1/environment-setup-cortexa72-cortexa53-xilinx-linux;
./build.sh
```
Sao chép chương trình thực thi đã biên dịch vào thẻ nhớ SD.

---
<div id='cfg'/>

### 9) Cấu hình board và kết nối
Cắm thẻ nhớ SD đã flash image và được sao chép mô hình và chương trình thực thi vào board, cấu hình board như sau:
![config](images/Screenshot%20from%202023-03-10%2020-47-04.png)
Kết nối board với PC thông qua cáp USB đi kèm theo bộ kit, giao thức UART được sử dụng, sử dụng minicom trên Ubuntu:
```
sudo apt update 
sudo apt install minicom -y
sudo minicom -D /dev/ttyUSB1
```
Cấu hình UART như sau:
Bps / Par / Bits: 115200 8N1
Hardware Flow Control: No

---
<div id='runeval'/>

### 10) Chạy chương trình thực thi và đánh giá kết quả trên board
Bước cuối cùng là chạy chương trình thực thi trên board ZCU104. Đầu tiên là chương trình xử lý, lưu ảnh và kết quả. Các tham số đầu vào là đường dẫn tới thư mục chứa ảnh, đường dẫn thư mục lưu ảnh kết quả, đựờng dẫn lưu file text chứa kết quả phát hiện. 

```
cd app;
./yolov3 dpu_yolov4_pruned data/test_set \
						   result/compiled_model/dpu_yolov4/test_set/images \
						   result/compiled_model/dpu_yolov4/test_set/output.txt \
						   model/classes.txt \
						   ZCU104
```
Ảnh kết quả được lưu trong *result/compiled_model/dpu_yolov4/test_set/images*, tiếp theo là đánh giá mAP:

```
python3 src/eval.py \
	--results_file result/compiled_model/dpu_yolov4/test_set/output.txt \
	--gt_file data/labels_anchors_test_set.txt \
	--detection_thresh 0.15 \
	--iou_thresh 0.25
```

---
<div id='demo'/>

### 11) Demo với camera
Camera được kết nối với board qua USB, kết nối board với màn hình bằng cáp DisplayPort đực-đực, cấu hình màn hình hiển thị như sau:
```
export DISPLAY=:0.0
xrandr --output DP-1 --mode 800x600
```

Chạy chương trình demo với camera, số luồng có thể tuỳ chỉnh.
```
cd app
./test_video_yolov4 dpu_yolov4 0 -t4
```

---
<div id='references'/>

## Tài liệu tham khảo
- [TF Keras YOLOv4/v3/v2 Modelset by David8862](https://github.com/david8862/keras-YOLOv3-model-set "david8862/keras-YOLOv3-model-set")
- [Xilinx - Vitis AI Tutorials](https://github.com/Xilinx/Vitis-Tutorials/tree/master/Machine_Learning "Vitis AI tutorials")
