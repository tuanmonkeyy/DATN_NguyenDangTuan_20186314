# Huấn luyện và đánh giá các mô hình trên GPU

## Huấn luyện
Sử dụng framework Darknet để huấn luyện và đánh giá mô hình trên GPU. Chi tiết về cách cài đặt có thể tham khảo tại [đây](https://github.com/AlexeyAB/darknet#how-to-compile-on-linux-using-make "Darknet").

Quá trình huấn luyện mô hình trong đồ án được tiến hành trong Jupyter Notebook trên nền tảng cloud VastAI, code và scripts thực hiện ở đây được lưu trong file yolo_training.ipynb

Các mô hình đã huấn luyện và config đi kèm trong đồ án được lưu trữ tại [Google Drive sau](https://drive.google.com/file/d/1SxeBKTm3O7w6J5JcgcHx-zuFq44dvctB/view?usp=share_link "weights").

## Pruning

Các kỹ thuật cắt tỉa được triển khai là:
- Các kỹ thuật cắt tỉa dựa trên tiêu chuẩn (Criteria-based): L0-Norm, L1-Norm, L2-Norm, L-Inf Norm và Random.
- Các kỹ thuật cắt tỉa dựa trên phép chiếu (Projection-based): PLS(Single)+VIP, PLS(Multi)+VIP, CCA(Multi)+CV và PLS(Multi)+LC.
- Các kỹ thuật cắt tỉa dựa trên cụm (Cluster-based): HAC+PCC.

1. Sau khi cài đặt darknet, sao chép tất cả file trong */pruning/* vào trong đường dẫn darknet. 
2. Lưu ý là tất cả file `.cfg`, `.data`, `.names`, `.weights`, `train.txt` và `valid.txt` phải để trong đường dẫn darknet.
3. Cắt tỉa mô hình với phương pháp L1-norm:
```
`python prune.py --cfg yolo.cfg --data yolo.data --names yolo.names --weights yolo.weights --network YOLOv4 --img-size 640 --technique L1 --pruning-rate 0.60 --pruning-iter 2 --lr 0.005`
```
Kết quả mô hình đã cắt tỉa được lưu trong */darknet/temp/*, sau đó tiến hành fine-tune mô hình đã cắt tỉa.

<!--
| File | Description |
| -------- | ----------- |
| `yolo_training.ipynb` | Huấn luyện mô hình trên Jupyter Notebook |
| `xml2yolo.py` | Chuyển chú thích nhãn định dạng từ xml sang định dạng YOLO |
| `train_val_create.py` | Chia tập dữ liệu thành tập train và val theo tỷ lệ ngẫu nhiên |
| `check_train_val_freq.py` | Kiểm tra tần suất xuất hiện của các lớp mục tiêu trong tập train và val |
| `train.txt` | Danh sách ảnh tập train được sử dụng |
| `val.txt` | Danh sách ảnh tập val được sử dụng |
-->

## Tài liệu tham khảo
- [YOLO: Real-Time Object Detection by pjreddie](https://pjreddie.com/darknet/yolo/ "YOLO: Real-Time Object Detection")
- [YOLO Darknet by AlexeyAB](https://github.com/AlexeyAB/darknet "Darknet")
- [Pruning-Darknet by pedbrgs](https://github.com/pedbrgs/Pruning-Darknet "Pruning")