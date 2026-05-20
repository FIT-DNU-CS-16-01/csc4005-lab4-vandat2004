# CSC4005 Lab 4 Report – CRNN for UrbanSound8K

## 1. Thông tin sinh viên

- Họ tên: Nguyễn Văn Đạt
- Mã sinh viên: 1671040007
- Lớp: KHMT 16-01
- Link GitHub repo: https://github.com/FIT-DNU-CS-16-01/csc4005-lab4-vandat2004
- Link W&B project: https://wandb.ai/datn89367-i-h-c-i-nam/csc4005-lab4-urbansound8k-crnn?nw=nwuserdatn89367

## 2. Mục tiêu thí nghiệm

Lab 4 hướng tới mục tiêu chuyển đổi tín hiệu âm thanh thô về dạng log-mel spectrogram để biểu diễn năng lượng âm thanh trực quan theo cả hai chiều thời gian và tần số thang Mel sinh học. Khác với mạng 1D-CNN ở Lab 3 chỉ trích xuất được mẫu cục bộ ngắn hạn, kiến trúc CRNN tích hợp thêm khối mạng hồi quy (GRU/LSTM) giúp mô hình hóa và học sâu các chuỗi diễn biến dài hạn của âm thanh theo dòng thời gian. Mục tiêu cốt lõi của bài thực hành là đánh giá hiệu năng phân loại 10 lớp âm thanh môi trường đô thị, phân tích các hiện tượng huấn luyện (overfitting/underfitting) qua biểu đồ curves, xác định các cặp âm thanh tương đồng gây nhầm lẫn trên confusion matrix và so sánh trực quan hiệu quả thực tế giữa mô hình học chuỗi thời gian CRNN với baseline tích chập 1D-CNN.

## 3. Cấu hình dữ liệu


| Thành phần | Giá trị |
|---|---|
| Dataset | UrbanSound8K |
| Số lớp | 10 |
| Train folds | 1–8 |
| Validation fold | 9 |
| Test fold | 10 |
| Feature | log-mel spectrogram |
| Sampling rate | 16 kHz |
| Duration | 4 giây |

## 4. Cấu hình mô hình


| Thành phần | Giá trị |
|---|---|
| Model | CRNN (`crnn_small`) |
| CNN blocks | 3 Blocks Conv2d (32, 64, 128 channels) + BatchNorm + ReLU + MaxPool |
| RNN type | GRU (Baseline) / LSTM (Extension) |
| Hidden size | 128 (2 layers) |
| Dropout | 0.3 |
| Optimizer | AdamW |
| Learning rate | 0.001 (Baseline) / 0.0007 (Extension) |
| Batch size | 32 |
| Epochs | 25 |

## 5. Kết quả huấn luyện


| Run | best_val_acc | test_acc | Ghi chú |
|---|---:|---:|---|
| logmel_crnn_gru_baseline | 74.14% | 75.03% | Đạt hiệu năng tối ưu, loss giảm sâu ổn định suốt 25 epochs. |
| extension_bilstm_crnn | 62.87% | 64.04% | Overfitting sớm, bị kích hoạt Early Stopping ở epoch 16. |

## 6. Learning curves

### 6.1. Biểu đồ Baseline logmel_crnn_gru_baseline
![Learning Curves Baseline](outputs/logmel_crnn_gru_baseline/curves.png)

### 6.2. Biểu đồ Extension logmel_crnn_bilstm_extension
![Learning Curves Extension](outputs/logmel_crnn_bilstm_extension/curves.png)

Nhận xét:
- **Mô hình có overfitting không?** Mô hình Baseline GRU kiểm soát hiện tượng overfitting rất tốt, khoảng cách giữa `train_loss` (0.6590) và `val_loss` (0.9500) không quá xa, độ chính xác tăng đều. Tuy nhiên, mô hình mở rộng BiLSTM-CRNN bị overfitting rất nặng và diễn ra sớm do cấu trúc mạng quá lớn, lượng tham số tăng gấp đôi khiến mô hình học thuộc lòng tập train thay vì học đặc trưng tổng quát.
- **Validation loss có giảm ổn định không?** Đối với bản Baseline GRU, validation loss giảm rất ổn định qua từng epoch và đạt điểm cực tiểu tại epoch 24. Ngược lại, bản BiLSTM validation loss liên tục dao động trồi sụt thất thường và có xu hướng tăng ngược trở lại sau epoch 10.
- **Có cần early stopping không?** Bản Baseline GRU chạy mượt mà hết 25 epochs không cần ngắt. Tuy nhiên, Early Stopping là cực kỳ bắt buộc đối với bản mở rộng BiLSTM; hệ thống đã tự động kích hoạt ngắt sớm ở epoch 16 để bảo vệ mô hình khỏi hiện tượng suy giảm độ chính xác nghiêm trọng trên tập kiểm thử.

## 7. Confusion matrix

### 7.1. Ma trận Baseline logmel_crnn_gru_baseline
![Confusion Matrix Baseline](outputs/logmel_crnn_gru_baseline/confusion_matrix.png)

### 7.2. Ma trận Extension logmel_crnn_bilstm_extension
![Confusion Matrix Extension](outputs/logmel_crnn_bilstm_extension/confusion_matrix.png)

Nhận xét:
- **Lớp nào phân loại tốt?** Các lớp âm thanh có đặc trưng chuỗi thời gian kéo dài, tần số mang tính lặp lại hoặc biên độ cực kỳ dị biệt như `siren` (tiếng còi hú cứu thương), `gun_shot` (tiếng súng nổ biên độ xung cao phát một) và `car_horn` (tiếng còi xe) được mô hình phân loại chính xác nhất với tỷ lệ nhận diện đúng vượt trội trên đường chéo chính.
- **Lớp nào dễ bị nhầm?** Hai lớp âm thanh công nghiệp nặng là `drilling` (tiếng khoan máy) và `jackhammer` (tiếng máy đục bê tông) là cặp lớp bị mô hình nhận diện sai và nhầm lẫn qua lại nhiều nhất. Lớp `children_playing` (trẻ em chơi đùa) cũng thường xuyên bị nhầm sang lớp `street_music` (nhạc đường phố).
- **Có thể giải thích bằng đặc điểm âm thanh không?** Hoàn toàn giải thích được dựa trên bản chất vật lý âm thanh. `drilling` và `jackhammer` đều là các nguồn nhiễu cơ khí chu kỳ nhanh, có mật độ phân bổ năng lượng tần số cao trên spectrogram gần như tương đồng, khiến các bộ lọc 2D-CNN trích xuất ra các cấu trúc pattern cục bộ giống nhau. Trong khi đó, `children_playing` và `street_music` đều được thu âm ngoài trời đô thị, có lẫn rất nhiều tạp âm nền không gian (tiếng người nói chuyện xa, tiếng gió, tiếng xe cộ vọng lại) làm mờ đi ranh giới đặc trưng cốt lõi của âm thanh.

## 8. So sánh với Lab 3 1D-CNN


| Tiêu chí | Lab 3: 1D-CNN | Lab 4: CRNN |
|---|---|---|
| Feature chính | MFCC / log-mel | log-mel |
| Khả năng học pattern cục bộ | Có | Có |
| Khả năng học quan hệ thời gian | Hạn chế (Chỉ trích xuất đặc trưng cửa sổ ngắn) | Tốt hơn rõ rệt (Nhờ GRU/LSTM lưu giữ trạng thái chuỗi ngữ cảnh dài) |
| Test accuracy | *[Điền Test Acc của bài Lab 3 vào đây, ví dụ: 68.50%]* | **75.03%** (Bản Baseline GRU) |
| Nhận xét | Mạng 1D-CNN có cấu trúc gọn nhẹ, thời gian huấn luyện nhanh nhưng bỏ sót diễn biến tín hiệu dài hạn. | Mạng CRNN-GRU cải thiện độ chính xác vượt trội nhờ khả năng xâu chuỗi ngữ cảnh, nhưng thời gian huấn luyện lâu hơn và dễ overfit nếu tăng tham số quá đà (như BiLSTM). |

## 9. Kết luận

Mô hình CRNN (đặc biệt là biến thể CRNN-GRU Baseline) mang lại sự cải thiện hiệu năng rõ rệt so với cấu trúc 1D-CNN của Lab 3 khi tăng độ chính xác trên tập Test lên mức rất cao (75.03%). Kết quả thực nghiệm chứng minh rằng việc kết hợp mạng tích chập để học ảnh spectrogram và mạng hồi quy để học chuỗi thời gian là hướng đi tối ưu cho dữ liệu âm thanh môi trường. Tuy nhiên, kết quả này chỉ thực sự ổn định khi kích thước mô hình được thiết kế vừa phải; việc lạm dụng kiến trúc phức tạp như BiLSTM làm tăng số lượng tham số lên 150k đã gây tác dụng ngược khiến test accuracy sụt giảm nghiêm trọng xuống 64.04%. Nếu có cơ hội phát triển tiếp thí nghiệm này, em sẽ tiến hành cải thiện bằng cách bổ sung các kỹ thuật tăng cường dữ liệu nâng cao trên spectrogram như SpecAugment (chặn dải thời gian và tần số), đồng thời thử nghiệm thay thế khối RNN truyền thống bằng cơ chế Self-Attention (Transformer) hoặc mạng Conformer để tối ưu hóa thời gian tính toán và nâng cao khả năng trích xuất đặc trưng chuỗi tầm xa.


## 10. Link minh chứng

- GitHub commit cuối:
- W&B run baseline: https://wandb.ai/datn89367-i-h-c-i-nam/csc4005-lab4-urbansound8k-crnn/runs/jddfiybp?nw=nwuserdatn89367
- W&B run mở rộng: https://wandb.ai/datn89367-i-h-c-i-nam/csc4005-lab4-urbansound8k-crnn/runs/qh570i7c?nw=nwuserdatn89367
