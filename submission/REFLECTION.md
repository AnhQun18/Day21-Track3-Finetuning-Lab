# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

LoRA tăng điểm target từ 0.765 lên 0.970 nhưng làm điểm regression giảm từ 0.7911 xuống 0.4778. Mô hình làm tốt hơn việc phân loại ticket nhưng lại trả JSON phân loại cho một số câu hỏi phổ thông. Cải thiện tác vụ chính chưa đủ để kết luận fine-tuning thành công.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Phần chạy đủ bốn cấu hình và đánh giá FULL mất khoảng 54 phút. Ngoài training, việc xử lý Colab mất kết nối, kiểm tra artifact cũng tốn thời gian.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi từng nghĩ loss thấp hơn thì mô hình sẽ tốt hơn. Tuy nhiên, attn_only có loss 0.5378, thấp hơn correct 0.6260, nhưng cả hai cùng đạt target 0.970. Tôi cũng không còn xem điểm target tăng là đủ; cần kiểm tra regression và so với baseline dùng prompt tối ưu.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng AI assistant để xử lý lỗi, tổng hợp kết quả, viết báo cáo. Một lỗi trong quy trình là biến MAX_LENGTH=256 ảnh hưởng đến test kiểm tra mặc định 1024; assistant đã sửa bằng cách cô lập biến môi trường trong test. Nó cũng thử sao lưu adapter qua output trình duyệt, khiến kết nối không ổn định; sau đó chuyển sang upload trực tiếp từ Colab lên Hugging Face.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Tôi sẽ thống nhất tác vụ, tiêu chí thành công và những năng lực không được suy giảm, rồi xây dựng tập đánh giá độc lập gồm cả tình huống thực tế và regression. Tôi sẽ đo baseline với prompt tối ưu trước khi quyết định fine-tune, thay vì bắt đầu ngay bằng training.
