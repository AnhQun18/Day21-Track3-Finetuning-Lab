# Lab 21 - Báo cáo đánh giá fine-tuning bằng LoRA

Ngày thực hiện: 07/10/2026. Hình thức nộp: Option B, GitHub + Hugging Face Hub. 

Họ tên: **Trần Anh Quân**  
Mã học viên: **2A202602598**.

## 1. Setup và quyết định trước thí nghiệm

Base model là **unsloth/Qwen3.5-4B**, tier T4, GPU Tesla T4 14,6 GiB VRAM theo torch.cuda.get_device_properties. Đây là model mặc định của phiên bản repository đang chạy, phù hợp GPU Colab và có chat template đã được kiểm tra. Model dùng cho cả hai baseline và mọi adapter là cùng một checkpoint; tên model thực tế được ghi đúng theo artifact, không đổi sang tên Qwen khác trong report.

Dataset mặc định gồm 250 ticket CSKH tiếng Việt, ánh xạ sang JSON có bốn trường intent, urgency, product và sentiment. Corpus được giữ để tập trung vào thiết kế đối chứng và kiểm tra mask, tránh đưa thêm biến về nguồn dữ liệu vào thí nghiệm đầu tiên. Chia train/validation 225/25, seed 42. Đánh giá FULL gồm 50 ticket và 15 câu hỏi phổ thông, không đặt EVAL_LIMIT. Validation split được tạo để tái lập; pipeline mặc định không dùng validation loss để chọn checkpoint tốt nhất.

Thống kê tokenizer: p95=98, p99=100, dài nhất=101, mean=93.1. Chọn max_length=256 theo suggested_max_length=256, đủ cho toàn bộ corpus đo được. MAX_LENGTH là override rõ ràng của get_tier, giữ mặc định tier trong code và áp dụng cùng giá trị 256 cho NB1, NB3, NB4. Không có mẫu train dài hơn 256 theo số đo này.

Cấu hình chung: MASK_MODE=assistant-only; 2 epochs; 30 optimizer steps; per-device batch=1, gradient accumulation=16, effective batch=16<32. T4 chạy fp16 vì không hỗ trợ bf16; các tham số trainable bf16 được chuyển sang fp32 trước GradScaler khi cần. Packing và padding_free tắt để giữ căn chỉnh labels và vì T4 không có FlashAttention-2. Loss dùng chunked_nll, cosine schedule, 3 warmup steps.

## 2. Bằng chứng loss mask và chat template

Nguồn: results/mask_proof.json, template_check.json và token_stats.json.

- answer_is_supervised=true.
- question_is_masked=true.
- supervised_fraction=0.4149; 39/94 token ở mẫu kiểm chứng được tính loss.
- Template check: reasoning preserved — safe to train on traces.

Đoạn giải mã từ labels khác -100:

~~~text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>

~~~

Mask được tạo bằng đường pre-tokenized của labkit.data rồi nạp trực tiếp vào SFTTrainer. Không dựa riêng vào assistant_only_loss=True, vì template Qwen3.5 không có marker generation phù hợp cho đường đó. So sánh everything ở NB1 cho thấy prompt/user sẽ vào loss nếu bỏ mask. Template giữ được trace ở phép kiểm tra có trace; corpus train hiện tại là bare JSON, nên việc giữ template không đồng nghĩa model đã học suy luận. valid_trace_rate=0.0 không đủ để kết luận reasoning collapse trên corpus này.

## 3. Baseline FULL đóng băng trước training cuối

Đóng băng lúc 2026-10-07T04:13:16.298593+00:00 UTC. Prompt optimized SHA=719e74d3b6232053. Hai checksum SHA-256 của eval được lưu trong baselines_frozen.json và kiểm tra lại khi tạo báo cáo. Prompt, eval và base model không thay đổi giữa NB2 và NB3-NB5. Lượt chạy cuối chạy tuần tự NB1 -> NB2 FULL -> NB3 -> NB4 -> NB5 FULL; log gốc ở results/final_run.log. Các artifact cũ được giữ riêng ngoài thư mục results của bản cuối.

| run | target | regression | format | latency_ms | n |
| --- | --- | --- | --- | --- | --- |
| (a) base + naive prompt | 0.0 | 0.7911 | 0.0 | 3648.9 | 50 |
| (b) base + optimized prompt | 0.765 | 0.7911 | 1.0 | 1124.4 | 50 |
| (c) LoRA fine-tune | 0.97 | 0.4778 | 1.0 | 1696.4 | 50 |

Target là trung bình accuracy của bốn trường, không phải tỷ lệ toàn bộ JSON đúng tuyệt đối. Regression là keyword recall trung bình trên 15 câu; đây là proxy hạn chế cho năng lực chung. Format kiểm tra parse JSON và đủ khóa. Latency đo theo greedy decoding và batch của harness, cần hiểu là số đo trên T4 của phiên chạy này.

Baseline b=0.765 mạnh hơn baseline a=0.0. Optimized prompt giữ nguyên schema và few-shot của repository; không làm yếu baseline để tăng chênh lệch của fine-tune. Baseline a thiếu hướng dẫn schema nên format thấp là một điểm yếu thực tế của prompt naive; kết luận về giá trị fine-tuning luôn dựa vào đối thủ b.

## 4. Bốn cấu hình và đối chứng NB4

| Run | Vị trí | r | alpha | Trainable | LR | Train loss | Target FULL | Format | Train s | VRAM GB | Latency ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| correct | text-linear | 16 | 32 | 32464896 | 0.0001 | 0.626 | 0.97 | 1.0 | 458.9 | 8.78 | 1696.4 |
| attn_only | attn-only | 283 | 566 | 32456704 | 0.0001 | 0.5378 | 0.97 | 1.0 | 317.8 | 8.79 | 1036.8 |
| wrong_lr | text-linear | 16 | 32 | 32464896 | 1e-05 | 1.5702 | 0.0 | 0.0 | 476.9 | 8.78 | 6203.8 |
| qlora | text-linear | 16 | 32 | 32464896 | 0.0001 | 0.7058 | 0.94 | 1.0 | 574.0 | 3.86 | 2046.4 |

Trường final_loss trong runs.csv lấy từ Trainer.training_loss: đây là loss trung bình của toàn lượt, không phải loss của batch cuối. Log giữ cả loss theo bước và cảnh báo grad_norm nan. Ngân sách 30 bước là Trainer global steps; GradScaler có thể bỏ một update khi overflow, nên không khẳng định mọi bước đều cập nhật trọng số thành công.

Tất cả bốn run dùng 30 steps và cùng dataset, mask, context length, batch và seed. attn_only chỉ đổi placement sang q,v, đồng thời matched_rank nâng r=283 để giữ ngân sách trainable; alpha luôn 2r. Chênh ngân sách là 0.0252%, dưới 5%. wrong_lr chỉ giảm LR từ 0.0001 xuống 1e-05. qlora đổi base sang quantization 4-bit và được đánh giá trên base 4-bit tương ứng; không ghép adapter QLoRA với base fp16 để chấm.

Xếp hạng theo target FULL: **correct (0.97) = attn_only (0.97) > qlora (0.94) > wrong_lr (0.0)**. Nếu điểm bằng nhau, coi là hòa về target; không dùng latency hoặc train loss để tự nhận run thắng accuracy.

### 4.1. Placement so với rank

attn_only **hòa** correct trên target (0.97 so với 0.97). Train loss của attn_only thấp hơn correct; thứ tự hai thước đo không giống nhau. Xếp thứ tự train loss từ thấp lên cao: attn_only (0.5378) < correct (0.626) < qlora (0.7058) < wrong_lr (1.5702). Vì vậy, kết luận về placement phải theo bảng target, kể cả khi loss đưa ra thứ tự khác; đây là bằng chứng đo trực tiếp về placement ở ngân sách gần bằng nhau. Train loss trung bình lần lượt 0.5378 và 0.626 mô tả mức tối ưu trên train, không thay thế được target FULL. Rank cao 283 ở q,v là hệ quả của matched budget; không thể dùng phép so sánh này để tuyên bố rank càng cao càng tốt. Một corpus hẹp và tổng cộng 50 ticket chỉ cho phép kết luận trong phạm vi thí nghiệm; cần thêm miền ticket và nhiều seed để kết luận tổng quát về adapter placement.

### 4.2. Learning rate

wrong_lr dùng 1e-05, thấp hơn correct một bậc độ lớn, với cùng 30 optimizer steps. Train loss trung bình của lượt 1.5702 so với 0.626 cho thấy sự khác biệt về tối ưu trong cùng ngân sách, và target tương ứng là 0.0 so với 0.97. Nếu chỉ thấy loss cao mà không biết LR, có thể kết luận sai rằng kiến trúc LoRA thiếu khả năng học hoặc dữ liệu không phù hợp. Log bước training cần được đọc cùng LR schedule và GradScaler; những grad_norm nan rời rạc là dấu hiệu cần theo dõi overflow, không tự động chứng minh toàn bộ run hỏng, cũng không được bỏ qua nếu kéo dài. Kết luận đúng ở đây dựa trên artifact đã lưu và score đánh giá.

### 4.3. Quantization

QLoRA tiết kiệm 4.92 GiB, tức 56.0% peak VRAM so với correct. Train time là 574.0 giây, so với 458.9 giây; target FULL là 0.94 so với 0.97, latency 2046.4 so với 1696.4 ms/mẫu. Giảm VRAM không tự động đi kèm tăng tốc hoặc giữ nguyên chất lượng; số đo này là trade-off cần cân nhắc theo giới hạn phần cứng. Phép đo trên corpus CSKH nhỏ không đủ để xác nhận một khuyến nghị cho mọi tác vụ Qwen3.5; chỉ hỗ trợ quyết định trong cấu hình và dữ liệu đã ghi lại. Số đo ủng hộ việc thận trọng với QLoRA trong thí nghiệm này vì target thấp hơn correct; chưa đủ để bác bỏ hoặc khẳng định khuyến nghị trên mọi tác vụ của dòng model. QLoRA vẫn là lựa chọn kỹ thuật hữu ích khi bộ nhớ là ràng buộc chính, nhưng cần kiểm tra chất lượng trước khi dùng.

## 5. Phán quyết và ý nghĩa triển khai

Verdict của regression gate: **FAIL**. Target delta=+0.2050; regression delta=-0.3133. Lý do từ gate: general capability regressed by 0.313 (tolerance 0.020). See deck §6.3 — add 1-5% replay data..

Fine-tune đạt target 0.97, so với 0.765 của base model đã prompt tốt. Đây là lợi ích về tác vụ chuyên biệt, cần đặt cạnh regression 0.4778 so với 0.7911, format 1.0 và latency 1696.4 ms/mẫu. Latency thay đổi +50.9% so với baseline b. Kết quả vì vậy không thể được diễn giải chỉ bằng accuracy ticket: một model dùng cho hội thoại chung cần giữ năng lực nền, trong khi hệ thống triage còn có yêu cầu đáp ứng và độ tin cậy của JSON. Gate giữ nguyên các ngưỡng của repository; không nới tiêu chí, không bỏ câu regression khó và không thay prompt sau khi thấy kết quả. Gate FAIL cho thấy cải thiện triage chưa bù được sự suy giảm năng lực chung; không nên dùng adapter này thay base model cho trợ lý đa năng. Ưu tiên thử replay 1–5% dữ liệu phổ thông, rồi đánh giá lại bằng cùng bộ đo đã đóng băng.  15 câu keyword recall là tập nhỏ; giảm điểm gợi ý regression nhưng không đủ để đo mọi dạng quên kiến thức. Báo cáo tách bạch phán quyết của model với gatekeeper kiểm tra tính đầy đủ bài nộp.

## 6. Đánh giá định tính có đối chiếu baseline b

Trên toàn bộ 50 ticket: FT thắng 33, thua 0, hòa 17 theo accuracy từng trường. Cả output baseline b lẫn fine-tune được giữ đầy đủ trong qualitative.json và prediction JSON, không cắt 90 ký tự như bảng in nhanh của notebook gốc. Chọn ca thua trước, sau đó ca thắng và hòa để tránh cherry-pick. Tập target không có đủ hai ca FT thua nên bảng bổ sung ca thua từ 15 câu regression đã có trong FULL eval, có cột Nhóm để phân biệt. Không tạo eval mới hay sửa nhãn để làm xuất hiện ca thua.

| Nhóm | i | Ticket | Ground truth | Baseline b | Fine-tune | Nhận xét |
| --- | --- | --- | --- | --- | --- | --- |
| regression | 2 | 1 km bằng bao nhiêu mét? | {"keywords": ["1000"]} | Trong hệ đo lường quốc tế, tiền tố "kilo" (ký hiệu là **km**) có nghĩa là **1000**. Do đó, 1 kilômét tương đương với 1000 mét.<br><br>Kết quả: **1 km = 1000 m**. | {"intent": "hoi_thong_tin", "urgency": "thap", "product": null, "sentiment": "trung_tinh"} | FT_THUA; b=1.0, FT=0.0; trường/keyword FT thiếu: 1000 |
| regression | 3 | Viết một câu chúc mừng sinh nhật bằng tiếng Việt. | {"keywords": ["sinh nhật"]} | Chúc bạn một ngày sinh nhật thật vui vẻ, tràn đầy niềm vui và sức khỏe, cùng với những điều may mắn và hạnh phúc nhất trong năm mới! | {"intent": "chuc_mung_sinh_nhat", "urgency": "trung_tinh", "tone": "trung_thinh", "product": null, "sentiment": "tich_cuc"} | FT_THUA; b=1.0, FT=0.0; trường/keyword FT thiếu: sinh nhật |
| target | 0 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Gấp. Shop hỗ trợ tốt. | {"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"} | {"intent": "hoan_tien", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"} | {"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"} | FT_THANG; b=0.75, FT=1.0; trường/keyword FT thiếu: không |
| target | 1 | Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhé. Bực mình. | {"intent": "hoan_tien", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "tieu_cuc"} | {"intent": "hoan_tien", "urgency": "cao", "product": "ốp lưng điện thoại", "sentiment": "tieu_cuc"} | {"intent": "hoan_tien", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "tieu_cuc"} | FT_THANG; b=0.75, FT=1.0; trường/keyword FT thiếu: không |
| target | 2 | Xin chào, mình đặt đèn bàn LED mã đơn VN880807. Hoàn tiền. Quá hạn rồi. Cảm ơn shop nhiều. | {"intent": "hoan_tien", "urgency": "cao", "product": "đèn bàn LED", "sentiment": "tich_cuc"} | {"intent": "hoan_tien", "urgency": "cao", "product": "đèn bàn LED", "sentiment": "tich_cuc"} | {"intent": "hoan_tien", "urgency": "cao", "product": "đèn bàn LED", "sentiment": "tich_cuc"} | HOA; b=1.0, FT=1.0; trường/keyword FT thiếu: không |

Các trường sai trong các ca FT thua (đếm trên toàn bộ target): {}. Trên target không có ca thua tương đối so với b. Hai ca regression chọn trong bảng cho thấy cùng một mẫu: fine-tune áp hành vi xuất JSON triage vào cả câu hỏi đổi đơn vị và yêu cầu chúc sinh nhật, làm mất câu trả lời mà instruction cần. Đây là dấu hiệu quá chuyên biệt về hành vi; keyword recall chỉ là proxy, không đủ để khẳng định mọi kiến thức nội tại đã bị xóa. Các ca thua được xác định bằng FT score thấp hơn baseline b trên cùng ticket, không gọi một ca FT=0,75 là thua nếu baseline cũng=0,75. Cột nhận xét chỉ rõ trường sai của FT; cần phân biệt lỗi urgency/sentiment với lỗi nhận diện sản phẩm hay intent. Dataset tổng hợp có các tín hiệu từ vựng ngắn, nên quy tắc về nhãn và cách diễn đạt có thể tạo lỗi ở những ticket thiếu dấu hiệu rõ ràng. Không lấy lỗi đơn lẻ làm bằng chứng về mọi khách hàng thật; đây là dữ liệu để đề xuất bổ sung ví dụ và kiểm tra ngoài miền.

Số ca FT thua trong bảng: 2. Bảng có ít nhất hai ca FT thua thực tế. Số ca regression thua trong toàn bộ 15 câu: 6.

## 7. Kết luận và điều học được

### Kết luận

Quyết định triển khai phải xuất phát từ mục tiêu sử dụng. Với một hệ thống chỉ nhận ticket và xuất JSON, lợi ích của fine-tune là chuyển một phần hướng dẫn dài vào adapter, cải thiện khả năng ánh xạ nhãn và giữ output có cấu trúc với prompt ngắn hơn. Tuy nhiên, trong bài này baseline optimized đã là một đối thủ rõ ràng và mạnh hơn prompt naive; không thể lấy chiến thắng trước prompt yếu để chứng minh fine-tuning cần thiết. Kết quả FULL cho thấy target, regression, format và latency phải được đọc cùng nhau. Với verdict FAIL, tôi không deploy bản này như một assistant đa năng; cần thử replay và đánh giá lại trước. Verdict FAIL cùng các ca thua định tính là cơ sở để đánh giá giới hạn của adapter, thay vì chỉ nhìn loss giảm. Tôi chưa đề xuất thay thế một assistant phổ thông bằng bản LoRA này nếu regression gate không đạt; phương án đáng cân nhắc là giữ base cùng optimized prompt cho năng lực chung và thử adapter trong một tuyến triage chuyên biệt có validation dữ liệu thật. Ngay cả khi gate đạt, vẫn cần đánh giá ticket ngoài corpus tổng hợp, thay đổi cách diễn đạt, phủ định và sản phẩm mới trước khi deploy. Đòn bẩy đầu tiên là loss mask và dữ liệu: nếu prompt bị tính loss hoặc nhãn thiếu nhất quán thì điều chỉnh rank không giải quyết được nguyên nhân. NB4 bổ sung bằng chứng về learning rate và vị trí adapter với ngân sách tham số được kiểm soát. QLoRA giúp giảm bộ nhớ nhưng phải trả giá theo số đo chất lượng và thời gian của chính run. Các thí nghiệm tiếp theo cần đăng ký trước cấu hình và tiêu chí, giữ eval độc lập, bổ sung một lượng dữ liệu replay hợp lý nếu regression là vấn đề, rồi báo cáo cả thắng lẫn thua mà không ép kết quả PASS.

### Ba điều tôi học được từ số đo

1. Tôi cần phân biệt tối ưu train với năng lực tác vụ: correct loss=0.626, target=0.97; attn_only loss=0.5378, target=0.97. Chọn adapter theo loss thay vì target sẽ bỏ qua câu hỏi mà lab cần trả lời.
2. Mask phải được chứng minh bằng token labels: supervised_fraction=0.4149 và hai assert đúng là bằng chứng cụ thể, trong khi assistant_only_loss chỉ là một cờ có thể phụ thuộc chat template.
3. Một kết quả target tốt chưa đủ để triển khai: delta target=+0.2050 cần được đối chiếu delta regression=-0.3133 và latency thay đổi +50.9%. Chạy smoke 8 mẫu không bảo đảm kết luận giữ nguyên ở 50/15 mẫu FULL.

Nếu có thêm hai giờ, tôi sẽ thiết kế trước một thử nghiệm replay 1-5% dữ liệu phổ thông trên train, giữ nguyên eval cũ làm mốc và thêm một tập đánh giá mới độc lập. Tôi cũng sẽ đo nhiều seed và sai số quanh target thay vì chỉ so một con số duy nhất; không chỉnh criterion để cứu verdict.

## 8. Option B và khả năng tái lập

- GitHub: https://github.com/AnhQun18/Day21-Track3-Finetuning-Lab
- Hugging Face adapter: https://huggingface.co/QunAnh/Lab21-LoRA-CSKH
- Các file results, report, code và LINKS.md cần có trong repository nộp; adapter correct đặt trên Hub công khai.
- NB6, custom dataset, reasoning-trace ablation và rank sweep chưa thực hiện; không khai bonus tương ứng.

Code bổ sung lưu checksum/timestamp và prediction nguyên văn, hỗ trợ MAX_LENGTH theo p95 và sửa test chọn setup cell theo cell_type thay vì chỉ số cố định. Không sửa hàm chấm điểm hoặc tiêu chí regression gate. Notebook .ipynb được tạo từ script .py để tái lập; không có token trong file nộp.
