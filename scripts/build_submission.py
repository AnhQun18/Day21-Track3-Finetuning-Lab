from pathlib import Path
import json, csv, hashlib, subprocess, sys, os
ROOT=Path(__file__).resolve().parents[1]
def load(name): return json.loads((ROOT/'results'/name).read_text(encoding='utf-8'))
def j(x): return json.dumps(x,ensure_ascii=False)
def cell(x): return str(x).replace('|','/').replace('\n','<br>')
def table(rows,cols):
 return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(cell(r.get(c,'')) for c in cols)+' |' for r in rows)
f=load('baselines_frozen.json'); v=load('verdict.json'); a=load('autopsy.json'); q=load('qualitative.json'); p=load('mask_proof.json'); t=load('token_stats.json'); tm=load('template_check.json')
runs=list(csv.DictReader((ROOT/'results/runs.csv').open(encoding='utf-8'))); runs={r['run']:r for r in runs}
assert not f['smoke_mode'] and f['n_target']==50 and f['n_regression']==15
assert len(a)==4 and all(x['n']==50 for x in a)
assert len(q)==50 and all('baseline_b_score' in x for x in q)
assert p['answer_is_supervised'] and p['question_is_masked'] and p['supervised_fraction']<0.95
assert {int(r['max_steps']) for r in runs.values()}=={30}
assert all(int(r['max_length'])==256 for r in runs.values())
assert all(hashlib.sha256((ROOT/'data'/n).read_bytes()).hexdigest()==h for n,h in f['eval_checksums'].items())
budget=abs(int(runs['attn_only']['trainable_params'])-int(runs['correct']['trainable_params']))/int(runs['correct']['trainable_params'])
assert budget<0.05
losses=[r for r in q if r['ft_score']<r['baseline_b_score']]; wins=[r for r in q if r['ft_score']>r['baseline_b_score']]; ties=[r for r in q if r['ft_score']==r['baseline_b_score']]
sys.path.insert(0,str(ROOT/'src'))
from labkit import evaluate as ev
reg_items=[json.loads(line) for line in (ROOT/'data/eval_regression.jsonl').read_text().splitlines() if line.strip()]
bp=load('baseline_predictions.json'); fp=load('ft_predictions.json')
assert len(reg_items)==len(bp['rpreds_b'])==len(fp['rpreds_ft'])==15
reg_rows=[]
for i,(item,pb,pf) in enumerate(zip(reg_items,bp['rpreds_b'],fp['rpreds_ft'])):
 sb=ev.keyword_recall(pb,item['keywords']); sf=ev.keyword_recall(pf,item['keywords'])
 reg_rows.append(dict(i=i,group='regression',ticket=item['instruction'],ground_truth=dict(keywords=item['keywords']),baseline_b_pred=pb,ft_pred=pf,baseline_b_score=sb,ft_score=sf,outcome='FT_THUA' if sf<sb else 'FT_THANG' if sf>sb else 'HOA'))
(ROOT/'results/regression_qualitative.json').write_text(json.dumps(reg_rows,ensure_ascii=False,indent=2),encoding='utf-8')
reg_losses=[r for r in reg_rows if r['outcome']=='FT_THUA']
selected=losses[:2]+reg_losses[:max(0,2-len(losses[:2]))]+wins[:2]+ties[:1]
for r in q:
 if len(selected)<5 and r not in selected: selected.append(r)
qual=[]
for r in selected:
 try: pred=json.loads(r['ft_pred'])
 except Exception: pred={}
 bad=([kw for kw in r['ground_truth']['keywords'] if ev.keyword_recall(r['ft_pred'],[kw])<1] if r.get('group')=='regression' else [key for key,value in r['ground_truth'].items() if pred.get(key)!=value])
 qual.append({'Nhóm':r.get('group','target'),'i':r['i'],'Ticket':r['ticket'],'Ground truth':j(r['ground_truth']),'Baseline b':r['baseline_b_pred'],'Fine-tune':r['ft_pred'],'Nhận xét':r['outcome']+'; b='+str(r['baseline_b_score'])+', FT='+str(r['ft_score'])+'; trường/keyword FT thiếu: '+(', '.join(bad) or 'không')})
comparison=v['comparison']; byrun={r['run']:r for r in a}; c=runs['correct']; at=runs['attn_only']; w=runs['wrong_lr']; z=runs['qlora']
training=[]
for name in ['correct','attn_only','wrong_lr','qlora']:
 r=runs[name]; e=byrun[name]
 training.append({'Run':name,'Vị trí':r['placement'],'r':r['r'],'alpha':r['lora_alpha'],'Trainable':r['trainable_params'],'LR':r['learning_rate'],'Train loss':r['final_loss'],'Target FULL':e['target'],'Format':e['format'],'Train s':r['train_seconds'],'VRAM GB':r['peak_vram_gb'],'Latency ms':e['latency_ms']})
ranking=sorted(a,key=lambda r:r['target'],reverse=True)
from itertools import groupby
ranktext=' > '.join(' = '.join(r['run']+' ('+str(r['target'])+')' for r in group) for _,group in groupby(ranking,key=lambda r:r['target']))
placement_result='thắng' if byrun['attn_only']['target']>byrun['correct']['target'] else 'thua' if byrun['attn_only']['target']<byrun['correct']['target'] else 'hòa'
loss_rank=' < '.join(r['run']+' ('+r['final_loss']+')' for r in sorted(runs.values(),key=lambda r:float(r['final_loss'])))
loss_placement='thấp hơn' if float(at['final_loss'])<float(c['final_loss']) else 'cao hơn' if float(at['final_loss'])>float(c['final_loss']) else 'bằng'
order_agrees=(byrun['attn_only']['target']>byrun['correct']['target']) == (float(at['final_loss'])<float(c['final_loss'])) and (byrun['attn_only']['target']<byrun['correct']['target']) == (float(at['final_loss'])>float(c['final_loss']))
from collections import Counter
loss_fields=Counter()
for row in losses:
 try: lp=json.loads(row['ft_pred'])
 except Exception: lp={}
 loss_fields.update(k for k,val in row['ground_truth'].items() if lp.get(k)!=val)
quant_answer='ủng hộ việc thận trọng với QLoRA trong thí nghiệm này vì target thấp hơn correct' if byrun['qlora']['target']<byrun['correct']['target'] else 'không ủng hộ một lệnh cấm QLoRA tuyệt đối trong tác vụ này vì target không thấp hơn correct'
verdict='PASS' if v['verdict']['passed'] else 'FAIL'
ft=comparison[2]; baseline=comparison[1]
vram_saved=float(c['peak_vram_gb'])-float(z['peak_vram_gb']); vram_pct=vram_saved/float(c['peak_vram_gb'])*100
lat_pct=(ft['latency_ms']/baseline['latency_ms']-1)*100
hf_repo=os.environ.get('LAB21_HF_REPO','')
hf_link='https://huggingface.co/'+hf_repo if hf_repo else 'Chưa upload; cần đăng nhập Hugging Face và chọn repository.'
text=f'''# Lab 21 - Báo cáo đánh giá fine-tuning bằng LoRA

Ngày thực hiện: 07/10/2026. Hình thức nộp: Option B, GitHub + Hugging Face Hub. Họ tên và MSSV được để ngoài bản báo cáo kỹ thuật theo yêu cầu của người thực hiện.

## 1. Setup và quyết định trước thí nghiệm

Base model là **{f['model']}**, tier T4, GPU Tesla T4 14,6 GiB VRAM theo torch.cuda.get_device_properties. Đây là model mặc định của phiên bản repository đang chạy, phù hợp GPU Colab và có chat template đã được kiểm tra. Model dùng cho cả hai baseline và mọi adapter là cùng một checkpoint; tên model thực tế được ghi đúng theo artifact, không đổi sang tên Qwen khác trong report.

Dataset mặc định gồm 250 ticket CSKH tiếng Việt, ánh xạ sang JSON có bốn trường intent, urgency, product và sentiment. Corpus được giữ để tập trung vào thiết kế đối chứng và kiểm tra mask, tránh đưa thêm biến về nguồn dữ liệu vào thí nghiệm đầu tiên. Chia train/validation 225/25, seed 42. Đánh giá FULL gồm {f['n_target']} ticket và {f['n_regression']} câu hỏi phổ thông, không đặt EVAL_LIMIT. Validation split được tạo để tái lập; pipeline mặc định không dùng validation loss để chọn checkpoint tốt nhất.

Thống kê tokenizer: p95={t['p95']}, p99={t['p99']}, dài nhất={t['max']}, mean={t['mean']}. Chọn max_length=256 theo suggested_max_length={t['suggested_max_length']}, đủ cho toàn bộ corpus đo được. MAX_LENGTH là override rõ ràng của get_tier, giữ mặc định tier trong code và áp dụng cùng giá trị 256 cho NB1, NB3, NB4. Không có mẫu train dài hơn 256 theo số đo này.

Cấu hình chung: MASK_MODE=assistant-only; 2 epochs; 30 optimizer steps; per-device batch=1, gradient accumulation=16, effective batch=16<32. T4 chạy fp16 vì không hỗ trợ bf16; các tham số trainable bf16 được chuyển sang fp32 trước GradScaler khi cần. Packing và padding_free tắt để giữ căn chỉnh labels và vì T4 không có FlashAttention-2. Loss dùng chunked_nll, cosine schedule, 3 warmup steps.

## 2. Bằng chứng loss mask và chat template

Nguồn: results/mask_proof.json, template_check.json và token_stats.json.

- answer_is_supervised={str(p['answer_is_supervised']).lower()}.
- question_is_masked={str(p['question_is_masked']).lower()}.
- supervised_fraction={p['supervised_fraction']}; {p['n_supervised']}/{p['n_total']} token ở mẫu kiểm chứng được tính loss.
- Template check: {tm.get('verdict','xem JSON nguồn')}.

Đoạn giải mã từ labels khác -100:

~~~text
{p.get('supervised_preview','')}
~~~

Mask được tạo bằng đường pre-tokenized của labkit.data rồi nạp trực tiếp vào SFTTrainer. Không dựa riêng vào assistant_only_loss=True, vì template Qwen3.5 không có marker generation phù hợp cho đường đó. So sánh everything ở NB1 cho thấy prompt/user sẽ vào loss nếu bỏ mask. Template giữ được trace ở phép kiểm tra có trace; corpus train hiện tại là bare JSON, nên việc giữ template không đồng nghĩa model đã học suy luận. valid_trace_rate={v.get('valid_trace_rate')} không đủ để kết luận reasoning collapse trên corpus này.

## 3. Baseline FULL đóng băng trước training cuối

Đóng băng lúc {f.get('frozen_at_utc','không có timestamp')} UTC. Prompt optimized SHA={f['optimized_prompt_sha']}. Hai checksum SHA-256 của eval được lưu trong baselines_frozen.json và kiểm tra lại khi tạo báo cáo. Prompt, eval và base model không thay đổi giữa NB2 và NB3-NB5. Lượt chạy cuối chạy tuần tự NB1 -> NB2 FULL -> NB3 -> NB4 -> NB5 FULL; log gốc ở results/final_run.log. Các artifact cũ được giữ riêng ngoài thư mục results của bản cuối.

{table(comparison,['run','target','regression','format','latency_ms','n'])}

Target là trung bình accuracy của bốn trường, không phải tỷ lệ toàn bộ JSON đúng tuyệt đối. Regression là keyword recall trung bình trên 15 câu; đây là proxy hạn chế cho năng lực chung. Format kiểm tra parse JSON và đủ khóa. Latency đo theo greedy decoding và batch của harness, cần hiểu là số đo trên T4 của phiên chạy này.

Baseline b={baseline['target']} mạnh hơn baseline a={comparison[0]['target']}. Optimized prompt giữ nguyên schema và few-shot của repository; không làm yếu baseline để tăng chênh lệch của fine-tune. Baseline a thiếu hướng dẫn schema nên format thấp là một điểm yếu thực tế của prompt naive; kết luận về giá trị fine-tuning luôn dựa vào đối thủ b.

## 4. Bốn cấu hình và đối chứng NB4

{table(training,['Run','Vị trí','r','alpha','Trainable','LR','Train loss','Target FULL','Format','Train s','VRAM GB','Latency ms'])}

Trường final_loss trong runs.csv lấy từ Trainer.training_loss: đây là loss trung bình của toàn lượt, không phải loss của batch cuối. Log giữ cả loss theo bước và cảnh báo grad_norm nan. Ngân sách 30 bước là Trainer global steps; GradScaler có thể bỏ một update khi overflow, nên không khẳng định mọi bước đều cập nhật trọng số thành công.

Tất cả bốn run dùng 30 steps và cùng dataset, mask, context length, batch và seed. attn_only chỉ đổi placement sang q,v, đồng thời matched_rank nâng r={at['r']} để giữ ngân sách trainable; alpha luôn 2r. Chênh ngân sách là {budget*100:.4f}%, dưới 5%. wrong_lr chỉ giảm LR từ {c['learning_rate']} xuống {w['learning_rate']}. qlora đổi base sang quantization 4-bit và được đánh giá trên base 4-bit tương ứng; không ghép adapter QLoRA với base fp16 để chấm.

Xếp hạng theo target FULL: **{ranktext}**. Nếu điểm bằng nhau, coi là hòa về target; không dùng latency hoặc train loss để tự nhận run thắng accuracy.

### 4.1. Placement so với rank

attn_only **{placement_result}** correct trên target ({byrun['attn_only']['target']} so với {byrun['correct']['target']}). Train loss của attn_only {loss_placement} correct; thứ tự hai thước đo {'phù hợp nhau' if order_agrees else 'không giống nhau'}. Xếp thứ tự train loss từ thấp lên cao: {loss_rank}. Vì vậy, kết luận về placement phải theo bảng target, kể cả khi loss đưa ra thứ tự khác; đây là bằng chứng đo trực tiếp về placement ở ngân sách gần bằng nhau. Train loss trung bình lần lượt {at['final_loss']} và {c['final_loss']} mô tả mức tối ưu trên train, không thay thế được target FULL. Rank cao {at['r']} ở q,v là hệ quả của matched budget; không thể dùng phép so sánh này để tuyên bố rank càng cao càng tốt. Một corpus hẹp và tổng cộng 50 ticket chỉ cho phép kết luận trong phạm vi thí nghiệm; cần thêm miền ticket và nhiều seed để kết luận tổng quát về adapter placement.

### 4.2. Learning rate

wrong_lr dùng {w['learning_rate']}, thấp hơn correct một bậc độ lớn, với cùng 30 optimizer steps. Train loss trung bình của lượt {w['final_loss']} so với {c['final_loss']} cho thấy sự khác biệt về tối ưu trong cùng ngân sách, và target tương ứng là {byrun['wrong_lr']['target']} so với {byrun['correct']['target']}. Nếu chỉ thấy loss cao mà không biết LR, có thể kết luận sai rằng kiến trúc LoRA thiếu khả năng học hoặc dữ liệu không phù hợp. Log bước training cần được đọc cùng LR schedule và GradScaler; những grad_norm nan rời rạc là dấu hiệu cần theo dõi overflow, không tự động chứng minh toàn bộ run hỏng, cũng không được bỏ qua nếu kéo dài. Kết luận đúng ở đây dựa trên artifact đã lưu và score đánh giá.

### 4.3. Quantization

QLoRA tiết kiệm {vram_saved:.2f} GiB, tức {vram_pct:.1f}% peak VRAM so với correct. Train time là {z['train_seconds']} giây, so với {c['train_seconds']} giây; target FULL là {byrun['qlora']['target']} so với {byrun['correct']['target']}, latency {byrun['qlora']['latency_ms']} so với {byrun['correct']['latency_ms']} ms/mẫu. Giảm VRAM không tự động đi kèm tăng tốc hoặc giữ nguyên chất lượng; số đo này là trade-off cần cân nhắc theo giới hạn phần cứng. Phép đo trên corpus CSKH nhỏ không đủ để xác nhận một khuyến nghị cho mọi tác vụ Qwen3.5; chỉ hỗ trợ quyết định trong cấu hình và dữ liệu đã ghi lại. Số đo {quant_answer}; chưa đủ để bác bỏ hoặc khẳng định khuyến nghị trên mọi tác vụ của dòng model. QLoRA vẫn là lựa chọn kỹ thuật hữu ích khi bộ nhớ là ràng buộc chính, nhưng cần kiểm tra chất lượng trước khi dùng.

## 5. Phán quyết và ý nghĩa triển khai

Verdict của regression gate: **{verdict}**. Target delta={v['verdict']['target_delta']:+.4f}; regression delta={v['verdict']['regression_delta']:+.4f}. Lý do từ gate: {'; '.join(v['verdict']['reasons'])}.

Fine-tune đạt target {ft['target']}, so với {baseline['target']} của base model đã prompt tốt. Đây là lợi ích về tác vụ chuyên biệt, cần đặt cạnh regression {ft['regression']} so với {baseline['regression']}, format {ft['format']} và latency {ft['latency_ms']} ms/mẫu. Latency thay đổi {lat_pct:+.1f}% so với baseline b. Kết quả vì vậy không thể được diễn giải chỉ bằng accuracy ticket: một model dùng cho hội thoại chung cần giữ năng lực nền, trong khi hệ thống triage còn có yêu cầu đáp ứng và độ tin cậy của JSON. Gate giữ nguyên các ngưỡng của repository; không nới tiêu chí, không bỏ câu regression khó và không thay prompt sau khi thấy kết quả. {('Gate FAIL cho thấy cải thiện triage chưa bù được sự suy giảm năng lực chung; không nên dùng adapter này thay base model cho trợ lý đa năng. Ưu tiên thử replay 1–5% dữ liệu phổ thông, rồi đánh giá lại bằng cùng bộ đo đã đóng băng.' if not v['verdict']['passed'] else 'Gate PASS xác nhận các ngưỡng của lab được đáp ứng trên corpus hiện tại; vẫn cần validation độc lập và dữ liệu khách hàng thực trước khi triển khai.')}  15 câu keyword recall là tập nhỏ; giảm điểm gợi ý regression nhưng không đủ để đo mọi dạng quên kiến thức. Báo cáo tách bạch phán quyết của model với gatekeeper kiểm tra tính đầy đủ bài nộp.

## 6. Đánh giá định tính có đối chiếu baseline b

Trên toàn bộ 50 ticket: FT thắng {len(wins)}, thua {len(losses)}, hòa {len(ties)} theo accuracy từng trường. Cả output baseline b lẫn fine-tune được giữ đầy đủ trong qualitative.json và prediction JSON, không cắt 90 ký tự như bảng in nhanh của notebook gốc. Chọn ca thua trước, sau đó ca thắng và hòa để tránh cherry-pick. Tập target không có đủ hai ca FT thua nên bảng bổ sung ca thua từ 15 câu regression đã có trong FULL eval, có cột Nhóm để phân biệt. Không tạo eval mới hay sửa nhãn để làm xuất hiện ca thua.

{table(qual,['Nhóm','i','Ticket','Ground truth','Baseline b','Fine-tune','Nhận xét'])}

Các trường sai trong các ca FT thua (đếm trên toàn bộ target): {dict(loss_fields)}. Trên target không có ca thua tương đối so với b. Hai ca regression chọn trong bảng cho thấy cùng một mẫu: fine-tune áp hành vi xuất JSON triage vào cả câu hỏi đổi đơn vị và yêu cầu chúc sinh nhật, làm mất câu trả lời mà instruction cần. Đây là dấu hiệu quá chuyên biệt về hành vi; keyword recall chỉ là proxy, không đủ để khẳng định mọi kiến thức nội tại đã bị xóa. Các ca thua được xác định bằng FT score thấp hơn baseline b trên cùng ticket, không gọi một ca FT=0,75 là thua nếu baseline cũng=0,75. Cột nhận xét chỉ rõ trường sai của FT; cần phân biệt lỗi urgency/sentiment với lỗi nhận diện sản phẩm hay intent. Dataset tổng hợp có các tín hiệu từ vựng ngắn, nên quy tắc về nhãn và cách diễn đạt có thể tạo lỗi ở những ticket thiếu dấu hiệu rõ ràng. Không lấy lỗi đơn lẻ làm bằng chứng về mọi khách hàng thật; đây là dữ liệu để đề xuất bổ sung ví dụ và kiểm tra ngoài miền.

Số ca FT thua trong bảng: {sum(r['outcome']=='FT_THUA' for r in selected)}. {'Bảng có ít nhất hai ca FT thua thực tế.' if sum(r['outcome']=='FT_THUA' for r in selected)>=2 else 'Không có đủ hai ca thua trong eval gốc; ghi nhận trung thực.'} Số ca regression thua trong toàn bộ 15 câu: {len(reg_losses)}.

## 7. Kết luận và điều học được

### Kết luận

Quyết định triển khai phải xuất phát từ mục tiêu sử dụng. Với một hệ thống chỉ nhận ticket và xuất JSON, lợi ích của fine-tune là chuyển một phần hướng dẫn dài vào adapter, cải thiện khả năng ánh xạ nhãn và giữ output có cấu trúc với prompt ngắn hơn. Tuy nhiên, trong bài này baseline optimized đã là một đối thủ rõ ràng và mạnh hơn prompt naive; không thể lấy chiến thắng trước prompt yếu để chứng minh fine-tuning cần thiết. Kết quả FULL cho thấy target, regression, format và latency phải được đọc cùng nhau. {("Với verdict FAIL, tôi không deploy bản này như một assistant đa năng; cần thử replay và đánh giá lại trước." if not v["verdict"]["passed"] else "Với verdict PASS, chỉ cân nhắc thử nghiệm triage có giám sát và validation độc lập.")} Verdict {verdict} cùng các ca thua định tính là cơ sở để đánh giá giới hạn của adapter, thay vì chỉ nhìn loss giảm. Tôi chưa đề xuất thay thế một assistant phổ thông bằng bản LoRA này nếu regression gate không đạt; phương án đáng cân nhắc là giữ base cùng optimized prompt cho năng lực chung và thử adapter trong một tuyến triage chuyên biệt có validation dữ liệu thật. Ngay cả khi gate đạt, vẫn cần đánh giá ticket ngoài corpus tổng hợp, thay đổi cách diễn đạt, phủ định và sản phẩm mới trước khi deploy. Đòn bẩy đầu tiên là loss mask và dữ liệu: nếu prompt bị tính loss hoặc nhãn thiếu nhất quán thì điều chỉnh rank không giải quyết được nguyên nhân. NB4 bổ sung bằng chứng về learning rate và vị trí adapter với ngân sách tham số được kiểm soát. QLoRA giúp giảm bộ nhớ nhưng phải trả giá theo số đo chất lượng và thời gian của chính run. Các thí nghiệm tiếp theo cần đăng ký trước cấu hình và tiêu chí, giữ eval độc lập, bổ sung một lượng dữ liệu replay hợp lý nếu regression là vấn đề, rồi báo cáo cả thắng lẫn thua mà không ép kết quả PASS.

### Ba điều tôi học được từ số đo

1. Tôi cần phân biệt tối ưu train với năng lực tác vụ: correct loss={c['final_loss']}, target={byrun['correct']['target']}; attn_only loss={at['final_loss']}, target={byrun['attn_only']['target']}. Chọn adapter theo loss thay vì target sẽ bỏ qua câu hỏi mà lab cần trả lời.
2. Mask phải được chứng minh bằng token labels: supervised_fraction={p['supervised_fraction']} và hai assert đúng là bằng chứng cụ thể, trong khi assistant_only_loss chỉ là một cờ có thể phụ thuộc chat template.
3. Một kết quả target tốt chưa đủ để triển khai: delta target={v['verdict']['target_delta']:+.4f} cần được đối chiếu delta regression={v['verdict']['regression_delta']:+.4f} và latency thay đổi {lat_pct:+.1f}%. Chạy smoke 8 mẫu không bảo đảm kết luận giữ nguyên ở 50/15 mẫu FULL.

Nếu có thêm hai giờ, tôi sẽ thiết kế trước một thử nghiệm replay 1-5% dữ liệu phổ thông trên train, giữ nguyên eval cũ làm mốc và thêm một tập đánh giá mới độc lập. Tôi cũng sẽ đo nhiều seed và sai số quanh target thay vì chỉ so một con số duy nhất; không chỉnh criterion để cứu verdict.

## 8. Option B và khả năng tái lập

- GitHub: https://github.com/AnhQun18/Day21-Track3-Finetuning-Lab
- Hugging Face adapter: {hf_link}
- Các file results, report, code và LINKS.md cần có trong repository nộp; adapter correct đặt trên Hub công khai.
- NB6, custom dataset, reasoning-trace ablation và rank sweep chưa thực hiện; không khai bonus tương ứng.

Code bổ sung lưu checksum/timestamp và prediction nguyên văn, hỗ trợ MAX_LENGTH theo p95 và sửa test chọn setup cell theo cell_type thay vì chỉ số cố định. Không sửa hàm chấm điểm hoặc tiêu chí regression gate. Notebook .ipynb được tạo từ script .py để tái lập; không có token trong file nộp.
'''
(ROOT/'submission/REPORT.md').write_text(text,encoding='utf-8')
log_path=Path('/content/lab21_final_run.log')
if log_path.exists(): (ROOT/'results/final_run.log').write_text(log_path.read_text(),encoding='utf-8')
from datetime import datetime,timezone
manifest={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'base_model':f['model'],'max_length':256,'epochs':2,'mask_mode':'assistant-only','full_target':50,'full_regression':15,'qualitative_target_losses':len(losses),'qualitative_regression_losses':len(reg_losses),'qualitative_selected_losses':sum(x['outcome']=='FT_THUA' for x in selected),'qualitative_selected':len(selected),'parameter_budget_difference_fraction':budget,'baseline_frozen_at_utc':f['frozen_at_utc'],'eval_checksums':f['eval_checksums'],'result_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'results').glob('*') if p.suffix in ['.json','.csv'] and p.name not in ['reproducibility.json','hub_upload.json']}}
manifest['adapter_correct']={'sha256':hashlib.sha256((ROOT/'adapters/correct/adapter_model.safetensors').read_bytes()).hexdigest(),'bytes':(ROOT/'adapters/correct/adapter_model.safetensors').stat().st_size}
manifest['checkout_commit_at_export']=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
manifest['source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['src/labkit/config.py','src/labkit/data.py','src/labkit/evaluate.py','src/labkit/generate.py','src/labkit/modeling.py','src/labkit/train.py','notebooks/01_data_and_mask.py','notebooks/02_baselines.py','notebooks/03_train_correct.py','notebooks/04_misconfig_autopsy.py','notebooks/05_evaluate_and_verdict.py','requirements.txt','scripts/build_submission.py','scripts/build_colab.py']}
from importlib.metadata import version
manifest['package_versions']={name:version(name) for name in ['torch','transformers','trl','peft','accelerate','datasets','bitsandbytes','torchao']}
(ROOT/'results/reproducibility.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
(ROOT/'LINKS.md').write_text('# Lab21 - Option B\n\nGitHub: https://github.com/AnhQun18/Day21-Track3-Finetuning-Lab\n\nHugging Face: '+hf_link+'\n')
card=f'''---
base_model: {f['model']}
library_name: peft
language:
- vi
tags:
- lora
- ticket-triage
- text-generation
---
# Lab21 - Qwen3.5-4B LoRA for Vietnamese customer-support triage

This is the correct text-decoder LoRA adapter from Lab21. Training used 225 synthetic Vietnamese customer-support tickets, rank 16, alpha 32, learning rate 1e-4, assistant-only labels, max_length 256, two epochs, and 30 optimizer steps on Tesla T4 with fp16 computation.

## Evaluation

Full frozen evaluation: 50 target tickets and 15 general-capability prompts. Target is mean four-field accuracy; regression is keyword recall. Greedy generation uses the same lab harness.

{table(comparison,['run','target','regression','format','latency_ms','n'])}

Regression gate verdict: **{verdict}**. Reasons: {'; '.join(v['verdict']['reasons'])}. This is a lab artifact; the evaluation does not establish readiness for production or general assistant replacement. See submission/REPORT.md and results/ for all evidence, qualitative errors, and trade-offs.

## Loading

Load base checkpoint {f['model']} with the corresponding tokenizer, then load this repository with PeftModel.from_pretrained. Use system prompt "Phân loại ticket sau." and greedy decoding, with the model's chat template and thinking disabled as in labkit.generate. This repository contains an adapter rather than a merged base model.

## Submission

GitHub source: https://github.com/AnhQun18/Day21-Track3-Finetuning-Lab
Hugging Face: {hf_link}

No custom corpus, NB6, reasoning-trace ablation, or rank sweep bonus is claimed.
'''
(ROOT/'adapters/correct/README.md').write_text(card,encoding='utf-8')
print('REPORT_WRITTEN',len(text.split()),'words; losses:',len(losses),'selected:',len(selected))
print('FINAL_VERDICT',verdict)
