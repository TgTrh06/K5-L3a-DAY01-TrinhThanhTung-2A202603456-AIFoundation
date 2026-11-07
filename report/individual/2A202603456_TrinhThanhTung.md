# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603456
- **Họ và tên:** TRỊNH THANH TÙNG
- **SEED** (in ở ô đầu notebook): 9146 · Thử thách cá nhân: **#3**

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước (bài 1–5) | ☑ |

## 2. Kết quả chính

| Phương án | precision | recall | F1 (test) | F1 (SUB 40) | train (s) | tin/s (lô) | p50 ms | p95 ms | USD/1k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg | 0.986 | 0.802 | 0.885 | 0.947 | 2.1 | 31 520 | 0.07 | **0.09** | 0 |
| 2. BoW + MLP | 0.988 | 0.942 | **0.964** | 1.000 | 4.1 | 36 828 | 0.26 | **0.31** | 0 |
| 3a. Encoder + kNN | 0.888 | 0.919 | 0.903 | 0.976 | 26.2 | 39 | 41.0 | 44.1 | 0 |
| 3b. Encoder + LogReg | 0.920 | 0.942 | 0.931 | 0.976 | 26.6 | 103 | 22.9 | 25.4 | 0 |
| 3c. Encoder + MLP head | 0.930 | 0.930 | 0.930 | 1.000 | 27.8 | 102 | 23.5 | 27.3 | 0 |
| 4a. LLM nhỏ zero-shot (`gpt-6-luna`) | 1.000 | 0.950 | — | 0.974 | 0 | 0.74 | 1 084 | 2 705 | 0.007 |
| 4b. LLM lớn zero-shot (`gpt-6.1-terra`) | 1.000 | 1.000 | — | **1.000** | 0 | 0.61 | 1 301 | 3 356 | 0.070 |
| 4c. LLM nhỏ few-shot | 1.000 | 0.950 | — | 0.974 | 0 | 0.56 | 1 036 | 5 446 | 0.023 |

Hàng LLM chỉ đo trên SUB 40 tin (precision/recall của hàng 4 là trên SUB).
Đường Pareto (SUB): BoW + LogReg → BoW + MLP. Đạt p95 < 50 ms: hàng 1, 2, 3a, 3b, 3c. LLM chậm hơn ~100 lần so với SLA.

## 3. Dự đoán và thực tế

1. **`hard_eval`: "LLM ~75–85% trên 50 case khó".** Thực tế **100% (50/50)**, cả prompt thường (4a) lẫn prompt phòng thủ (4d).
   Các model tự train chỉ còn 54–72%. Tôi đánh giá thấp khoảng cách giữa "học khuôn từ 3.600 tin" và "hiểu ý định người gửi".
2. **`dl_curve`: "MLP ≈ LogReg (±0.02)".** Thực tế MLP 0.964 so với LogReg 0.885. Nhưng bản *không ReLU* đạt 0.959, nên lợi thế
   không đến từ hidden layer mà từ cách train (Adam + mini-batch + 30 epoch, so với 300 bước GD còn underfit: recall train chỉ 0.847).
3. **`ml_lr`: "lr=50 dao động, không hội tụ mượt".** Loss vọt lên ~2.6 ở vài bước đầu rồi về **0.005**, thấp hơn cả lr=0.5 (0.083).
   Dữ liệu BoW gần tách tuyến tính, nên bước lớn chỉ làm ‖w‖ tăng nhanh. Loss train thấp nhưng mô hình quá tự tin.

Dự đoán đúng: model lớn hơn model nhỏ ~0.03 F1 (1 tin) và đắt **10 lần**. Few-shot không đổi F1 mà tốn **~3.3 lần**. LLM không thắng rõ 3b trên SUB.

## 4. AI log và điều đã học

| Phần | Hỏi AI gì | AI làm gì | Kiểm chứng |
| --- | --- | --- | --- |
| 4 | Viết `call_openai`, `build_classification_prompt`, `parse_label`, `llm_classify_batch` theo TODO 4.1–4.4 (Claude Code, Opus 5.5) | Code gọi Chat Completions với `**provider_kwargs(max_tokens)`, prompt zero/few-shot, parse nhãn chịu lỗi, tính chi phí theo token ước tính | Đọc lại trước khi gọi API; ô kiểm tra bằng API giả lập ✅ |
| Bonus | Side lab `src/optional_llm_api/` (Claude Code) | Viết `solution.py` (7 hàm theo template), chạy bài 2.1 với API thật, trả lời bài tập 2.1–2.3 | `pytest tests/test_solution.py -v`: 19/19 pass; chạy thử `compare_models`, `batch_compare` và chatbot streaming với API thật |
| 4 | Lỗi `TypeError: process() takes no keyword arguments`; LLM F1 = 0 ở lần chạy đầu | Nâng `brotli` 1.0.9 → 1.2.1; phát hiện `max_completion_tokens=3` làm `gpt-6-luna` trả `content=''` (`finish_reason="length"`), đổi thành 16 | Gọi API thử với 3 và 50 token; chạy lại notebook: invalid = 0, F1 LLM 0.974–1.000 |

**Điều hiểu ra:** muốn biết một thành phần (hidden layer) có tác dụng không thì phải làm ablation. MLP thắng LogReg nhưng bản bỏ ReLU cũng thắng,
nên lý do thật là cách tối ưu. Điểm F1 ~0.9–1.0 trên test cùng phân phối gần như không nói gì về độ bền: các model tự train chỉ còn 0.38–0.74
trên 50 case khó, còn LLM giữ 1.000 nhưng chậm hơn ~10 000 lần và tốn tiền. Một lỗi cấu hình nhỏ (`max_tokens=3`) có thể làm F1 = 0 mà không báo
lỗi, nên phải kiểm tra cột `invalid` trước khi tin vào số liệu.
