Quy tắc thay đổi code (cực quan trọng)

Không xoá code cũ khi chưa có bằng chứng thay thế tốt hơn.

Nếu cần thay: tạo file mới hoặc thêm option flag.

Mỗi thay đổi phải chạy được: python -m scripts.check_data và ít nhất 1 training script.

Mỗi PR/commit chỉ làm 1 việc (atomic changes).

Quy tắc module boundaries

rabit/data/*: load/resample/features (pure functions, deterministic)

rabit/env/*: execution + ledger + env (no ML)

rabit/rl/*: policies + trainers (no pandas heavy inside loops nếu có thể)

rabit/regime/*: regime detector, confidence gate (deterministic, explainable)

Quy tắc speed/perf

Loop backtest/training ưu tiên numpy arrays, tránh .iloc trong loop.

Bất cứ chỗ nào gọi backtest nhiều lần → ưu tiên caching windows hoặc parallel.

Quy tắc reproducibility

Mọi trainer/policy phải nhận seed.

Báo cáo output phải ghi:

params (sigma/alpha/n_dirs/window_size/eval_windows)

feature list

dataset split (train/val/hold)

metrics train/val/hold

Quy tắc testing tối thiểu

Mỗi milestone phải có:

sanity check số rows > 0

no NaN explosion

backtest runs end-to-end

metrics file saved

Quy tắc commit message

feat: thêm tính năng

fix: sửa bug

perf: tối ưu tốc độ

chore: dọn repo/doc

exp: thử nghiệm (không merge main nếu fail)

Workflow Tech Lead ↔ Codex

Tech Lead (tôi) cung cấp: spec + acceptance criteria + files impacted.

Codex chỉ implement theo spec, không tự ý đổi reward/regime logic.

Mọi thay đổi reward/regime phải qua review.