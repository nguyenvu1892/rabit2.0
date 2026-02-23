Mục tiêu tối thượng

Xây AI Bot trade XAUUSD scalping M1/M5 (multi-TF sau) có khả năng tự học từ “kiến thức” (EMA/SMA/MACD/RSI/ATR/Stoch/Volume/Spread/Gap), không hardcode setup entry/TP/SL.

Hệ thống phải tự điều chỉnh entry decision + TP/SL/hold theo trạng thái thị trường.

Nguyên tắc “knowledge-only”

Được phép: feature từ indicator, market microstructure (spread, gap), regime detection, confidence estimation, reward shaping (DD, trade penalty).

Không được phép: if-else setup cố định kiểu “RSI<30 buy”, “TP=10 pip”, “SL=5 pip”, “RR=1:2” hard-set.

TP/SL phải là hàm thích nghi (vd ATR-multiple) do policy quyết định.

Định nghĩa thành công (Success Metrics)

Đánh giá trên Holdout OOS (cuối dataset) và Walk-forward:

Tier-0 (baseline sống được):

PF ≥ 1.05

Balance ≥ 0

MaxDD không phình (≤ 2× baseline)

Trades không “spam” (≤ 1.5× baseline)

Tier-1 (đủ điều kiện forward/paper trade):

PF ≥ 1.15

Balance dương ổn định qua ≥ 3 runs (seed khác)

MaxDD giảm hoặc giữ ổn

Không phụ thuộc 1 regime duy nhất

Tier-2 (candidate live micro-lot):

Walk-forward ổn định ≥ 5 folds

PF ≥ 1.2 trên holdout + PF ≥ 1.1 trung bình WF

Risk controls pass (max loss/day, spread spike handling)

Roadmap ưu tiên (ngắn gọn)

Stability/OOS generalization (walk-forward + robust reward + abstain)

Regime/Confidence intelligence (mixture-of-experts mềm)

Multi-timeframe (sau khi single TF ổn)

Online learning (shadow/paper trade loop)