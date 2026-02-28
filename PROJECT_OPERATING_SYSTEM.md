Universal Engineering Standard
1. Plan Before Build

Bắt buộc lập kế hoạch khi:

Task ≥ 3 bước

Có thay đổi kiến trúc

Có thay đổi logic lõi

Yêu cầu:

Xác định mục tiêu

Xác định phạm vi ảnh hưởng

Xác định rủi ro

Nếu đi sai hướng → DỪNG → re-plan.
Không “fix tiếp cho tới khi chạy”.

2. Change With Discipline

Mỗi thay đổi chỉ giải quyết một vấn đề.

Không trộn feature + refactor + bug fix.

Chỉ sửa đúng phần cần thiết.

Không hack. Không temporary fix.

Luôn tìm root cause.

3. Verify Before Done

Không được mark Done nếu chưa:

Kiểm tra hành vi trước/sau

Kiểm tra edge case

Xác minh không gây regression

Câu hỏi bắt buộc:

Nếu deploy production ngay bây giờ, có tự tin không?

Nếu không chắc → chưa Done.

4. Keep It Simple

Ưu tiên giải pháp đơn giản nhất đáp ứng yêu cầu.

Không over-engineer.

Không thêm complexity không cần thiết.

Clarity > Cleverness.

5. Minimize Impact

Thay đổi phải có phạm vi nhỏ nhất.

Tránh ripple effect.

Không phá vỡ interface nếu chưa có migration plan.

Stability > Convenience.

6. Ownership

Người triển khai chịu trách nhiệm end-to-end:

Tự đọc log

Tự debug

Tự xác minh

Không đẩy complexity sang người khác

7. Learn From Errors

Sau mỗi lỗi:

Xác định root cause

Rút ra rule phòng ngừa

Ghi lại bài học

Không chỉ sửa lỗi — phải giảm xác suất lặp lại.

8. Definition of Done

Một task chỉ được coi là Done khi:

Đúng spec

Đã verify behavior

Không gây regression

Không tạo technical debt

Nếu còn nghi ngờ → chưa Done.

Final Rule

Mục tiêu không phải “chạy được”.

Mục tiêu là hệ thống:

Ổn định

Dễ hiểu

Dễ mở rộng

Dễ bảo trì

Mọi thay đổi phải bảo vệ tương lai của hệ thống.