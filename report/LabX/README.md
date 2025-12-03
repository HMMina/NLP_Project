BÁO CÁO TỔNG QUAN NGHIÊN CỨU: CÔNG NGHỆ TỔNG HỢP TIẾNG NÓI (TEXT-TO-SPEECH)
1. Giới thiệu
Trong bức tranh toàn cảnh của Trí tuệ nhân tạo (AI) và Tương tác Người - Máy (HCI), công nghệ Text-to-Speech (TTS) đóng vai trò cầu nối quan trọng, chuyển đổi thông tin văn bản tĩnh thành tín hiệu âm thanh động, mang đầy đủ sắc thái và ngữ điệu.
Từ những nỗ lực sơ khai, TTS hiện đại đã trải qua cuộc cách mạng về phương pháp luận, đạt đến ngưỡng "Human-parity" (chất lượng ngang bằng con người). Sự chuyển dịch từ xử lý tín hiệu truyền thống sang các kiến trúc Deep Learning (Học sâu) và gần đây nhất là Speech Language Models (SpeechLMs) đã mở ra khả năng nhân bản giọng nói (Voice Cloning) và tổng hợp đa ngôn ngữ Zero-shot.
Báo cáo này cung cấp khảo sát từ các kiến trúc nền tảng như Tacotron, FastSpeech đến các hệ thống sinh tiên tiến như VITS, VALL-E và StyleTTS 2, đồng thời phân tích bối cảnh dữ liệu cho tiếng Việt và các giải pháp an toàn AI như Watermarking4.
2. Lịch sử và Cơ sở Lý thuyết: Từ Ghép nối đến Tham số
Trước khi Deep Learning bùng nổ, công nghệ TTS dựa trên hai trụ cột chính:
2.1. Concatenative Synthesis (Tổng hợp Ghép nối)
Phương pháp tiêu biểu là Unit Selection (Lựa chọn đơn vị).
•	Cơ chế: Hệ thống lưu trữ một cơ sở dữ liệu khổng lồ chứa các đoạn ghi âm được phân đoạn tỉ mỉ thành các đơn vị nhỏ (âm vị, từ). Khi nhận văn bản, thuật toán tìm kiếm (thường dùng Viterbi search) sẽ chọn ra chuỗi đơn vị khớp nhất và ghép nối chúng lại.
•	Hạn chế:
o	Phase/Spectral mismatch: Sự không khớp về pha hoặc đường bao phổ tại các điểm nối gây ra tiếng lách cách (clicks).
o	Thiếu linh hoạt: Việc thay đổi cảm xúc yêu cầu phải thu âm lại toàn bộ cơ sở dữ liệu.
2.2. Statistical Parametric Speech Synthesis - SPSS (Tổng hợp Tham số Thống kê)
Đại diện tiêu biểu là HMM-based TTS.
•	Cơ chế: Thay vì lưu sóng âm, hệ thống mô hình hóa các tham số thống kê (tần số cơ bản $F0$, phổ năng lượng, thời lượng). Một bộ Vocoder sẽ tái tạo sóng âm từ các tham số này.
•	Ưu điểm: Linh hoạt điều chỉnh cao độ, tốc độ mà không cần thu âm lại.
•	Hạn chế: Âm thanh thường bị Oversmoothing (quá trơn). Việc lấy trung bình thống kê làm mất đi các chi tiết tinh tế (fine structure), khiến giọng nói nghe máy móc ("robotic") và có tiếng rè.
3. Kỷ nguyên Neural TTS: Sự dịch chuyển sang End-to-End
Các hệ thống End-to-End học trực tiếp ánh xạ từ cặp dữ liệu <Văn bản, Âm thanh>.
3.1. Các Mô hình Autoregressive (Tự hồi quy): Tacotron 2
Tacotron 2 sử dụng kiến trúc Sequence-to-Sequence (Seq2Seq) với cơ chế Attention.
•	Cơ chế: Bao gồm Encoder (xử lý văn bản) và Decoder (sinh Mel-spectrogram). Tại mỗi bước, Decoder dự đoán khung hình tiếp theo dựa trên thông tin quá khứ.
•	Hạn chế:
o	Robustness (Độ ổn định): Cơ chế Soft Attention đôi khi gây lỗi căn chỉnh (alignment failures), dẫn đến lặp từ hoặc bỏ sót từ.
o	Inference Speed (Tốc độ suy luận): Chậm do bản chất sinh tuần tự từng khung hình.
3.2. Các Mô hình Non-Autoregressive (Không tự hồi quy): FastSpeech 2
FastSpeech 2 cho phép sinh toàn bộ chuỗi Mel-spectrogram song song (parallel generation)18.
•	Cải tiến cốt lõi: Variance Adaptor
o	Duration Predictor: Dự đoán thời lượng của từng âm vị để thực hiện Hard Alignment, loại bỏ hoàn toàn lỗi lặp từ.
o	Pitch & Energy Predictors: Dự đoán cao độ và năng lượng giúp tăng khả năng biểu cảm.
•	Hiệu năng: Tốc độ suy luận nhanh hơn gấp 270 lần so với Tacotron 2 (phần sinh Mel).
4. Neural Vocoders: Cầu nối tới Sóng âm Trung thực
Mô hình âm học (như FastSpeech) chỉ tạo ra Mel-spectrogram, cần Neural Vocoder để chuyển đổi thành sóng âm.
HiFi-GAN (High-Fidelity GAN)
HiFi-GAN là chuẩn mực công nghiệp nhờ cân bằng giữa tốc độ và chất lượng.
•	Multi-Period Discriminator (MPD): Phân tích âm thanh ở các chu kỳ khác nhau, nắm bắt cấu trúc tuần hoàn của giọng nói.
•	Multi-Scale Discriminator (MSD): Phân tích ở các tỷ lệ mẫu khác nhau để đảm bảo sự liền mạch.
•	Kết quả: Đạt điểm MOS cao hơn WaveNet trong khi tốc độ nhanh hơn hàng trăm lần.
5. Các Mô hình Sinh Tiên tiến: VITS và StyleTTS 2
5.1. VITS (Variational Inference with Adversarial Learning)
VITS là mô hình Fully End-to-End, huấn luyện từ văn bản đến sóng âm trong một chu trình.
•	Monotonic Alignment Search (MAS): Tự động tìm ra sự căn chỉnh tối ưu giữa văn bản và âm thanh mà không cần dữ liệu label bên ngoài.
•	Normalizing Flows: Tăng cường khả năng biểu diễn của phân phối tiềm ẩn, giúp sinh ra giọng nói tự nhiên hơn.
5.2. StyleTTS 2: Sức mạnh của Diffusion Models
Sử dụng Diffusion (Mô hình khuếch tán) để kiểm soát phong cách.
•	Style Diffusion: Coi "phong cách" là một biến tiềm ẩn và dùng mạng diffusion để sinh ra nó, cho phép thích ứng Zero-shot vượt trội.
•	Adversarial Training với SLM: Sử dụng các mô hình ngôn ngữ tiếng nói lớn (như WavLM) làm Discriminator để học các đặc trưng ngữ nghĩa cấp cao.
6. Sự Trỗi dậy của Speech Language Models (SpeechLMs)
Hướng tiếp cận mới coi TTS như bài toán mô hình ngôn ngữ (Language Modeling).
6.1. Neural Codec & Discrete Tokens
Các mô hình như VALL-E sử dụng bộ mã hóa âm thanh (như Encodec) để chuyển đổi âm thanh thành các token rời rạc (Discrete Tokens). Quá trình TTS trở thành bài toán dự đoán token tiếp theo (next-token prediction).
6.2. VALL-E và Zero-shot Voice Cloning
•	Khả năng: Nhân bản giọng nói chỉ với 3 giây âm thanh mẫu (acoustic prompt).
•	In-context Learning: Không cần tinh chỉnh (fine-tune). Mô hình dùng đoạn mẫu 3 giây làm ngữ cảnh để sinh ra giọng nói khớp môi trường âm học.
•	VALL-E 2: Cải thiện độ ổn định, đạt hiệu năng ngang bằng con người (Human Parity).

7. Tổng hợp Tiếng nói cho Tiếng Việt: Thách thức và Cơ hội
7.1. Thách thức Ngôn ngữ học
Tiếng Việt là ngôn ngữ đơn âm tiết có thanh điệu (Tonal language). Sáu thanh điệu quyết định ngữ nghĩa, đòi hỏi mô hình phải sinh ra đường bao Pitch Contour (F0) cực kỳ chính xác. Hiện tượng biến âm (Sandhi) và đa dạng vùng miền cũng là rào cản lớn.
7.2. Bối cảnh Dữ liệu và Mô hình
•	PhoAudiobook (2025): Bộ dữ liệu đột phá với 941 giờ sách nói chất lượng cao và ngữ cảnh dài (long-form), khắc phục hạn chế của các dữ liệu cắt ngắn trước đây.
•	Thực nghiệm cho thấy các mô hình Zero-shot (như VALL-E, VoiceCraft) huấn luyện trên PhoAudiobook xử lý ngữ điệu tiếng Việt vượt trội so với các phương pháp cũ.
8. So sánh Voice Cloning: Fine-tuning vs. Zero-shot
Tiêu chí	Zero-shot (VALL-E, XTTS)	Fine-tuning
Dữ liệu yêu cầu	3 giây - 30 giây (Prompt)	1 phút - 1 giờ (Training data)
Thời gian triển khai	Tức thì (Real-time inference)	Chậm (Cần thời gian train/adapt)
Độ giống	Khá tốt về âm sắc (Timbre)	Rất cao, nắm bắt được thói quen phát âm (Prosody quirks)
Tính ứng dụng	Chatbot, Trợ lý ảo cá nhân hóa	Sách nói, phim ảnh, nội dung chuyên nghiệp

9. An toàn AI, Đạo đức và Thủy vân số (Watermarking)
9.1. Công nghệ Watermarking
Để chống lại rủi ro Deepfake, các giải pháp đóng dấu bản quyền ẩn đã ra đời:
•	AudioSeal (Meta): Nhúng watermark cục bộ (localized), cho phép phát hiện chính xác đoạn bị giả mạo trong file dài với tốc độ cực nhanh.
•	SynthID (Google): Sử dụng biến đổi phổ, đảm bảo tính bền vững (Robustness) trước các thao tác nén MP3 hay thay đổi tốc độ.
9.2. Chính sách Đạo đức
•	OpenAI: Áp dụng "No-go voice list" để bảo vệ người nổi tiếng và yêu cầu sự đồng ý rõ ràng.
•	ElevenLabs: Yêu cầu xác minh giọng nói (Voice Verification) để chống mạo danh.
10. Kết luận
Lĩnh vực Tổng hợp Tiếng nói đang chuyển dịch mạnh mẽ từ các mô hình chuyên biệt sang các mô hình tổng quát dựa trên Discrete Tokens và SpeechLMs. Đối với Việt Nam, việc khai thác các bộ dữ liệu lớn như PhoAudiobook và làm chủ công nghệ SpeechLM sẽ là chìa khóa để xây dựng các trợ lý ảo thuần Việt tự nhiên và an toàn.






Nguồn trích dẫn 
Recent Advances in Speech Language Models: A Survey - ACL Anthology 
•	https://aclanthology.org/2025.acl-long.682.pdf
[2410.03751] Recent Advances in Speech Language Models: A Survey - arXiv 
•	https://arxiv.org/abs/2410.03751
A Survey on Neural Speech Synthesis - arXiv 
•	https://arxiv.org/pdf/2106.15561
Tacotron 1 and 2 Documentation - Coqui TTS 
•	https://docs.coqui.ai/en/latest/models/tacotron1-2.html
FastSpeech 2: Fast and Robust AI for Natural Text-to-Speech - Zignuts Technolab 
•	https://www.zignuts.com/ai/fastspeech-2
FastSpeech: New text-to-speech model improves on speed, accuracy, and controllability - Microsoft Research 
•	https://www.microsoft.com/en-us/research/blog/fastspeech-new-text-to-speech-model-improves-on-speed-accuracy-and-controllability/
 Fast Inference End-to-End Speech Synthesis with Style Diffusion - MDPI 
•	https://www.mdpi.com/2079-9292/14/14/2829
HiFi-GAN Explained: Mastering High-Fidelity Audio in AI Solutions - Vapi AI Blog 
•	https://vapi.ai/blog/hifi-gan
HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis - GitHub 
•	https://github.com/jik876/hifi-gan
A Voice Cloning Method Based on the Improved HiFi-GAN Model - PMC NIH 
•	https://pmc.ncbi.nlm.nih.gov/articles/PMC9578849/
VITS Documentation - Coqui TTS 
•	https://docs.coqui.ai/en/dev/models/vits.html

VALL-E Project Page - Microsoft Research 
•	https://www.microsoft.com/en-us/research/project/vall-e-x/vall-e/
Vietnamese TTS Model - Hugging Face (ZaloPay) 
•	https://huggingface.co/zalopay/vietnamese-tts
Non-Standard Vietnamese Word Detection and Normalization for Text–to–Speech - arXiv 
•	https://arxiv.org/pdf/2209.02971
VLSP 2021: Vietnamese Text-To-Speech Evaluation 
•	https://vlsp.org.vn/vlsp2021/eval/tts
VinBigdata shares 100-hour data for the community - VinBigdata Institute 
•	https://institute.vinbigdata.org/en/events/vinbigdata-shares-100-hour-data-for-the-community/
VinBigdata shares 100-hour data news - VinBigdata 
•	https://vinbigdata.com/en/news/vinbigdata-shares-100-hour-data-for-the-community
Zero-Shot Text-to-Speech for Vietnamese - arXiv 
•	https://arxiv.org/html/2506.01322v1
Comparison of Voice Cloning Algorithms in Zero-shot and Few-shot Scenarios – ResearchGate
•	https://www.researchgate.net/publication/385500848_Comparison_of_Voice_Cloning_Algorithms_in_Zero-shot_and_Few-shot_Scenarios
Text-to-Speech Models: Multilingual Capabilities and Voice Cloning - Eidos.ai 
•	https://www.eidos.ai/blog-posts/text-to-speech-models-multilingual-capabilities-and-voice-cloning
Proactive Detection of Voice Cloning with Localized Watermarking (AudioSeal) - AI at Meta 
•	https://ai.meta.com/research/publications/proactive-detection-of-voice-cloning-with-localized-watermarking/



