#  Fundamentals of Accelerated Computing with CUDA Python

##  Giới thiệu về CUDA Python với Numba

Nền tảng tính toán **[CUDA](https://en.wikipedia.org/wiki/CUDA)** cho phép tăng tốc ứng dụng nhanh chóng bằng cách cho phép các developer thực thi code theo kiểu song song hàng loạt trên GPU NVIDA.

**[Numba](http://numba.pydata.org/)** là một trình biên dịch hàm Python tức thời (a just-in-time python function compiler), hiển thị một giao diện đơn giản để tăng tốc các hàm Python tập trung vào số. Numba là một tùy chọn rất hấp dẫn dành cho các lập trình viên Python muốn GPU tăng tốc ứng dụng của họ mà không cần viết mã C/C++, đặc biệt là đối với các nhà phát triển đã thực hiện các thao tác tính toán nặng trên mảng NumPy. Numba có thể được sử dụng để tăng tốc các chức năng Python cho CPU, cũng như cho GPU NVIDIA. **Trọng tâm của khóa học này là các kỹ thuật cơ bản cần thiết để tăng tốc GPU cho các ứng dụng Python bằng cách sử dụng Numba.**

###  Cấu trúc khóa học
 
Khóa học này được chia thành **ba** phần chính:

- _Giới thiệu về CUDA Python với Numba_
- _Tùy chỉnh CUDA Kernels trong Python với Numba_
- _Grids đa chiều và Shared Memory cho CUDA Python với Numba_

Mỗi phần chứa một vấn đề đánh giá cuối cùng, việc hoàn thành thành công sẽ cho phép bạn kiếm được Chứng chỉ năng lực cho khóa học. Mỗi phần cũng có một phụ lục với các tài liệu nâng cao dành cho những bạn quan tâm.

#### Phần 1: Introduction to CUDA Python with Numba

Trong phần đầu tiên này, trước tiên bạn sẽ tìm hiểu cách sử dụng Numba để biên dịch các chức năng cho CPU và sẽ được giới thiệu về hoạt động bên trong của trình biên dịch Numba. Sau đó, bạn sẽ tiếp tục tìm hiểu cách GPU tăng tốc các hàm mảng NumPy thông minh theo từng phần tử, cùng với một số kỹ thuật để di chuyển dữ liệu hiệu quả giữa máy chủ CPU và thiết bị GPU.

Vào cuối phiên đầu tiên, bạn sẽ có thể tăng tốc GPU mã Python để thực hiện các hoạt động thông minh về phần tử trên mảng NumPy.

#### Phần 2: Custom CUDA Kernels in Python with Numba

Trong phần thứ hai, bạn sẽ mở rộng khả năng của mình để có thể khởi chạy song song các chức năng tập trung vào số, tùy ý, không chỉ dựa trên phần tử trên GPU bằng cách viết các kernel CUDA tùy chỉnh. Để phục vụ mục tiêu này, bạn sẽ tìm hiểu về cách GPU NVIDIA thực thi mã song song. Ngoài ra, bạn sẽ được tiếp xúc với một số kỹ thuật lập trình song song cơ bản bao gồm cách phối hợp công việc của các luồng song song và cách giải quyết các điều kiện race. Bạn cũng sẽ tìm hiểu các kỹ thuật gỡ lỗi mã thực thi trên GPU.

Đến cuối phần thứ hai, bạn sẽ sẵn sàng để GPU tăng tốc một loạt chức năng tập trung vào số đáng kinh ngạc trên các tập dữ liệu 1D.

#### Phần 3: Multidimensional Grids and Shared Memory for CUDA Python with Numba

Trong phần thứ ba, bạn sẽ bắt đầu làm việc song song với dữ liệu 2D và sẽ tìm hiểu cách sử dụng không gian bộ nhớ trên chip trên GPU được gọi là bộ nhớ dùng chung.

Khi kết thúc phần thứ ba, bạn sẽ có thể viết mã tăng tốc GPU bằng Python bằng cách sử dụng Numba trên bộ dữ liệu 1D và 2D trong khi sử dụng một số chiến lược tối ưu hóa quan trọng nhất để viết mã tăng tốc GPU nhanh nhất quán.

### Điều kiện tiên quyết của khóa học

* Khả năng viết Python, cụ thể là viết và gọi các hàm, làm việc với các biến, vòng lặp và điều kiện cũng như imports.
* Quen thuộc với thư viện NumPy Python dành cho Python tập trung vào số. Nếu bạn chưa bao giờ sử dụng NumPy, nhưng đã quen thuộc với Python, bạn có thể sẽ thấy việc sử dụng NumPy trong phiên này rất đơn giản. Nhận xét và liên kết được cung cấp khi một số làm rõ có thể hữu ích.
* Hiểu biết ở mức độ cao về một số thuật ngữ khoa học máy tính như phân bổ bộ nhớ, loại giá trị, độ trễ và lõi xử lý.
* Hiểu biết cơ bản về vectơ và ma trận, cũng như phép nhân ma trận.

**Note**: Chi tiết của mỗi phần năm trong từng thư mục của mỗi phần.