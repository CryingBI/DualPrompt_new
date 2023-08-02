# Đồ án đề xuất cải tiến của mô hình DualPrompt

Đây là kho lưu trữ code của đồ án dựa trên code sử dụng trong bài <a href="https://arxiv.org/pdf/2204.04799.pdf">DualPrompt</a>, <br>
Wang, Zifeng, et al. "DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning." ECCV. 2022.

## Environment
Phần cứng sử dụng:
- GPU P100 trên Kaggle
- GPU T4 trên Google Colab
## Cách sử dụng trên Kaggle (Google Colab tương tự)
Đầu tiên, tải kho lưu trữ về máy của bạn.
```
git clone https://github.com/CryingBI/DualPrompt_new.git
```
Tiếp theo tạo một kho lưu trữ mới trên Github của bạn, sau đó loại bỏ git remote chính và thêm git remote cho kho lưu trữ bạn vừa tạo.
```
git remote remove origin
git remote add origin yourRemoteUrl
git push -u origin master
```
Đăng nhập vào tài khoản Kaggle của bạn, tạo một Notebook mới, chọn thiết bị GPU P100. <br>
Sử dụng các câu lệnh sau để bắt đầu chạy thử nghiệm (Tạo mỗi câu lệnh một ô để chạy). <br>
Tập dữ liệu Split-CIFAR-100 tương ứng chọn cifar100_dualprompt, tập dữ liệu Split-ImageNet-R tương ứng chọn imr_dualprompt trong lệnh torchrun. <br>
```
!git clone Your_Repo

%cd /kaggle/working/Your_Repo

!pip install -r requirements.txt

!torchrun \
        --nproc_per_node=1 \
        main.py \
        <imr_dualprompt> or <cifar100_dualprompt> \
        --model vit_base_patch16_224 \
        --batch-size 24 \
        --data-path /local_datasets/ \
        --output_dir ./output 
```

## Trích dẫn bài báo DualPrompt
```
@article{wang2022dualprompt,
  title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
  author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
  journal={European Conference on Computer Vision},
  year={2022}
}
```
