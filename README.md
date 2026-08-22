# تشخیص بیماری گیاهان با ResNet-18


<img width="1500" height="1200" alt="confusion_matrix" src="https://github.com/user-attachments/assets/e6a9d770-10cd-40b0-8269-5330a7fcfeb0" />
<img width="1800" height="600" alt="training_curves" src="https://github.com/user-attachments/assets/9f752814-c50e-4d4d-93af-024beafad830" />
<img width="900" height="900" alt="gradcam" src="https://github.com/user-attachments/assets/7db9a8be-2903-4ee4-b770-d47a052f94ed" />

پروژه‌ای آموزشی برای طبقه‌بندی تصاویر گیاهان در **۳۸ کلاس** با PyTorch و
Transfer Learning روی ResNet-18.

## وضعیت فعلی

- تقسیم داده: ۷۰٪ آموزش، ۱۵٪ اعتبارسنجی، ۱۵٪ آزمون
- بهترین مدل آموزش‌دیده: `weights/resnet18_best.pth`
- ارزیابی قبلی روی ۶٬۵۳۵ تصویر آزمون: accuracy حدود ۹۸٪
- خروجی‌های آموزشی و ارزیابی: پوشه `outputs/`

> دقت بالا روی PlantVillage الزاماً به معنی دقت یکسان روی عکس‌های موبایل و
> تصاویر واقعی مزرعه نیست؛ برای کاربرد واقعی باید با داده‌های بیرونی هم آزمون شود.

## ساختار رسمی

```text
data/
├── raw/                  داده خام؛ هر کلاس در یک پوشه
└── split/                خروجی train/val/test
weights/                  checkpoint مدل
outputs/                  نمودار آموزش و گزارش ارزیابی
split_data.py             تقسیم پایدار داده‌ها
train.py                 آموزش، scheduler و ذخیره checkpoint کامل
evaluate_resnet.py       ارزیابی روی test و ماتریس آشفتگی
predict.py               پیش‌بینی یک تصویر جدید
analyze_errors.py        تحلیل تصاویر اشتباه و جفت‌کلاس‌های پرتکرار
app.py                   رابط کاربری Streamlit
gradcam.py              نمایش ناحیه توجه مدل
check_dataset.py         کنترل کلاس‌ها، تعداد تصاویر و هم‌پوشانی splitها
test_project.py          تست smoke مدل و preprocessing
classes.txt              فهرست ۳۸ کلاس، مطابق پوشه‌های داده
```

اسکریپت‌های داخل `New folder` نسخه‌های آزمایشی قدیمی هستند و در مسیر رسمی
استفاده نمی‌شوند.

## نصب

```bash
pip install -r requirements.txt
```

## اجرای مرحله‌به‌مرحله

### ۱. ساخت split

```bash
python split_data.py
```

این دستور `data/split` قبلی را بازسازی می‌کند. داده خام در `data/raw` باید
دست‌نخورده باقی بماند.

### ۲. کنترل سلامت داده

```bash
python check_dataset.py
```

### ۳. اجرای تست smoke

```bash
python test_project.py
```

### ۴. آموزش

```bash
python train.py
```

تنظیمات قابل تغییر هستند:

```bash
python train.py --epochs 15 --batch-size 32 --learning-rate 0.0005
```

در پایان، بهترین checkpoint در `weights/resnet18_best.pth` و نمودارهای آموزش
در `outputs/training_curves.png` ذخیره می‌شوند. برای ادامه آموزش:

```bash
python train.py --resume --epochs 20
```

### ۵. ارزیابی نهایی

```bash
python evaluate_resnet.py
```

گزارش Precision/Recall/F1 در ترمینال و ماتریس آشفتگی در
`outputs/confusion_matrix.png` ذخیره می‌شود.

### ۶. پیش‌بینی تصویر جدید

```bash
python predict.py --image path/to/leaf.jpg
```

پنج پیش‌بینی برتر به‌همراه confidence نمایش داده می‌شوند. برای تغییر تعداد:

```bash
python predict.py --image path/to/leaf.jpg --top-k 3
```

### ۷. تحلیل خطاهای مدل

```bash
python analyze_errors.py
```

تصاویر اشتباه در `outputs/misclassified_images.csv` ثبت می‌شوند و پرتکرارترین
جفت‌کلاس‌های اشتباه در ترمینال نمایش داده می‌شوند.

### ۸. رابط کاربری

پس از نصب وابستگی‌ها:

```bash
streamlit run app.py
```

### ۹. توضیح ناحیه تشخیص با Grad-CAM

```bash
python gradcam.py --image path/to/leaf.jpg
```

خروجی در `outputs/gradcam.png` ذخیره می‌شود.

## نکات آموزشی مهم

- آموزش و ارزیابی از preprocessing یکسان با استاندارد ImageNet استفاده می‌کنند.
- checkpoint شامل وزن مدل، optimizer، epoch، کلاس‌ها و بهترین دقت validation است.
- `classes.txt` برای مستندسازی است؛ منبع اصلی ترتیب کلاس‌ها در checkpoint و
  `ImageFolder` ثبت می‌شود.
- برای پژوهش جدی‌تر، عدم‌توازن کلاس‌ها، تصاویر تکراری و آزمون روی دیتاست بیرونی
  باید بررسی شوند.
