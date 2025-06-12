# 🍃 Patates Yaprağı Hastalık Tespiti – Makine Öğrenmesi ile Görüntü İşleme

Bu proje, patates bitkisi yapraklarının görüntüleri üzerinden **hastalık tespiti ve sınıflandırması** yapmak için geliştirilmiştir. Görüntülerden çıkarılan **renk, doku, şekil ve HOG** özellikleri kullanılarak farklı sınıflandırma algoritmalarıyla karşılaştırmalı analiz yapılmıştır.

## 🎯 Proje Amacı

Patates yapraklarını şu üç sınıfa ayıran bir model geliştirilmesi hedeflenmiştir:
- **Healthy (Sağlıklı)**
- **Early (Erken Hastalık Belirtisi)**
- **Late (İleri Düzey Hasta)**

Projede veri artırma (augmentation), öznitelik çıkarımı, farklı makine öğrenimi algoritmalarıyla eğitim ve performans karşılaştırmaları yer almaktadır.

---

## 🖼️ Kullanılan Veri Seti

Veri seti Kaggle üzerinden alınmıştır:  
🔗 [New Plant Diseases Dataset](https://www.kaggle.com/emmarex/plantdisease)

Bu projede sadece **patates yaprakları** kullanılmıştır. Lütfen veri setini indirip şu klasör yapısıyla `dataset/` klasörü altına yerleştiriniz:

## ⚙️ Kullanılan Kütüphaneler

- `numpy`, `opencv-python`, `matplotlib`, `seaborn`
- `scikit-learn`, `scikit-image`

🚀 Uygulama Nasıl Çalışır?
main.py:

Görüntüleri işler, veri artırımı yapar.

Renk, doku, şekil ve HOG özelliklerini çıkarır.

4 farklı makine öğrenimi modeliyle (RF, KNN, DT, NB) eğitim ve test işlemleri yapar.

grafik.py:

Confusion Matrix

Learning Curve

Cross-validation skorları

Train vs Test doğruluk analizlerini çizerek modellerin karşılaştırmasını görselleştirir.

Projeyi çalıştırmak için:

bash
Kopyala
Düzenle
python main.py
python grafik.py
📊 Kullanılan Sınıflandırıcılar
Algoritma	Başarı (Accuracy)	Açıklama
Random Forest	~97-99%	En yüksek doğruluğu sağlar, güçlü genelleme
Decision Tree	~97-98%	Görselleştirilebilir, hızlı
Naive Bayes	~83%	Basit ve hızlı, doğruluk sınırlı
KNN	~79%	Küçük veri setlerinde etkili

📈 Örnek Görselleştirmeler
Confusion Matrix

Learning Curves

CV Fold analizleri

Tüm grafikler grafik.py ile otomatik çizilir.

🧠 Kullanılan Öznitelikler
Özellik Tipi	Açıklama
Renk	HSV histogramları
Doku	Local Binary Pattern (LBP)
Şekil	Alan, çevre, kompaktlık
Kenar	Histogram of Oriented Gradients (HOG)

📚 Kaynaklar
Kaggle Dataset: Plant Disease

Gonzalez & Woods – Digital Image Processing

Scikit-learn, Scikit-image Documentation

👩‍💻 Geliştirici
İmran Ovacı
Çankırı Karatekin Üniversitesi
Bilgisayar Mühendisliği
📧 imran.ovc@hotmail.com

⚠️ Not
Bu proje, bitki hastalıklarının erken teşhisi ile tarımsal üretimde verimliliği artırmak amacıyla geliştirilmiştir. Eğitim amaçlı kullanılabilir.
