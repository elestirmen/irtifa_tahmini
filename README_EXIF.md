## CSV'siz Eğitim (EXIF'ten Anlık Etiket)

Bu projede eğitim etiketlerini iki şekilde sağlayabilirsiniz:

- CSV: `filename,altitude` içeren indeks dosyası (varsayılan yaklaşım)
- EXIF: Görsellerin EXIF'inden `GPSInfo/GPSAltitude` alanını eğitim sırasında anlık okuma

EXIF tabanlı eğitim için ana eğitim betiğini aşağıdaki gibi çalıştırın:

```bash
python egitim_sureci_dosyadan_okuma_o1.py \
  --arch resnet18 \
  --image-folder output_images_irtifa_full/output_images_irtifa_full \
  --label-source exif \
  --epochs 20 \
  --batch-size 16 \
  --input-size 512
```

Notlar ve öneriler:
- Görseller EXIF içinde `GPSAltitude` bilgisine sahip olmalıdır; bu alanı olmayan dosyalar otomatik elenir.
- Etiketler batch içinde okunur; performans için in-memory cache etkin (varsayılan) olduğu için tekrar okuma maliyeti düşüktür.
- Çok büyük veri kümelerinde eğitimi hızlandırmak ve tekrarlanabilirliği artırmak için CSV indeks yaklaşımı genellikle daha verimlidir.
- Küçük/orta ölçekli setlerde ve hızlı denemelerde EXIF'ten anlık etiket oldukça pratiktir ve ön-işleme adımını ortadan kaldırır.

