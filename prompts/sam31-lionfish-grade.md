# SAM 3.1 ile Aslan Balığı Subject Isolation + Cinematic Grade Prompt'u

Bir Claude Code bulut oturumunda çalışan kıdemli bir colorist ve VFX compositor'sün. Görevin, tek bir su altı klibinde aslan balığını (ve kadrajdaki diğer tüm balıkları) SAM 3.1 ile arka plandan tamamen ayırmak, ön planı referans fotoğraftaki aslan balığının renklerine göre canlandırmak ve arka planı sinematik, klostrofobik bir atmosfere çevirmek.

<kaynak_görüntü>
- Video: /root/.claude/uploads/153be1ff-df1c-5687-a3b2-7266a5d02122/52257c85-player_export.mov
- Özellikler: H.264, 1080×1920 (dikey), 29.97 fps (30000/1001), 340 kare, ~11.3 sn, AAC ses
- İçerik: Su altı karesi. Aslan balığı ana özne; etrafında başka balıklar var. Tüm kare sis/pus perdesi altında (düşük kontrast, soluk renk). Önce 0, 3, 6, 9 ve 11. saniyelerden kare çıkarıp içeriği kendin gör.
- Referans renk fotoğrafı: /root/.claude/uploads/153be1ff-df1c-5687-a3b2-7266a5d02122/e0aaadd9-image.webp
  Aslan balığı: koyu turuncu-kırmızı (kiremit/amber) gövde bantları, krem-beyaz ince çizgiler, siyah-beyaz benekli yüzgeç ışınları, ten rengi baş. Arka plan: derin elektrik mavisi. Bu paleti hedef al (sadece balığın rengi; fotodaki mavi arka planı kopyalama).
</kaynak_görüntü>

<yaratıcı_hedef>
İki katman:

1. ÖN PLAN (aslan balığı + diğer tüm balıklar): Sis perdesini (dehaze) SADECE bu katmanda kaldır. Kontrast, mikro-kontrast/netlik ve doğal renkleri canlandır. Aslan balığının turuncu-kırmızı bantları, krem çizgileri ve benekli yüzgeçleri referans fotodaki gibi canlı, zengin ve doygun görünsün. Diğer balıklar kendi doğal renklerinde, canlı ve net olsun; aslan balığıyla aynı ışık/kontrast seviyesinde ön planda durmalılar. Gölgeler kirli yeşil/gri değil, temiz ve derin olsun. Highlight kırpma yok.
2. ARKA PLAN (balıklar dışında her şey: su, mercan, kaya, zemin): Orta-güçlü sinematik klostrofobik atmosfer. Kontrastı ve doygunluğu belirgin şekilde düşür (derinlik hissi için). Koyu mavi-teal renk kayması, belirgin vinyet (kenarlar kapanıyor, kadraj daralıyor), hafif lens blur/defocus, ince film greni, pozlamayı düşür, gölgeleri ezilmiş mavi-siyaha çek. Sıkışmış, boğucu ve derin bir his; ama ön plan balıklar net şekilde parlasın.

Yoğunluk: Ön plan ile arka plan arasındaki ayrım ilk bakışta açıkça belli olmalı ama fotorealist kalmalı. Balık kenarlarında hale (halo), renk sızıntısı, dikiş ya da titreme olmamalı. Aşırı yapay "kesilip yapıştırılmış" görünüm istemiyorum.
</yaratıcı_hedef>

<kurallar>
1. Tüm maskeleme SAM 3.1 ile yapılsın: depodaki yerel checkpoint `checkpoints/sam3.1/sam3.1_multiplex.pt` ve CPU'da çalışan port (`shades/` paketi, `sam3/` deposu; örnek kullanım için `split_sea.py`, `cuttle_masks.py`, `diver_masks.py`). Harici API ya da başka segmentasyon modeli kullanma. Ortamda GPU ve torch yoksa önce `requirements_three_shades.txt` ve `README_three_shades.md`'ye göre CPU bağımlılıklarını kur. (Makine: 4 çekirdek, ~15 GB RAM.)
2. CPU maliyeti için: SAM 3.1'i düşük çözünürlüklü proxy karelerde (ör. uzun kenar 640–768) çalıştır, aralıklı karelerde (stride) maskele, aradaki kareleri SAM'in video takip (tracker) özelliğiyle ya da optik akışla taşı. Ardından maskeleri tam çözünürlüğe yükselt ve kenarları orijinal kareye göre iyileştir (guided filter / matte.py). Uzun işleri arka planda çalıştır, ilerlemeyi logla.
3. Metin prompt'u olarak "fish" (gerekirse "lionfish", "small fish") kullan. Tek bir birleşik ön plan maskesi üret: aslan balığı + tüm balıklar. Metin dedektörü bir balığı kaçırırsa nokta prompt'larıyla (önceki karenin maskesinden türetilmiş) tamamla. Mercan, kaya, su, kabarcık ve yüzey ışığı ARKA PLAN sayılır. Aslan balığının uzun yüzgeç ışınları ve dikenleri maskeye dahil olmalı (ince detay kaybolmasın).
4. Maske temizliği: delikleri doldur, küçük adacıkları sil, 3–5 karelik zamansal yumuşatma uygula (medyan/EMA), sonra kenarları feather'la (kare genişliğinin yaklaşık %0.5–1.5'i; ince yüzgeç ışınlarında daha dar). Ön plan + arka plan ağırlıkları her pikselde toplam 1.0 olsun.
5. Grade Python ile (NumPy/OpenCV) 32-bit float'ta yapılsın. Ön plan ve arka plan grade'lerini tüm kareye ayrı ayrı uygula, sonra feather'lı maskeyle birleştir. Dehaze sadece ön plan için (örn. dark channel / veil tahmini tüm klip için sabit olsun ki titreme olmasın). Arka plan blur'ü maske sınırında ön plan renginin sızmasını önlemek için ön plan maskesi genişletilip dolgulanarak (inpaint/normalized blur) uygulansın.
6. Renk hedefi referans fotoğrafa göre ayarlansın: balık piksellerinin ton/doygunluk dağılımını referanstaki balığın dağılımıyla karşılaştır (HSV histogram ya da ortalama turuncu-kırmızı ton ve doygunluk). Ama diğer balıkların doğal renklerini aslan balığına çevirme: tonu sadece aslan balığı örnek maskesine uygula, diğer balıklara yalnızca dehaze + kontrast + doğal doygunluk uygula. Aslan balığını ayırt etmek için SAM'e "lionfish" prompt'u ver (ya da en büyük/en turuncu-çizgili balık örneğini seç).
7. Orijinal çözünürlüğü (1080×1920), fps'i (30000/1001), kare sayısını (340) ve süreyi koru. Ses kanalını değiştirmeden kopyala. Kaynak dosyayı değiştirme.
8. Tüm ara dosyaları (maskeler, kareler, scriptler, loglar) bir scratch klasöründe tut, kaynak dosyanın yanına koyma.
</kurallar>

<iş_akışı>
1. Kaynağı ffprobe ile incele; 0, 3, 6, 9, 11. saniyelerden kare çıkar ve balıkların sayısını/konumunu, sisin yoğunluğunu not et.
2. Gerekli bağımlılıkları kur, SAM 3.1'in CPU'da yüklendiğini küçük bir testle doğrula (tek kare, "fish" prompt'u).
3. Proxy karelerde SAM 3.1'i tüm klip üzerinde çalıştır: (a) tüm balıklar birleşik maske, (b) aslan balığı ayrı maske. Ham maskeleri kaydet.
4. Maskeleri temizle, zamansal yumuşat, feather'la, tam çözünürlüğe yükselt.
5. Kalite kontrolü (öz-değerlendirme döngüsü): 4–6 kontrol karesi için maske overlay'leri üret (ön plan yeşil, arka plan kırmızı ton gibi) ve kendin incele. Özellikle: yüzgeç ışınları ve dikenler, kuyruk/yüzgeç şeffaf bölgeleri, uzaktaki küçük balıklar, mercan ile balık arasındaki temas, kenar halesi. Hata varsa prompt'ları, noktaları ya da temizleme ayarlarını değiştirip en fazla 3 kez tekrar dene. Kalan sorunlar render'ı durdurmasın, final raporda belirtilsin.
6. Grade'i önce kontrol karelerinde kur: ön plan canlı, arka plan sıkışık/klostrofobik ve düşük doygunluklu mu? Aslan balığı rengi referans fotoğrafa yakın mı? Sonra tüm klibi render et.
7. Encode: H.264, CRF 14–16, preset slow, yuv420p, BT.709 renk etiketleri, `-c:a copy`. Dosya adı `52257c85_lionfish_grade.mp4`, çalışma klasörüne kaydedilsin.
8. Çıktıyı ffprobe ile doğrula (çözünürlük, fps, kare sayısı, süre, ses). Final videodan 4 kontrol karesini çıkar ve rapordan önce bunlara bak.
</iş_akışı>

Tam render'dan önce şunları düşün: sis kaldırma ön planda kontrast ve gürültüyü ne kadar artırıyor, blur'lü arka plan ile keskin balık kenarının birleştiği yerde hale oluşuyor mu, ve vinyet balıkları kesiyor mu (vinyet ön plan balık parlaklığını düşürmemeli).

<çıktı_formatı>
Sadece grade'li video teslim edilir (karşılaştırma videosu veya maske dizisi istemiyorum). İş bitince kısa bir rapor ver:
- Grade'li final videonun yolu
- Kullanıcıya gösterilen 4 final kontrol karesi (yolları)
- Ön plan ve arka plan grade özeti (dehaze miktarı, kontrast/netlik, ton/doygunluk, vinyet/blur/grain değerleri; her biri birkaç satır)
- Kalan maske sorunları ve zaman kodları
Rapor kısa olsun, süreci adım adım anlatma.
</çıktı_formatı>
