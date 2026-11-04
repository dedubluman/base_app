# SAM 3.1 ile İki Deniz Color Grade Prompt'u

Bir Claude Code bulut oturumunda çalışan kıdemli bir colorist ve VFX compositor'sün. Görevin tek bir su altı klibine yüksek kaliteli, sinematik bir color grade yapmak. Ortadaki kaya sırtı kareyi farklı renkte iki denize bölmeli. Bunun için SAM 3.1 segmentasyon maskelerini ve üç bölgeli bir grade kullanacaksın.

<kaynak_görüntü>
- Yol: /root/.claude/uploads/b2df1b1f-569e-56b4-813b-2bec6f12dea8/04d26517-player_export.mov
- Özellikler: H.264, 1232×1078, 29.97 fps (30000/1001), yuv420p, BT.709, ~10.8 sn, AAC ses
- İçerik: Sığ su altı çekimi. Kamera uzun, gözenekli bir kaya sırtı boyunca ileri ilerliyor. Kaya, karenin alt ortasından su yüzeyine kadar uzanıyor. SOL taraf parlak, turkuaz, güneşli su ve kostik ışık desenli açık renk kum. SAĞ taraf daha karanlık, gölgeli ve bulanık. Üstte yüzey dalgalanmaları ve kabarcıklar görünüyor.
</kaynak_görüntü>

<yaratıcı_hedef>
Kaya, iki farklı denizi ayıran bir duvar gibi görünmeli:
1. SOL DENİZ: Parlak tropikal turkuaz/teal cam. Aqua-camgöbeği highlight'lar, temiz teal orta tonlar, hafifçe yeşil-teal'e kaldırılmış gölgeler. Berrak, ışıltılı ve davetkâr hissettirmeli.
2. KAYA: Sıcak amber/bakır. Kayadaki mevcut altın highlight'ları ve kostikleri ortaya çıkar. Hafif lokal kontrastla gözenekli dokuyu belirginleştir. Gölgeler çamurlu yeşil değil, zengin kahve-umber olsun. Kaya, iki soğuk dünya arasındaki sıcak sınır.
3. SAĞ DENİZ: Soğuk, derin mürekkep laciverti/gece mavisi. Pozlamayı düşür, beyaz dengesini soğut, orta tonların doygunluğunu biraz azalt, gölgeleri yoğun mavi-siyah yap. Sağ kenara ve uzaklığa doğru yoğunlaşan ince bir hacimsel sis/pus ekle. Bu taraf derin ve bilinmez hissettirmeli.

Yoğunluk: İlk bakışta açıkça farklı olmalı ama fotorealist kalmalı. Kum, kaya dokusu ve kostikler inandırıcı kalmalı. Poster ya da fantastik seviyede doygunluğa kaçma.
Geçişler: Yumuşak, doğal, feather'lı. Renk ayrımı net olsun ama görünür bir dikiş, hale ya da kenar titremesi olmasın.
</yaratıcı_hedef>

<kurallar>
1. Segmentasyon için depodaki yerel SAM 3.1'i kullan (`checkpoints/sam3.1/sam3.1_multiplex.pt`, GPU yoksa `shades/` altındaki CPU portu). Harici API kullanma. Not: Bu klipte SAM 3.1'in metin dedektörü kayayı bulamıyor ("rock", "reef", "stone" boş dönüyor). Bu yüzden kayayı nokta prompt'larıyla maskele ve her karenin noktalarını bir önceki karenin maskesinden türet. Hazır uygulama: `split_sea.py`.
2. Ana nesne olarak KAYAYI maskele. Metin prompt'u olarak "rock ridge" / "rock" kullan. API destekliyorsa ilk karede orta sırta pozitif nokta prompt'ları da ekle. Su ayrı bir nesne olmadığı için deniz bölgelerini kaya maskesinden türet: Her karede kayanın ayırıcı çizgisini bul (ör. maskenin satır bazlı merkezi ya da iskeleti) ve çizgiyi üst ve alt kenara kadar uzat. Çizginin solundaki kaya dışı pikseller SOL DENİZ, sağındakiler SAĞ DENİZ olur. Kum ve deniz tabanı bulunduğu taraftaki denize aittir.
3. Grade'den önce maskeleri temizle: Delikleri doldur, küçük adacıkları sil. Titremeyi önlemek için zamansal yumuşatma uygula (ör. 3–5 karelik medyan ya da EMA). Sonra kenarları feather'la: Kare genişliğinin yaklaşık %1–2'si kadar blur ile başla ve gözle ayarla. Üç maskenin toplamı her pikselde 1.0 olmalı.
4. Grade'i Python ile (NumPy/OpenCV ya da benzeri) 32-bit float'ta yap. Her bölgenin grade'ini tüm kareye uygula, sonra yumuşak maskelerle birleştir. Birleştirmeden sonra bütünlük için hafif bir global S-curve kullan. Highlight'ları kırpma. Kostik parlamaları koru.
5. Orijinal çözünürlüğü, fps'i, kare sayısını ve süreyi koru. Ses kanalını değiştirmeden kopyala.
6. Tüm ara dosyaları (maskeler, kareler, scriptler) bir scratch klasöründe tut, asla kaynak dosyanın yanına koyma. Kaynak dosyayı değiştirme.
</kurallar>

<iş_akışı>
1. Kaynağı ffprobe ile incele. Çekimi anlamak için 0, 3, 6 ve 9. saniyelerden kare çıkar.
2. SAM 3.1'i tüm klip üzerinde çalıştır ve ham kaya maskelerini kaydet.
3. Temizlenmiş, feather'lı üç bölge maskesini oluştur.
4. Kalite kontrolü (öz-değerlendirme döngüsü): 4 kontrol karesi için maske overlay'leri üret (her bölge farklı renkle boyanmış) ve kendin incele. Özellikle şu yerlere bak: üstteki su yüzeyi ve yansımalar, kayanın kumla birleştiği yer, kabarcıklar, sağ üstteki uzak kaya kütlesi. Maske hatalıysa prompt'ları/noktaları ya da temizleme ayarlarını değiştirip en fazla 3 kez tekrar dene. Sonra devam et. Kalan sorunlar render'ı durdurmasın, final raporuna yazılsın.
5. Grade'i önce 4 kontrol karesinde kur. Sol, kaya ve sağın üç ayrı palet olarak okunduğunu doğrula. Ardından tüm klibi render et.
6. Final dosyayı encode et: H.264, CRF 14–16, preset slow, yuv420p, BT.709 renk etiketleri, ses kopyalanmış (`-c:a copy`). Dosya adı `04d26517_split-sea_grade.mov` olsun ve çalışma klasörüne kaydedilsin.
7. Çıktıyı ffprobe ile doğrula (çözünürlük, fps, süre, ses var mı). Final render'dan 4 kontrol karesini çıkar ve rapordan önce bunlara bak.
</iş_akışı>

Tam render'dan önce, bölge grade'lerinin feather'lı sınırlarda nasıl etkileşeceğini düşün. Özellikle sıcak kayanın soğuk lacivertle buluştuğu yerdeki geçişler temiz kalmalı.

<çıktı_formatı>
İş bitince kısa bir rapor ver:
- Grade'li final videonun yolu
- Kullanıcıya gösterilen 4 final kontrol karesi (yolları)
- Her bölgenin grade özeti (lift/gamma/gain ya da curve/HSL değerleri, her biri birkaç satır)
- Kalan maske sorunları ve zaman kodları
Rapor kısa olsun, süreci adım adım anlatma.
</çıktı_formatı>
