## Kelompok 5 ##
- Intan Trinanda (L0124018)
- Izanahda Nurkhasna (L0124019)
- Waldani Nabila Tamamah (L0124122)

# Aplikasi Pengenalan Wajah dengan EigenFace
## Tujuan Program
- Mengenali wajah dari gambar uji berdasarkan dataset wajah yang dimasukkan
- Menggunakan metode EigenFace untuk proyeksi ke ruang eigen
- Menampilkan wajah paling mirip berdasarkan jarak Euclidean

## Teknologi yang Digunakan
- Python
- Streamlit (untuk GUI berbasis web)
- NumPy
- OpenCV (untuk gambar)
- PIL (untuk konversi gambar)
- Manual power iteration (untuk mencari eigenvector)

## Struktur Proyek
eigenface_project/
	- src/
		-- app.py
		-- eigenface_utils.py

	- doc/
		-- Laporan
		-- Readme.txt
	- test
		-- dataset.zip #Dataset wajah dalam bentuk zip
		-- gambar_uji/
			--- uji1.jpg
			--- uji2.jpg
			--- uji3.jpg

## Cara Menjalankan Program
- Buka terminal di folder eigenface_project
- Install library:
  pip install -r requirements.txt
- Jalankan program:
  streamlit run src/app.py

## Cara Menggunakan Aplikasi
- Upload dataset
  Tekan "Choose Folder Zip", kemudian pilih dataset.zip yang berisi gambar wajah
- Upload gambar uji
  Tekan "Choose Test Image", kemudian pilih gambar uji
- Atur threshold
  Gunakan slider di sidebar kiri, kemudian atur threshould mulai dari 1000 sampai 300000 (default threshold adalah 10000)
- Lihat hasil
  Kolom sebelah kiri menampilkan gambar uji
  Kolom sebelah kanan menampilkann wajah paling mirip dari dataset
  Di bagian bawah akan tampil status Match atau No Match dan nilai jaraknya
- Eksekusi selesai
  Di bawah hasil akan tampil waktu yang dibutuhkan untuk eksekusi program

## Penjelasan Metode (EigenFace)
- Dataset diubah jadi grayscale 100×100 dan diflatten jadi vektor
- Dihitung wajah rata-rata (`mean_face`)
- Dibentuk matriks deviasi `A = X - mean_face`
- Dihitung kovarian `C = A.T @ A`
- Dicari eigenvector dengan metode Power Iteration
- Diambil `K` eigenvector dominan disebut EigenFace
- Gambar uji diproyeksikan ke ruang eigenface
- Jarak Euclidean dihitung untuk semua proyeksi wajah dataset
- Diambil wajah dengan jarak terkecil sebagai hasil pengenalan

## Catatan Penting
- Sistem hanya mengenali wajah yang ada di dataset
- Jika gambar uji adalah orang yang tidak dikenal, maka akan selalu “No Match”
- Jarak Euclidean akan lebih besar jika:
  - Angle wajah beda
  - Ekspresi beda
  - Pencahayaan sangat beda
- Ukuran gambar 100×100 --> threshold cocok: 100000 – 300000 