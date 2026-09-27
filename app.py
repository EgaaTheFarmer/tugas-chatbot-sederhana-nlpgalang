import streamlit as st
import joblib
import re
import os

# ============================================================
# KONFIGURASI
# ============================================================
st.set_page_config(page_title="Chatbot Medis", layout="centered")

CONFIDENCE_THRESHOLD = 0.40  # di atas ini langsung tampilkan hasil
MAX_FOLLOW_UP = 3            # maksimal pertanyaan lanjutan


# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(base_dir, "svm_model.joblib"))
    tfidf = joblib.load(os.path.join(base_dir, "tfidf_vectorizer.joblib"))
    le = joblib.load(os.path.join(base_dir, "label_encoder.joblib"))
    return model, tfidf, le


# ============================================================
# KAMUS GEJALA: Indonesian -> English
# Diurutkan dari frasa terpanjang agar matching lebih akurat
# ============================================================
SYMPTOM_MAP = {
    # -- Frasa panjang (prioritas match) --
    "sering buang air kecil": "frequent urination",
    "tekanan darah tinggi": "high blood pressure hypertension",
    "gula darah tinggi": "high blood sugar diabetes",
    "berat badan turun": "weight loss",
    "nafsu makan turun": "loss of appetite",
    "luka sulit sembuh": "slow healing wounds",
    "air kencing keruh": "cloudy urine",
    "kulit mengelupas": "peeling skin",
    "hidung tersumbat": "nasal congestion",
    "keringat dingin": "cold sweats",
    "pandangan kabur": "blurred vision",
    "sensitif cahaya": "sensitivity to light photophobia",
    "sakit tenggorokan": "sore throat",
    "tenggorokan sakit": "sore throat",
    "sulit bergerak": "difficulty moving reduced mobility",
    "sulit menelan": "difficulty swallowing",
    "sulit tidur": "insomnia difficulty sleeping",
    "sesak napas": "difficulty breathing shortness of breath",
    "sesak nafas": "difficulty breathing shortness of breath",
    "sakit kepala": "headache",
    "nyeri sendi": "joint pain",
    "sakit sendi": "joint pain",
    "sakit perut": "stomach pain abdominal pain",
    "nyeri perut": "stomach pain abdominal pain",
    "perut perih": "burning stomach pain",
    "hidung meler": "runny nose",
    "bercak merah": "red patches rash",
    "kulit kering": "dry skin",
    "kulit kuning": "yellow skin jaundice",
    "kulit gatal": "itchy skin",
    "mata kuning": "yellow eyes jaundice",
    "mata berair": "watery eyes tearing",
    "nyeri dada": "chest pain",
    "nyeri otot": "muscle pain",
    "demam tinggi": "high fever",
    "darah tinggi": "high blood pressure",
    "gula darah": "blood sugar",
    "badan pegal": "body aches",
    "leher kaku": "stiff neck",
    "sering haus": "excessive thirst",
    "sering kencing": "frequent urination",
    "ruam kulit": "skin rash",
    "nyeri kencing": "painful urination burning urination",
    "sakit kencing": "painful urination",
    "napas berbunyi": "wheezing",
    # -- Kata tunggal --
    "demam": "fever",
    "panas": "fever",
    "batuk": "cough",
    "pilek": "cold runny nose",
    "bersin": "sneezing",
    "bengek": "wheezing",
    "mengi": "wheezing",
    "gatal": "itching itchy",
    "ruam": "rash",
    "bersisik": "scaly patches",
    "pusing": "dizziness",
    "mual": "nausea",
    "muntah": "vomiting",
    "bengkak": "swelling swollen",
    "kaku": "stiffness",
    "pegal": "aches body aches",
    "lelah": "fatigue tired",
    "capek": "fatigue",
    "lemas": "weakness",
    "menggigil": "chills shivering",
    "keringat": "sweating",
    "kembung": "bloating",
    "diare": "diarrhea",
    "mencret": "diarrhea",
    "sembelit": "constipation",
    "mulas": "stomach cramps heartburn",
    "jerawat": "acne pimples",
    "komedo": "blackheads whiteheads",
    "lepuhan": "blisters",
    "keropeng": "crusting scabs",
    "bentol": "hives bumps",
    "perdarahan": "bleeding",
    "kesemutan": "tingling numbness",
    "varises": "varicose veins",
    "wasir": "hemorrhoids piles",
    "ambeien": "hemorrhoids piles",
    "merokok": "smoking cigarettes",
    "alergi": "allergy allergic reaction",
    "asma": "asthma",
    "migrain": "migraine severe headache",
    "diabetes": "diabetes",
    "infeksi": "infection",
    "radang": "inflammation",
    "nyeri": "pain",
    "sakit": "pain",
    "terbakar": "burning sensation",
    "perih": "burning stinging",
    "haus": "thirsty excessive thirst",
    "berdenyut": "throbbing pulsating",
    "berminyak": "oily skin",
    "dahak": "phlegm sputum",
    "berdarah": "bleeding blood",
    "kuning": "yellow jaundice",
    "gatal-gatal": "itching rash hives",
}


# ============================================================
# PERTANYAAN LANJUTAN PER PENYAKIT
# q = pertanyaan, yes_add = keyword English yang ditambahkan jika jawaban Ya
# ============================================================
FOLLOW_UP = {
    "Asma Bronkial": [
        {"q": "Apakah sesak napas atau bengek lebih sering muncul di malam hari atau dini hari?",
         "yes_add": "breathing difficulty worse at night early morning nocturnal asthma"},
        {"q": "Apakah Anda merokok atau sering terpapar asap rokok?",
         "yes_add": "smoking exposure cigarette smoke trigger"},
        {"q": "Apakah ada riwayat asma atau alergi dalam keluarga?",
         "yes_add": "family history asthma allergy hereditary"},
    ],
    "Radang Sendi": [
        {"q": "Apakah nyeri sendi terasa lebih kaku di pagi hari setelah bangun tidur?",
         "yes_add": "morning stiffness joint pain worse in morning"},
        {"q": "Apakah sendi yang nyeri terlihat bengkak atau kemerahan?",
         "yes_add": "swollen joints redness inflammation"},
        {"q": "Apakah nyeri terjadi di lebih dari satu sendi?",
         "yes_add": "multiple joints affected pain in several joints"},
    ],
    "Demam Berdarah": [
        {"q": "Apakah demam muncul secara mendadak dan sangat tinggi?",
         "yes_add": "sudden onset high fever acute"},
        {"q": "Apakah ada nyeri di belakang mata?",
         "yes_add": "pain behind eyes retro orbital"},
        {"q": "Apakah muncul bintik-bintik merah pada kulit?",
         "yes_add": "red spots rash petechiae skin"},
    ],
    "Flu / Pilek": [
        {"q": "Apakah disertai sakit tenggorokan atau hidung tersumbat?",
         "yes_add": "sore throat nasal congestion blocked nose"},
        {"q": "Apakah ada demam ringan yang tidak terlalu tinggi?",
         "yes_add": "mild fever low grade"},
        {"q": "Apakah gejala baru muncul dalam 1-2 minggu terakhir?",
         "yes_add": "recent onset started recently"},
    ],
    "GERD / Asam Lambung": [
        {"q": "Apakah terasa panas atau terbakar di area dada atau ulu hati?",
         "yes_add": "burning chest heartburn epigastric"},
        {"q": "Apakah gejala memburuk setelah makan atau saat berbaring?",
         "yes_add": "worse after eating lying down reflux"},
        {"q": "Apakah kadang terasa asam atau pahit di mulut?",
         "yes_add": "sour taste acid regurgitation"},
    ],
    "Migrain": [
        {"q": "Apakah sakit kepala terasa berdenyut dan biasanya di satu sisi kepala?",
         "yes_add": "throbbing headache one sided pulsating"},
        {"q": "Apakah disertai mual atau sensitif terhadap cahaya terang?",
         "yes_add": "nausea sensitivity to light photophobia"},
        {"q": "Apakah sakit kepala bisa berlangsung berjam-jam?",
         "yes_add": "headache lasting hours prolonged"},
    ],
    "Diabetes": [
        {"q": "Apakah Anda sering merasa sangat haus dan sering buang air kecil?",
         "yes_add": "excessive thirst frequent urination polyuria polydipsia"},
        {"q": "Apakah ada penurunan berat badan tanpa sebab yang jelas?",
         "yes_add": "unexplained weight loss"},
        {"q": "Apakah luka pada kulit sembuh lebih lama dari biasanya?",
         "yes_add": "slow wound healing"},
    ],
    "Psoriasis": [
        {"q": "Apakah bercak kulit terasa tebal, kering, dan bersisik keperakan?",
         "yes_add": "thick dry scaly patches silvery scales"},
        {"q": "Apakah bercak sering muncul di area siku, lutut, atau kulit kepala?",
         "yes_add": "patches elbows knees scalp"},
        {"q": "Apakah ada anggota keluarga yang memiliki kondisi kulit serupa?",
         "yes_add": "family history skin condition hereditary"},
    ],
    "Tifus": [
        {"q": "Apakah demam naik secara bertahap dari hari ke hari?",
         "yes_add": "gradually increasing fever stepwise"},
        {"q": "Apakah disertai sakit perut dan gangguan pencernaan?",
         "yes_add": "abdominal pain digestive problems"},
        {"q": "Apakah Anda sangat lemas dan kehilangan nafsu makan?",
         "yes_add": "severe weakness loss of appetite"},
    ],
    "Malaria": [
        {"q": "Apakah demam disertai menggigil hebat dan berkeringat?",
         "yes_add": "chills rigors sweating cyclical"},
        {"q": "Apakah Anda pernah ke daerah yang banyak nyamuk?",
         "yes_add": "mosquito endemic area exposure"},
        {"q": "Apakah demam muncul secara periodik (setiap 2-3 hari)?",
         "yes_add": "periodic fever cyclical recurring"},
    ],
    "Cacar Air": [
        {"q": "Apakah ruam berbentuk gelembung kecil berisi cairan yang sangat gatal?",
         "yes_add": "fluid filled blisters vesicles very itchy"},
        {"q": "Apakah ruam menyebar dari wajah ke seluruh tubuh?",
         "yes_add": "rash spreading face trunk whole body"},
        {"q": "Apakah Anda belum pernah terkena cacar air sebelumnya?",
         "yes_add": "no prior infection first time"},
    ],
    "Pneumonia / Radang Paru": [
        {"q": "Apakah batuk disertai dahak berwarna kuning, hijau, atau berdarah?",
         "yes_add": "cough phlegm sputum yellow green bloody"},
        {"q": "Apakah disertai demam tinggi dan nyeri dada saat bernapas?",
         "yes_add": "high fever chest pain breathing pleuritic"},
        {"q": "Apakah napas terasa cepat dan dangkal?",
         "yes_add": "rapid shallow breathing tachypnea"},
    ],
    "Hipertensi / Darah Tinggi": [
        {"q": "Apakah sering sakit kepala terutama di bagian belakang?",
         "yes_add": "headache back of head occipital"},
        {"q": "Apakah pernah cek tekanan darah dan hasilnya tinggi?",
         "yes_add": "high blood pressure measured elevated"},
        {"q": "Apakah kadang pandangan kabur atau telinga berdenging?",
         "yes_add": "blurred vision ringing ears tinnitus"},
    ],
    "Tukak Lambung": [
        {"q": "Apakah nyeri perut terasa perih seperti terbakar di ulu hati?",
         "yes_add": "burning stomach pain epigastric gnawing"},
        {"q": "Apakah nyeri terkait waktu makan (membaik/memburuk setelah makan)?",
         "yes_add": "pain related meals eating food"},
        {"q": "Apakah Anda sering minum obat penghilang nyeri (ibuprofen, aspirin)?",
         "yes_add": "NSAID use painkiller ibuprofen"},
    ],
    "Infeksi Saluran Kemih": [
        {"q": "Apakah terasa nyeri atau panas saat buang air kecil?",
         "yes_add": "burning pain urination dysuria"},
        {"q": "Apakah merasa ingin kencing terus meskipun hanya sedikit?",
         "yes_add": "frequent urge urinate urgency"},
        {"q": "Apakah urine terlihat keruh atau berbau tidak biasa?",
         "yes_add": "cloudy urine foul smelling"},
    ],
    "Alergi": [
        {"q": "Apakah gejala muncul setelah terpapar sesuatu (debu, makanan, obat)?",
         "yes_add": "triggered exposure allergen dust food"},
        {"q": "Apakah disertai bersin-bersin, mata berair, atau hidung gatal?",
         "yes_add": "sneezing watery eyes itchy nose allergic rhinitis"},
        {"q": "Apakah gejala membaik setelah menjauhi pemicunya?",
         "yes_add": "improves avoiding trigger"},
    ],
    "Infeksi Jamur": [
        {"q": "Apakah area yang gatal terasa lembab, misalnya di lipatan kulit?",
         "yes_add": "moist area skin folds groin armpit"},
        {"q": "Apakah ada bentuk seperti cincin atau perubahan warna di kulit?",
         "yes_add": "ring shaped discoloration circular rash ringworm"},
        {"q": "Apakah kuku menebal, rapuh, atau berubah warna?",
         "yes_add": "thick brittle discolored nails fungal"},
    ],
    "Jerawat": [
        {"q": "Apakah masalah kulit terutama di area wajah, punggung, atau dada?",
         "yes_add": "face back chest acne prone areas"},
        {"q": "Apakah ada komedo hitam atau putih selain bintik merah?",
         "yes_add": "blackheads whiteheads comedones"},
        {"q": "Apakah kulit Anda cenderung berminyak?",
         "yes_add": "oily skin sebaceous"},
    ],
    "Reaksi Obat": [
        {"q": "Apakah gejala muncul setelah mengonsumsi obat tertentu?",
         "yes_add": "symptoms after medication drug adverse reaction"},
        {"q": "Apakah muncul ruam, gatal, atau bengkak setelah minum obat?",
         "yes_add": "rash itching swelling after medication allergic"},
        {"q": "Apakah gejala membaik setelah berhenti minum obat tersebut?",
         "yes_add": "improved stopping medication"},
    ],
    "Varises": [
        {"q": "Apakah terlihat pembuluh darah menonjol dan berkelok di kaki?",
         "yes_add": "visible bulging twisted veins legs"},
        {"q": "Apakah kaki terasa berat terutama setelah berdiri lama?",
         "yes_add": "heavy aching legs worse standing"},
        {"q": "Apakah ada bengkak di pergelangan kaki di sore/malam hari?",
         "yes_add": "ankle swelling evening edema"},
    ],
    "Wasir / Ambeien": [
        {"q": "Apakah ada darah segar saat buang air besar?",
         "yes_add": "fresh blood bowel movement rectal bleeding"},
        {"q": "Apakah terasa ada benjolan di area dubur?",
         "yes_add": "lump around anus rectal mass"},
        {"q": "Apakah gatal atau nyeri di area dubur?",
         "yes_add": "itching pain anus rectal discomfort"},
    ],
    "Penyakit Kuning": [
        {"q": "Apakah kulit atau bagian putih mata terlihat menguning?",
         "yes_add": "yellow skin yellowing eyes scleral icterus"},
        {"q": "Apakah urine berwarna gelap seperti teh?",
         "yes_add": "dark urine tea colored"},
        {"q": "Apakah disertai mual, lemas, atau tidak nafsu makan?",
         "yes_add": "nausea fatigue loss of appetite"},
    ],
    "Impetigo": [
        {"q": "Apakah luka kulit dimulai sebagai lepuhan kecil lalu pecah dan berkeropeng?",
         "yes_add": "small blisters burst crusting honey colored scabs"},
        {"q": "Apakah luka terutama di area wajah, sekitar hidung dan mulut?",
         "yes_add": "sores around nose mouth face"},
        {"q": "Apakah luka menyebar atau menular ke orang lain?",
         "yes_add": "spreading contagious new sores"},
    ],
    "Spondilosis Servikal": [
        {"q": "Apakah nyeri atau kaku terutama di area leher?",
         "yes_add": "neck pain stiffness cervical"},
        {"q": "Apakah ada kesemutan atau mati rasa menjalar ke tangan/lengan?",
         "yes_add": "tingling numbness radiating arms hands"},
        {"q": "Apakah nyeri bertambah saat menunduk atau menoleh?",
         "yes_add": "pain worse bending turning head neck movement"},
    ],
}


# ============================================================
# INFORMASI PENYAKIT (Bahasa Indonesia)
# ============================================================
INFO_PENYAKIT = {
    "Jerawat": {
        "deskripsi": "Jerawat adalah kondisi kulit yang terjadi ketika pori-pori tersumbat oleh minyak dan sel kulit mati, menyebabkan komedo, bintik merah, atau pustula.",
        "saran": "Jaga kebersihan wajah, hindari menyentuh wajah terlalu sering, dan gunakan produk perawatan yang sesuai dengan jenis kulit Anda."
    },
    "Radang Sendi": {
        "deskripsi": "Radang sendi (arthritis) adalah peradangan pada persendian yang menyebabkan nyeri, bengkak, dan kaku, terutama saat bergerak.",
        "saran": "Lakukan olahraga ringan secara teratur, jaga berat badan ideal, dan kompres hangat atau dingin untuk meredakan nyeri."
    },
    "Asma Bronkial": {
        "deskripsi": "Asma bronkial adalah kondisi kronis di mana saluran pernapasan menyempit dan meradang, menyebabkan sesak napas, batuk, dan bengek (wheezing).",
        "saran": "Hindari pemicu seperti debu, asap, dan alergen. Jika Anda merokok, sangat disarankan untuk berhenti. Selalu siapkan inhaler sesuai resep dokter."
    },
    "Spondilosis Servikal": {
        "deskripsi": "Spondilosis servikal adalah keausan pada bantalan tulang belakang di area leher yang menyebabkan nyeri, kaku, dan kadang kesemutan di tangan.",
        "saran": "Jaga postur tubuh yang baik, lakukan peregangan leher secara berkala, dan hindari duduk terlalu lama dalam satu posisi."
    },
    "Cacar Air": {
        "deskripsi": "Cacar air adalah infeksi virus yang sangat menular, ditandai dengan ruam gatal berisi cairan di seluruh tubuh disertai demam.",
        "saran": "Istirahat yang cukup, jaga hidrasi, dan hindari menggaruk lepuhan agar tidak meninggalkan bekas. Losion calamine dapat membantu mengurangi rasa gatal."
    },
    "Flu / Pilek": {
        "deskripsi": "Flu atau pilek adalah infeksi virus pada saluran pernapasan atas yang menyebabkan hidung meler, bersin, batuk, dan sakit tenggorokan.",
        "saran": "Istirahat yang cukup, minum air hangat, dan konsumsi makanan bergizi. Gejala biasanya membaik dalam 7-10 hari."
    },
    "Demam Berdarah": {
        "deskripsi": "Demam berdarah (dengue) adalah infeksi virus yang ditularkan nyamuk Aedes, menyebabkan demam tinggi mendadak, nyeri di belakang mata, nyeri sendi, dan bintik merah.",
        "saran": "Jaga hidrasi tubuh, istirahat total, dan pantau tanda bahaya seperti muntah terus-menerus, nyeri perut hebat, atau mimisan. Segera ke rumah sakit jika kondisi memburuk."
    },
    "Wasir / Ambeien": {
        "deskripsi": "Wasir adalah pembengkakan pembuluh darah di area rektum yang menyebabkan nyeri, perdarahan saat BAB, dan rasa tidak nyaman.",
        "saran": "Konsumsi makanan tinggi serat, minum air putih yang cukup, dan hindari mengejan saat buang air besar."
    },
    "Infeksi Jamur": {
        "deskripsi": "Infeksi jamur dapat menyerang kulit, kuku, atau bagian tubuh lain, menyebabkan gatal, kemerahan, dan perubahan warna kulit.",
        "saran": "Jaga area yang terinfeksi tetap bersih dan kering. Gunakan krim antijamur sesuai petunjuk dan hindari berbagi barang pribadi."
    },
    "Hipertensi / Darah Tinggi": {
        "deskripsi": "Hipertensi adalah kondisi di mana tekanan darah terhadap dinding arteri terlalu tinggi secara konsisten, sering tanpa gejala yang jelas.",
        "saran": "Kurangi konsumsi garam, olahraga teratur, kelola stres, dan periksakan tekanan darah secara rutin."
    },
    "Impetigo": {
        "deskripsi": "Impetigo adalah infeksi bakteri kulit yang menular, terutama pada anak-anak, menyebabkan luka merah yang pecah dan membentuk keropeng kekuningan.",
        "saran": "Jaga kebersihan area yang terinfeksi, hindari menggaruk, dan habiskan seluruh dosis antibiotik yang diresepkan dokter."
    },
    "Penyakit Kuning": {
        "deskripsi": "Penyakit kuning (jaundice) adalah kondisi di mana kulit dan mata menguning karena kadar bilirubin tinggi, biasanya berkaitan dengan gangguan hati.",
        "saran": "Periksakan ke dokter untuk mengetahui penyebabnya. Jaga hidrasi dan konsumsi makanan yang mudah dicerna."
    },
    "Malaria": {
        "deskripsi": "Malaria adalah penyakit yang ditularkan melalui gigitan nyamuk Anopheles, menyebabkan demam periodik, menggigil hebat, dan keringat.",
        "saran": "Segera periksakan ke dokter untuk tes darah dan pengobatan. Gunakan kelambu dan obat nyamuk untuk pencegahan."
    },
    "Migrain": {
        "deskripsi": "Migrain adalah sakit kepala intens yang biasanya berdenyut di satu sisi, sering disertai mual, muntah, dan sensitif terhadap cahaya.",
        "saran": "Identifikasi dan hindari pemicu (stres, kurang tidur, makanan tertentu). Istirahat di ruangan gelap saat serangan terjadi."
    },
    "Pneumonia / Radang Paru": {
        "deskripsi": "Pneumonia adalah infeksi yang menyebabkan peradangan pada kantung udara di paru-paru, disertai batuk berdahak, demam tinggi, dan sesak napas.",
        "saran": "Segera periksakan ke dokter. Istirahat total, jaga hidrasi, dan habiskan seluruh pengobatan yang diberikan."
    },
    "Psoriasis": {
        "deskripsi": "Psoriasis adalah kondisi autoimun kronis yang menyebabkan penumpukan sel kulit secara cepat, menghasilkan bercak tebal, merah, dan bersisik keperakan.",
        "saran": "Jaga kelembaban kulit, hindari pemicu seperti stres, dan ikuti rencana pengobatan dari dokter kulit."
    },
    "Tifus": {
        "deskripsi": "Tifus (typhoid) adalah infeksi bakteri Salmonella typhi melalui makanan atau air terkontaminasi, menyebabkan demam bertahap, lemas, dan gangguan pencernaan.",
        "saran": "Segera periksakan ke dokter. Minum air bersih, jaga kebersihan makanan, dan habiskan antibiotik yang diresepkan."
    },
    "Varises": {
        "deskripsi": "Varises adalah pembuluh darah vena yang membesar dan berkelok-kelok, umumnya di kaki, akibat katup vena yang melemah.",
        "saran": "Tinggikan kaki saat istirahat, olahraga teratur, hindari berdiri terlalu lama, dan gunakan stoking kompresi jika disarankan dokter."
    },
    "Alergi": {
        "deskripsi": "Alergi terjadi ketika sistem kekebalan tubuh bereaksi berlebihan terhadap zat yang sebenarnya tidak berbahaya seperti debu, serbuk sari, atau makanan tertentu.",
        "saran": "Identifikasi dan hindari alergen pemicu, gunakan antihistamin jika diperlukan, dan konsultasikan ke dokter untuk penanganan jangka panjang."
    },
    "Diabetes": {
        "deskripsi": "Diabetes adalah kondisi metabolik di mana tubuh tidak dapat mengatur kadar gula darah dengan baik, menyebabkan berbagai komplikasi jika tidak ditangani.",
        "saran": "Pantau gula darah secara rutin, ikuti pola makan seimbang, olahraga teratur, dan minum obat sesuai resep dokter."
    },
    "Reaksi Obat": {
        "deskripsi": "Reaksi obat adalah efek samping yang tidak diinginkan setelah mengonsumsi obat tertentu, bisa berupa ruam, gatal, bengkak, hingga sesak napas.",
        "saran": "Hentikan obat yang dicurigai dan segera periksakan ke dokter. Catat nama obat yang menimbulkan reaksi untuk menghindari di kemudian hari."
    },
    "GERD / Asam Lambung": {
        "deskripsi": "GERD adalah kondisi di mana asam lambung sering naik ke kerongkongan, menyebabkan rasa terbakar di dada (heartburn) dan ketidaknyamanan.",
        "saran": "Hindari makanan pedas dan berlemak, makan dalam porsi kecil, jangan langsung berbaring setelah makan, dan tinggikan kepala saat tidur."
    },
    "Tukak Lambung": {
        "deskripsi": "Tukak lambung adalah luka terbuka pada lapisan lambung atau usus halus bagian atas yang menyebabkan nyeri perut seperti terbakar.",
        "saran": "Hindari obat anti-inflamasi (NSAID), alkohol, dan makanan yang mengiritasi. Makan teratur dan ikuti pengobatan dari dokter."
    },
    "Infeksi Saluran Kemih": {
        "deskripsi": "Infeksi saluran kemih (ISK) adalah infeksi pada sistem kemih yang menyebabkan nyeri saat kencing, sering ingin kencing, dan urine keruh.",
        "saran": "Minum banyak air putih, jaga kebersihan area genital, dan habiskan seluruh dosis antibiotik yang diresepkan."
    },
}


# ============================================================
# FUNGSI INTI
# ============================================================

def translate_symptoms(text_id):
    """Konversi teks gejala Indonesia ke English menggunakan kamus keyword.
    Lebih stabil dan cepat dibanding API terjemahan online."""
    text = text_id.lower().strip()
    en_parts = []

    # Urutkan keyword dari yang terpanjang agar frasa multi-kata cocok duluan
    sorted_keys = sorted(SYMPTOM_MAP.keys(), key=len, reverse=True)

    for key in sorted_keys:
        if key in text:
            en_parts.append(SYMPTOM_MAP[key])
            text = text.replace(key, " ")

    remaining = re.sub(r'\s+', ' ', text).strip()
    if remaining:
        en_parts.append(remaining)

    result = " ".join(en_parts)
    return result.strip() if result.strip() else text_id


def clean_text(text):
    """Preprocessing teks - harus sama dengan saat training."""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def predict_top_n(text_en, model, tfidf, le, n=3):
    """Prediksi top-N penyakit beserta probabilitasnya."""
    text_clean = clean_text(text_en)
    text_tfidf = tfidf.transform([text_clean])
    probas = model.predict_proba(text_tfidf)[0]
    top_idx = probas.argsort()[-n:][::-1]
    results = []
    for idx in top_idx:
        disease = le.inverse_transform([idx])[0]
        results.append((disease, float(probas[idx])))
    return results


def get_follow_up_questions(top3):
    """Ambil pertanyaan lanjutan berdasarkan prediksi teratas."""
    questions = []
    for rank, (disease, prob) in enumerate(top3[:2]):
        if disease in FOLLOW_UP:
            n_qs = 2 if rank == 0 else 1
            for q in FOLLOW_UP[disease][:n_qs]:
                entry = q.copy()
                entry["disease"] = disease
                questions.append(entry)
    return questions[:MAX_FOLLOW_UP]


YES_WORDS = {"ya", "iya", "iyah", "yap", "benar", "betul", "bener", "yoi", "yes", "y", "iy"}
NO_WORDS = {"tidak", "nggak", "ngga", "enggak", "nda", "ga", "gak", "no", "n", "tdk", "bukan", "blm", "belum"}


def parse_answer(text):
    """Parse jawaban follow-up: 'yes', 'no', atau 'info' + teks tambahan."""
    words = text.lower().strip().split()
    if not words:
        return "info", ""

    first = words[0]
    extra = " ".join(words[1:]).strip(" ,.")

    if first in YES_WORDS:
        return "yes", extra
    elif first in NO_WORDS:
        return "no", extra
    else:
        return "info", text.lower().strip()


def format_diagnosis(penyakit):
    """Format respons diagnosis akhir."""
    respons = f"Berdasarkan gejala dan informasi yang Anda sampaikan, kondisi Anda kemungkinan berkaitan dengan **{penyakit}**.\n\n"

    if penyakit in INFO_PENYAKIT:
        info = INFO_PENYAKIT[penyakit]
        respons += f"**Apa itu {penyakit}?**\n{info['deskripsi']}\n\n"
        respons += f"**Saran umum:**\n{info['saran']}\n\n"

    return respons


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("Chatbot Medis")
st.caption("Deskripsikan keluhan atau gejala Anda, dan saya akan membantu mengidentifikasi kemungkinan kondisi Anda.")

with st.sidebar:
    st.header("Tentang Aplikasi")
    st.write(
        "Chatbot ini menggunakan model machine learning (TF-IDF + SVM) "
        "yang dilatih pada dataset Symptom2Disease untuk memprediksi "
        "kemungkinan penyakit berdasarkan deskripsi gejala."
    )
    st.write("**Dataset:** Symptom2Disease (Kaggle)")
    st.write("**Jumlah data:** 1.200 entri")
    st.write("**Penyakit:** 24 kondisi")
    st.divider()
    st.subheader("Daftar Penyakit")
    for p in sorted(INFO_PENYAKIT.keys()):
        st.write(f"- {p}")
    st.divider()
    st.caption("Untuk keperluan edukasi.")

# Load model
try:
    model, tfidf, le = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Gagal memuat model. Pastikan file .joblib ada di folder yang sama dengan app.py.\n\nError: {e}")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": (
            "Halo, saya chatbot medis yang dapat membantu mengidentifikasi "
            "kemungkinan penyakit berdasarkan gejala Anda.\n\n"
            "Silakan ceritakan keluhan yang Anda rasakan. Contoh:\n"
            "- *\"Saya sesak napas dan kadang bengek, terutama malam hari\"*\n"
            "- *\"Kulit saya muncul bercak merah yang gatal dan bersisik\"*\n"
            "- *\"Saya demam tinggi, menggigil, dan badan pegal sudah 3 hari\"*"
        )}
    ]

if "diag" not in st.session_state:
    st.session_state.diag = {
        "active": False,
        "symptoms_en": "",
        "top3": [],
        "questions": [],
        "q_idx": 0,
    }

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle input
if prompt := st.chat_input("Ceritakan keluhan Anda di sini..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = ""
    diag = st.session_state.diag

    if not model_loaded:
        response = "Model belum dimuat. Silakan periksa pesan error di atas."

    elif not diag["active"]:
        # ---- INPUT GEJALA BARU ----
        symptoms_en = translate_symptoms(prompt)
        symptoms_clean = clean_text(symptoms_en)

        if len(symptoms_clean.split()) < 2:
            response = (
                "Mohon ceritakan gejala Anda dengan lebih detail agar saya dapat membantu.\n\n"
                "Contoh: *\"Saya sering bersin, hidung meler, dan mata berair sudah seminggu.\"*"
            )
        else:
            top3 = predict_top_n(symptoms_en, model, tfidf, le, 3)
            top_disease, top_prob = top3[0]

            if top_prob >= CONFIDENCE_THRESHOLD:
                response = format_diagnosis(top_disease)
            else:
                # Perlu follow-up
                questions = get_follow_up_questions(top3)
                if questions:
                    diag["active"] = True
                    diag["symptoms_en"] = symptoms_en
                    diag["top3"] = top3
                    diag["questions"] = questions
                    diag["q_idx"] = 0

                    first_q = questions[0]["q"]
                    response = (
                        "Saya mencatat gejala Anda. Untuk membantu mempersempit kemungkinan, "
                        "saya perlu menanyakan beberapa hal.\n\n"
                        f"**{first_q}**"
                    )
                else:
                    response = format_diagnosis(top_disease)

    else:
        # ---- JAWABAN FOLLOW-UP ----
        answer_type, extra_text = parse_answer(prompt)
        current_q = diag["questions"][diag["q_idx"]]

        # Tambahkan info berdasarkan jawaban
        if answer_type == "yes":
            diag["symptoms_en"] += " " + current_q.get("yes_add", "")
        if answer_type == "info":
            diag["symptoms_en"] += " " + translate_symptoms(extra_text)
        if extra_text and answer_type != "info":
            diag["symptoms_en"] += " " + translate_symptoms(extra_text)

        diag["q_idx"] += 1

        if diag["q_idx"] < len(diag["questions"]):
            # Masih ada pertanyaan
            next_q = diag["questions"][diag["q_idx"]]["q"]
            ack = "Baik, terima kasih." if answer_type == "yes" else "Baik, dicatat."
            response = f"{ack} Selanjutnya:\n\n**{next_q}**"
        else:
            # Semua pertanyaan sudah dijawab, prediksi ulang
            top3_new = predict_top_n(diag["symptoms_en"], model, tfidf, le, 3)
            final_disease = top3_new[0][0]
            response = "Terima kasih atas informasinya.\n\n" + format_diagnosis(final_disease)

            # Reset state
            diag["active"] = False
            diag["symptoms_en"] = ""
            diag["top3"] = []
            diag["questions"] = []
            diag["q_idx"] = 0

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
