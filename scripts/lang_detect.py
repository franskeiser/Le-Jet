import re, pandas as pd
TL = set("""ang ng mga sa na ka mo ko po opo ito iyan iyon yung yun ay ba pa pang kayo kayong namin natin niyo ninyo inyo inyong iyong
iyo ikaw ako ako'y siya sila kami tayo kung pero dahil kasi kaya hindi di wala walang mayroon meron ngayon bukas kahapon mamaya
salamat maraming magandang umaga hapon gabi tanghali araw buwan linggo lang lamang din rin daw raw nga naman muna pa pala sana
para upang at ni nina kay kina dito doon diyan nito niyan nila sayo sa'yo sa'yong akin amin atin ating aming namin mag nag ma
makakuha makakakuha bayaran magbayad bayad bago hanggang habang hanggang sobrang libre ngayong ngayon nakabinbin naka-hold
kumpirmahin i-verify i-click i-claim i-update tingnan silipin basahin buksan pindutin ilagay punan mag-login mag-sign magpatuloy
huwag wag 'wag paki pakiclick pakiusap kailangan kailangang dapat pwede puwede maaari nanalo panalo napili binabati
tulong ayuda sahod sweldo bahay opisina kapatid kuya ate beh pre sis tita tito mama papa anak
dumating naipadala ipinadala natanggap tanggap nakatanggap nakita napansin nag-expire mag-expire matatanggap makukuha
may""".split()) - {"may","at","pa","mag","ma","ni","ay","ba","sa","na","mo","ko","ka","di","din","po"} | {"po","yung","mga","ng","ang","kayo","namin","natin","niyo","ninyo","lang","daw","raw","kasi","dahil","para","hindi","wala","ngayon","bukas","naman","muna","dito","mo","ko","ka","ba","din","rin","nga","pala","sana","kung","pero","sa'yo","sayo","sa'yong","iyong","inyong","ito"}
# ambiguous with English / tiny: kept out: may, at, pa, mag, ma, ni, ay, sa, na, di
STRONG = TL
def tl_ratio(text):
    text = re.sub(r"DITO|Dito (Mobile|Home|Telecom|app|App|Level|Prepaid|SIM)", " ", text)   # the telco brand, not the word "dito" (here)
    t = re.sub(r"https?://\S+|www\.\S+", " ", text.lower())
    w = re.findall(r"[a-zñ'\-]+", t)
    w = [x for x in w if len(x) > 0]
    if not w: return 0, 0, 0
    hits = sum(1 for x in w if x in TL or x.split("-")[-1] in TL and x.startswith(("i-","mag-","nag-","ma-")) and False)
    return hits, len(w), hits / len(w)
def classify(text):
    h, n, r = tl_ratio(text)
    if h == 0: return "english"
    if h == 1 and n >= 12: return "english"          # one stray word (e.g. "po") in a long English text
    if r >= 0.42: return "tagalog"
    return "taglish"
if __name__ == "__main__":
    s = pd.read_csv("synthetic_smishing_messages_v3.csv", keep_default_na=False, dtype=str)
    m = {"en": "english", "taglish": "taglish", "tl": "tagalog"}
    pred = s.message_text.map(classify); true = s.language.map(m)
    print("accuracy on the labeled smishing set:", (pred == true).mean().round(3))
    print(pd.crosstab(true, pred))

EN = set("""the your you to and is of for on in with please now click verify account link update parcel delivery login confirm claim bill payment
order promo voucher load reload online sign loan approved details number wallet password code offer sale discount today tomorrow
will has have are was be or by from this that it as at we our us if not no yes all new get free received message messages
text stop reply call visit apply register download open check view secure security verification identity card bank transfer
transaction pending expired expire suspended blocked limited locked access email address id delivered package shipping fee
refund cashback points reward prize winner raffle won congratulations job work hiring application interview salary employee
customer service support team notice reminder alert warning urgent immediately within hours days minutes before after until
here there when what why how who which can could should would may must need want use using used app mobile phone sim
data internet plan prepaid postpaid bonus gift promo code link""".split())
def en_ratio(text):
    text = re.sub(r"DITO", " ", text)
    t = re.sub(r"https?://\S+|www\.\S+", " ", text.lower())
    w = re.findall(r"[a-zñ'\-]+", t)
    hits = sum(1 for x in w if x in EN)
    return hits, len(w)
def classify2(text, th=0.9):
    h, n, r = tl_ratio(text)
    eh, _ = en_ratio(text)
    if h == 0 or (h == 1 and n >= 12): return "english"
    return "tagalog" if eh / max(h, 1) < th else "taglish"

# ---- v2: add the common function words I left out, and Tagalog affixes on English verbs (i-register, pa-check, mag-message)
TL |= {"na", "sa", "ay", "ni", "ka", "alamin", "paalala", "salamat", "eto", "kanina", "tayo", "kami", "natin", "tungkol", "ulit", "agad",
       "po", "opo", "tara", "nandito", "nandyan", "sabi", "sinabi", "ayos", "sige", "oks", "nyo", "ninyo", "kayo", "iyong", "inyong", "ko", "mo",
       "ako", "ikaw", "siya", "sila", "sana", "kapag", "pag", "habang", "mga", "lamang", "dapat", "puwede", "pwede", "pwedeng", "talaga", "sobra"}
_AFFIX = re.compile(r"^(i|ma|mag|nag|pa|na|ipa|pag|nakaka|maka|ka)-[a-z]{3,}$")
def tl_ratio(text):
    text = re.sub(r"\bDITO\b|\bDito (Mobile|Home|Telecom|app|App|Level|Prepaid|SIM)", " ", text)
    t = re.sub(r"https?://\S+|www\.\S+|\S+\.(com|ph|net|org|gov)\S*", " ", text.lower())
    w = re.findall(r"[a-zñ'\-]+", t)
    hits = sum(1 for x in w if x in TL or _AFFIX.match(x))
    return hits, len(w), (hits / len(w) if w else 0)

# ---- v3: address terms and the name Pag-IBIG are normal in Philippine English, so they do not make a text Taglish
IGNORE = {"pre", "ate", "kuya", "sis", "mama", "papa", "tita", "tito", "beh", "bro", "anak", "lola", "lolo", "pag-ibig", "ninang", "ninong", "inay", "itay"}
def tl_ratio(text):
    text = re.sub(r"\bDITO\b|\bDito (Mobile|Home|Telecom|app|App|Level|Prepaid|SIM)", " ", text)
    t = re.sub(r"https?://\S+|www\.\S+|\S+\.(com|ph|net|org|gov)\S*", " ", text.lower())
    w = re.findall(r"[a-zñ'\-]+", t)
    hits = sum(1 for x in w if x not in IGNORE and (x in TL or _AFFIX.match(x)))
    return hits, len(w), (hits / len(w) if w else 0)
def classify3(text, th=0.1):
    h, n, r = tl_ratio(text)
    if h == 0: return "english"
    eh, _ = en_ratio(text)
    return "tagalog" if eh / h < th else "taglish"
