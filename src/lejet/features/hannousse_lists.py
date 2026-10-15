"""
hannousse_lists.py -- the word lists behind a few of the 56 URL-based features.

The dataset's authors took these lists from outside sources (101domain.com brands, Spamhaus and
Broadcom suspicious TLDs, a published list of top phishing hosts, URLTeam shorteners) and did not
publish the exact files. What is here:
  * SHORTENER_RE, PHISH_HINTS, STAT_DOMAIN_RE: the lists of the dataset's own extractor.
  * BRANDS, SUSPICIOUS_TLDS, STAT_HOSTS: RECONSTRUCTED from the 11,430 labelled rows (every domain / TLD / host
    for which the public set has the flag set). They reproduce the public set exactly but know nothing about
    brands or TLDs that are not in it. See docs/hannousse_87_features.md.
"""

import re

SHORTENER_RE = re.compile(
    r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|"
    r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|"
    r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|"
    r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|"
    r"qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|"
    r"po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|"
    r"prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|"
    r"tr\.im|link\.zip\.net"
)

PHISH_HINTS = ("wp", "login", "includes", "admin", "content", "site", "images", "js", "alibaba", "css",
               "myaccount", "dropbox", "themes", "plugins", "signin", "view")

STAT_DOMAIN_RE = re.compile(
    r"at\.ua|usa\.cc|baltazarpresentes\.com\.br|pe\.hu|esy\.es|hol\.es|sweddy\.com|myjino\.ru|96\.lt|ow\.ly")

# Registrable domain (or raw IP) -> the dataset's statistical_report value (1 = matched a top phishing host
# pattern, 2 = its address matched the published list of top phishing IPs). A 2020 snapshot.
STAT_HOSTS = {
    '103.229.125.10': 2, '103.75.1.198': 2, '104.237.152.236': 2, '106.12.192.247': 2, '107.180.44.78': 2,
    '107.189.6.150': 2, '108.166.202.103': 2, '113.125.21.66': 2, '124.156.218.50': 2, '129.226.227.72': 2,
    '138.36.41.142': 2, '139.59.201.27': 2, '143.95.251.183': 2, '144.172.64.51': 2, '144.172.64.57': 2,
    '148.204.63.249': 2, '149.210.143.165': 2, '157.240.2.20': 2, '159.203.115.201': 2, '159.65.133.234': 2,
    '162.213.251.24': 2, '165.22.103.235': 2, '165.227.4.30': 2, '172.217.21.162': 2, '173.199.186.15': 2,
    '173.212.240.54': 2, '174.138.36.47': 2, '174.139.46.123': 2, '174.139.46.28': 2, '178.33.67.12': 2,
    '179.185.89.94': 2, '185.136.204.3': 2, '185.17.106.194': 2, '189.36.52.206': 2, '193.135.153.242': 2,
    '199.192.18.162': 2, '2.136.95.251': 2, '200.113.195.171': 2, '201.131.38.208': 2, '201.163.98.162': 2,
    '202.77.121.186': 2, '206.189.118.168': 2, '207.241.227.38': 2, '208.82.115.230': 2, '209.140.29.226':
    2, '216.119.108.161': 2, '216.222.194.94': 2, '216.83.55.154': 2, '222.186.13.91': 2, '23.4.43.27': 2,
    '2u.se': 2, '31.3.88.166': 2, '35.188.36.185': 2, '35.199.84.117': 2, '35.239.229.191': 2,
    '45.238.23.82': 2, '45.40.130.40': 2, '47.74.231.192': 2, '47.90.0.38': 2, '51.255.64.58': 2,
    '51haoyayi.cn': 2, '62.234.180.243': 2, '67.199.68.195': 2, '67.229.29.8': 2, '68.178.252.148': 2,
    '68.66.220.82': 2, '78.143.96.35': 2, '87.138.95.150': 2, '96.lt': 2, '98.126.214.77': 2,
    'aboutkidshealth.ca': 2, 'act-secure.com': 2, 'adviceintl.com': 1, 'afritraders.com': 1, 'aieuv.cn': 2,
    'amazjon.top': 2, 'amazonip.top': 2, 'amberexpeditions.com': 1, 'analyticscluster.com': 1, 'anl.gov': 2,
    'anlanjiehostel.com': 2, 'apacessltaupersonalit.com': 2, 'apps-1and1.net': 2, 'arrowcase.com': 2,
    'arvixeshared.com': 2, 'at.ua': 1, 'aust.com': 2, 'azurewebsites.net': 2, 'babyloweb.eu': 2,
    'babywantsbling.com': 1, 'bdisl.com': 2, 'benefitcentre.com': 2, 'biz.ua': 2, 'bthei.com': 2,
    'byefree.org': 2, 'byethost5.com': 2, 'cargomanager.io': 2, 'carrotstown.com': 1, 'casasbahia.net': 1,
    'cheapcolumbusproperties.com': 1, 'checkbbsms.com': 2, 'client-redirection.com': 1, 'clipartpanda.com':
    2, 'cloud.goog': 2, 'cloudflareaccess.com': 2, 'cloudfront.net': 2, 'co.ua': 2, 'colibri.ga': 1,
    'com-devicealert.com': 2, 'com-sa.pm': 2, 'contacthawk.com': 1, 'cyberoxen.com': 1, 'dataforce.co.uk':
    2, 'dizzystreams.com': 1, 'docdeliveryapp.com': 2, 'donpepe.hu': 1, 'dotartprinting.com': 1,
    'ebi.ac.uk': 2, 'eligulinsaat.com': 2, 'embl-hamburg.de': 2, 'eperidemo.net': 2, 'everytalk.org': 1,
    'everythingwakeboard.com': 1, 'faa.gov': 2, 'farmoyunlari.com': 1, 'freebuddymovies.com': 2,
    'genome.jp': 2, 'getsviip7.com': 2, 'gloomky.com': 1, 'gol.com': 2, 'hbs.edu': 2, 'hitachi-cable.com':
    2, 'holdwiki.com': 1, 'holl-vips.com': 2, 'hsforms.com': 2, 'htmlcomponentservice.com': 2,
    'icondigital.com': 2, 'idocall.com': 1, 'indeedcontract.com': 2, 'innovaproteperuu.com': 2, 'irv.cc': 2,
    'isgroup-international.com': 2, 'jangansedihdandiam.com': 2, 'jcap.com.cn': 2, 'jreg99.com': 2,
    'karlspreis.de': 2, 'latos.co.kr': 2, 'ledger-event.com': 2, 'liraon.com': 1, 'long-term-service.com':
    2, 'lucky76.ml': 2, 'luigitavanti.com': 2, 'marblechamp.com': 1, 'marches-publics.gouv.fr': 2,
    'mazzoli.com': 1, 'mirrorimagecreations.com': 1, 'mondaymagi.com': 1, 'my-free.website': 2,
    'myfreesites.net': 2, 'myshn.net': 2, 'navy.mil': 2, 'nblz.ru': 2, 'nepa3d.com': 1, 'netaugusta.com': 1,
    'newmemorialsdirect.com': 2, 'nichost.ru': 2, 'nijiyamarket.com': 2, 'nouvelle-aquitaine.pro': 2,
    'nuix.com': 2, 'odns.fr': 2, 'ohio-state.edu': 2, 'pacom.mil': 2, 'pagedemo.co': 2,
    'patternsandshades.com': 1, 'phoneidtw.com': 2, 'pnp.co.za': 2, 'pollsbee.com': 1, 'popupbarbados.com':
    1, 'poweredbydivido.com': 2, 'preview-domain.com': 2, 'privado.info': 2, 'prizegiveaway.net': 2,
    'prizesforall.com': 2, 'productitem.com': 1, 'rackcdn.com': 2, 'rakuen.xyz': 2, 'regruhosting.ru': 2,
    'replayattack.us': 2, 'revenueemployementinterac.com': 1, 'sagam.sn': 2, 'satanswer.com': 1, 'sfu.ca':
    2, 'shineritethru.com': 1, 'siemon.com': 2, 'sitebeat.site': 2, 'skyfencenet.com': 2, 'smevmail.com': 2,
    'solvedassignmentaiou.com': 1, 'sos03.com': 1, 'southeasternhotelmanagement.com': 1, 'sslblindado.com':
    2, 'starshipsandqueens.com': 1, 'stockedsummit.com': 1, 'storybookalpacas.com': 1,
    'supersignfactory.com': 1, 'symphonykauharus.com': 2, 'taalim.ma': 2, 'takesaspark.com': 1,
    'temporary.link': 2, 'tempsite.link': 2, 'theclimatemakers.com': 1, 'thedigitallandscaping.com': 1,
    'themariettabilliardclub.com': 1, 'topgaysongs.com': 1, 'typosfera.net': 1, 'uncluster.com': 1,
    'underbaronline.com': 1, 'universe.wf': 2, 'upgrade-emailexchange.ml': 1, 'veromailer.com': 2,
    'viabcp.com': 2, 'waterdamagerestorationhouston.com': 1, 'waterloooptical.com': 1, 'webcountdown.de': 2,
    'webout.life': 2, 'webstarterz.com': 2, 'wifirouter.net': 1, 'wiwi.dev': 2, 'wsodqa.com': 2,
    'zlobek.stargard.pl': 1
}

# Second-level labels the dataset flags as brand domains.
BRANDS = frozenset({
    'amazon', 'apple', 'audi', 'bbc', 'bing', 'blogspot', 'canon', 'cnet', 'colgate', 'dell', 'disney',
    'dropbox', 'ebay', 'express', 'facebook', 'flickr', 'fujitsu', 'gartner', 'github', 'global', 'google',
    'ibm', 'imdb', 'instagram', 'intel', 'kfc', 'linkedin', 'mckinsey', 'microsoft', 'microsoftonline',
    'mtv', 'netflix', 'nintendo', 'nytimes', 'oracle', 'pinterest', 'snapchat', 'stackexchange',
    'stackoverflow', 'theguardian', 'tmall', 'tripadvisor', 'tumblr', 'twitter', 'ups', 'visa', 'walmart',
    'whatsapp', 'wikipedia', 'wordpress', 'yahoo', 'youtube'
})

# Public suffixes the dataset flags as suspicious.
SUSPICIOUS_TLDS = frozenset({
    'ae', 'asia', 'audio', 'bj', 'club', 'exposed', 'ga', 'go.id', 'gob.pe', 'icu', 'la', 'link', 'media',
    'ml', 'mx', 'online', 'party', 'pe', 'review', 'rs', 'sa.gov.au', 'tk', 'top', 'website', 'work', 'xyz'
})
