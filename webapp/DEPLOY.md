# Le-Jet SMS collection website — how it works and how to put it on Azure

What you end up with: `https://<your-name>.<region>.cloudapp.azure.com`. A groupmate opens it, pastes an SMS
(text with its link), picks **smishing** or **legitimate** and the **language** (English, Tagalog or Taglish) and
presses *Add to dataset*. The site then, by itself:

1. finds the first link in the text (same finder and rule as `scripts/crowd_to_urls.py`),
2. visits it with the lab pipeline (DNS, redirects, page, domain registration),
3. appends **one row** to one shared CSV (50 columns, see below),
4. shows the 37 extracted features, and lists the newest rows for everybody (with a *Download CSV* button).

## Adding from screenshots

Under the paste form there is **Add from screenshots**: pick up to 6 pictures (PNG, JPEG or WebP, 5 MB each; a single
message or a whole conversation per picture) and press *Read screenshots*.

* The server reads the text with **RapidOCR running on the VM itself** (free, no outside service). The pictures are
  processed in memory and **thrown away: they are never stored**. Only the text a person confirms goes into the CSV.
* Each message bubble becomes an **editable card**. The status bar, clock, date separators, "Delivered", the message box,
  the iPhone unknown-sender banner and Google Messages' "Sender can't accept replies" / "Tap to load preview" are dropped.
  Right-aligned bubbles are marked *looks sent by you*. Cards **without a link** are not useful for the dataset, so they
  are hidden behind a checkbox ("Show N text blocks without a link") instead of cluttering the list.
* On each card a person can fix the text, **merge it into the card above** (when a bubble was split), remove it, and choose
  smishing or legitimate and the language. *Add to dataset* saves one card; *Mark all as…*, *Language of all* and
  *Add N ready* do a whole batch (a card is "ready" once it has both a label and a language).
  Saving a card uses exactly the same steps as pasting (link found, inspected, row appended, duplicates refused).
* What the reader repairs, checked on real iPhone, Google Messages and Xiaomi screenshots: words the OCR ran together
  (it measures the blank gaps in the picture and puts the spaces back), links cut across two lines (`bit.ly/` + `ubrewardscard`,
  even when the `/` was read as `l`), and a blank line inside one message (it does not split it). It cannot fix everything:
  all-capital promo text can still get a stray space (`PL ATFORM`), so read each card before adding it.
* **OCR makes mistakes, especially in links** (`l`/`1`, `O`/`0`), and a wrong link gives wrong features. The page tells people
  to check every link letter by letter. Clear, un-cropped screenshots read best.
* If the reader cannot start on the server, the site still works for pasted text and says so (`"screenshots": false` in
  `/api/health`). `setup.sh` prints a warning in that case.
* It needs about 1 GiB of memory while reading, which is why the VM must have 4 GiB (`B2als_v2`).

## The CSV

* One file on the VM: `/var/lib/lejetweb/web_submissions.csv` (a copy is made daily in `/var/lib/lejetweb/backups`).
* **50 columns**: the crowd study's 57 (`build_crowd_dataset.COLUMNS`) minus seven the site does not collect or need:
  `category`, `sender_type`, `date_received`, `label_source`, `consent`, `inspected_at`, `hours_to_inspect`. A test checks
  that the two layouts differ by exactly those seven, with the same names and order for everything else. So this file is
  **not** the builder's 57-column layout any more: add those columns back (or adjust the builder) if you feed it to
  `build_crowd_dataset` or to code that expects them.
* What the website fills: 6 message columns (`message_id` = `web_000001`…, `sms_text` as typed, `label`, `language`,
  `date_submitted` in Philippine time, `label_verified_by`), 5 link columns (`link`, `final_url`, `link_domain`,
  `template_id`, `link_shortened`), 2 inspection columns (`page_fetched`, `fetch_status`) and the 37 features.
* `language` is **required**: `en` (English), `tl` (Tagalog) or `taglish` (a mix in the same message), chosen by the person
  adding the message. The server refuses anything else.
* What stays `-1`: `label_verified_by` (a second person checks the label later). `page_rank` is filled in
  **automatically** (see "page_rank" below) once an Open PageRank key is installed on the VM; without a key it is `-1`.
* Page features are `-1` whenever the page could not be fetched (`fetch_status` = dead / blocked / timeout). That is by
  design, not a fault: 13 columns (`nb_hyperlinks` … `domain_with_copyright`) describe the page itself. The domain columns
  (`whois_registered_domain`, `domain_registration_length`, `domain_age`) and the 20 link-text columns are still filled.
* There is **no consent column and the site no longer asks**: agreement to use the messages has to be settled with the
  group outside the site (the page carries a one-line notice). If you later need `consent=1` for the builder, add the column
  back when you export.
* **Old files are converted automatically.** If the file on the VM still has the previous 57 columns, the site rewrites it
  with the 50 columns when it starts, keeps the rows, and saves the old file next to it as `web_submissions.csv.57cols.bak`.
  Rows added before this change have `language = -1` (fill them in by hand, or delete the test rows).
* A dead or blocked page is **still saved** (`fetch_status` says `dead`/`blocked`/…, page features are `-1`), because
  the label does not depend on the page.
* A message with no link is **refused**; an exact duplicate (same text, same link) is **refused** and not fetched.
* All rows are kept: the builder's own rules (second-person label check, max 20 rows per domain and class, phone-number
  masking) are applied when you build the dataset from this file. The text is stored as typed, so the warning on the
  page asks people to remove personal details first.
* A text that starts with `=`, `+`, `-` or `@` gets a leading `'` so a spreadsheet cannot run it as a formula.

## page_rank (automatic, needs your Open PageRank key)

When a message is added and its page was fetched, the site asks Open PageRank for the rank (0 to 10) of the page's final
registrable domain and writes it into `page_rank`. The rules are the lab's (`src/lejet/page_rank.py`, shared with
`scripts/lookup_page_rank.py`): a domain with no rank is `0` (unrated is a real value), a **failed** look-up is `-1` and is
never turned into `0`, and a page that was **not reached** is not looked up at all (its domain is not sent to the service and
its `page_rank` stays `-1`, like every other page feature). Each domain is asked for only once (answers are cached in
`/var/lib/lejetweb/page_rank_cache.json`: domains and ranks, never the key), and requests are spaced half a second apart.

**Install the key once, on the VM:**

```bash
sudo bash /opt/lejet-web/webapp/deploy/set-rank-key.sh
```

It asks for the key at a hidden prompt (copy the value after `OPENPAGERANK_API_KEY=` from `D:\LeJetData\secrets\.env` on your
PC; do not paste it into a chat or a command line), stores it in `/etc/lejet-web/openpagerank.key` (owned by root, readable
only by root and the website's user), restarts the site and prints its health check. `"page_rank":true` means a key is
installed. To remove it: `sudo rm /etc/lejet-web/openpagerank.key && sudo systemctl restart lejet-web`.

**This is a deliberate exception to the lab's rule** that the key never goes onto a machine that visits hostile pages. What
is done to limit it: the key is read from a file, never from the environment (the page-parsing workers are separate processes
and only inherit the environment); it is never printed, logged, cached or put in the CSV or in an error message; it is sent
only to `openpagerank.com` and to `openpagerank.keywordseverywhere.com` (the address the service redirects to since it moved;
your lab's urllib scripts already follow that redirect silently) over https. The site follows that redirect only to those two
hosts, at most 3 hops; a redirect anywhere else makes the look-up fail (`-1`) without sending the key. What is NOT prevented: the website process itself holds the key in memory and the
service user can read the file, so if the site were ever taken over, the key could be stolen. It is a free-tier key and can
be rotated at openpagerank.com, so: use a key you use nowhere else, and rotate it when the project ends.

**Rows added before the key was installed** have `page_rank = -1`. Fill them on your PC after downloading the CSV (the key stays
there for this): `python scripts/fill_page_rank.py D:\path\lejet_web_submissions.csv` writes `<file>.with_page_rank.csv` and
never changes the original.

## Who can use it

There is **no access code**: anyone who has the address can open the page, add rows, see every message and download
the CSV. That is fine for a small group, but know what it means:

* Share the address only with your groupmates, and give the VM an **unguessable name** (step 3, e.g. `lejet-x7k2q`
  instead of `lejet`). The site tells search engines not to list it (`robots.txt`, `noindex`).
* Anyone who finds the address could add junk rows or make the VM visit links. Rows can be removed by editing the
  CSV (step 5), and the daily backup lets you roll back.
* The CSV holds the pasted messages exactly as typed. If someone pastes a phone number, everyone with the address can
  read it. Keep reminding the group to remove personal details first.
* **To add a code later:** on the VM run `sudo nano /etc/lejet-web.env`, remove the `#` before `LEJET_ACCESS_CODE=`,
  type a word after the `=`, save, then `sudo systemctl restart lejet-web`. The page then shows a code box by itself.

## How it is kept safe

```
browser --HTTPS--> Caddy (port 443) --> the app (127.0.0.1:8000) --> visits the link
```

* The app refuses private, loopback and cloud-metadata addresses before connecting, and **`egress.nft`** is a second,
  network-level barrier: the app's user can never connect to the VM's own services, the Azure metadata service
  (169.254.169.254) or any private network, even if a hostile domain changes its DNS answer. This matters most now:
  the sites it visits are hostile, and without a code the address is the only gate.
* The page shows links defanged (`hxxps://example[.]com`). The CSV holds the **real** links: open it in a spreadsheet
  or Python, never click them.
* Limits: 30 submissions and 12 screenshots per minute per person, 3 links inspected at the same time.
* Treat the VM as **disposable**: no personal files, no passwords, no keys other than your own login key.

## 1. Make the bundle (on your PC)

```powershell
cd D:\Le-Jet
powershell -ExecutionPolicy Bypass -File webapp\deploy\make_bundle.ps1
```

This creates `lejet-web-bundle.tar.gz` (about 100 KB: only the code the site needs; the Python packages, including the
OCR models, are downloaded by `setup.sh` on the VM).

## 2. Create the VM (Azure portal, portal.azure.com)

Create a resource -> **Virtual machine**:

| Setting | Value |
|---|---|
| Subscription | Azure for Students |
| Resource group | create new, e.g. `lejet-rg` |
| Region | one the portal lets you pick (Students accounts are limited to a few regions; choose the nearest, e.g. Southeast Asia) |
| Image | **Ubuntu Server 24.04 LTS** |
| Size | the cheapest with **at least 2 GiB RAM** and no "unavailable" warning (e.g. `B2als_v2` = 2 vCPU / 4 GiB, `B1ms` = 1 vCPU / 2 GiB, `B2s`). **Not** `B2ats_v2` or anything with 1 GiB: too small. |
| Authentication | **SSH public key**, username `azureuser`; let Azure generate the key pair and **download the .pem file** |
| Inbound ports | SSH (22), HTTP (80), HTTPS (443) |

Create it. When it is ready, open the VM -> **Networking** -> the SSH (22) rule -> set *Source* to *IP addresses* and
type your own public IP, so nobody else can even try to log in.

## 3. Give it a name

VM -> **Overview** -> click the **Public IP address** -> **Configuration** -> *DNS name label*: choose something
nobody would guess, e.g. `lejet-x7k2q` -> Save. Your host name is now `lejet-x7k2q.<region>.cloudapp.azure.com`
(the portal shows it in full). Use that name wherever the steps below say `lejet-demo`.

## 4. Upload and install (PowerShell on your PC)

```powershell
scp -i C:\path\to\lejet-key.pem D:\Le-Jet\lejet-web-bundle.tar.gz azureuser@lejet-demo.<region>.cloudapp.azure.com:~
ssh -i C:\path\to\lejet-key.pem azureuser@lejet-demo.<region>.cloudapp.azure.com
```

Now on the VM:

```bash
mkdir bundle && tar -xzf lejet-web-bundle.tar.gz -C bundle
sudo bash bundle/webapp/deploy/setup.sh lejet-demo.<region>.cloudapp.azure.com
```

It takes about 5 to 10 minutes (the screenshot reader is a large download). Near the end you should see
`ok: the site can read screenshots`. Check it works:

```bash
curl -s http://127.0.0.1:8000/api/health      # {"ok":true,"rows":0,"code_required":false,"screenshots":true,"rdap_tlds_known":1202}
```

Then open `https://lejet-demo.<region>.cloudapp.azure.com` in your browser and send the address to your groupmates.

## 5. Day-to-day

* **Get the data:** press *Download CSV* on the page. **Do this regularly**: the only copy lives on the VM, and deleting
  the VM or the resource group deletes it. The daily backup protects against mistakes, not against losing the VM.
* **Update the code:** repeat steps 1 and 4 — safe to re-run; the CSV and `/etc/lejet-web.env` are kept.
* **Fix a wrong row** (e.g. a wrong label): stop the app (`sudo systemctl stop lejet-web`), edit
  `/var/lib/lejetweb/web_submissions.csv`, start it again. Keep the header line and the 50 columns.
* **Logs:** `sudo journalctl -u lejet-web -n 50`
* **Save your credit:** when nobody needs it, **Stop** the VM in the portal (status must say *Deallocated*; a VM that
  is merely shut down from inside still costs money). Starting it again brings the site back by itself.
  Also create a **Budget alert** under *Cost Management*.
* **Delete everything at the end of the thesis:** download the CSV first, then delete the resource group `lejet-rg`.

## Things to know

* Domain age/expiry come from RDAP; some country domains (e.g. `.ph`) have no RDAP, so they show `-1`.
* Phishing pages often die within hours; this site inspects the link **at submission**, so `hours_to_inspect` is about 0.
* The site does **not** run the model. Adding a prediction later means loading `src/serving/film_model.py` into the app.
* Tests: `python -m pytest tests/test_webapp_dataset.py tests/test_webapp_screens.py` (project venv; the last test of the second
  file draws a screenshot and runs the real OCR, and is skipped without RapidOCR) and `tests/test_webapp_app.py` (needs
  `pip install -r webapp/requirements-web.txt` on Python 3.12; skipped without FastAPI).
