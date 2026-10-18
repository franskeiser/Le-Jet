CROWDSOURCED SMS - RAW SUBMISSIONS
==================================

This folder is inside the Le-Jet repo (data/crowd/raw) so every contributor
can add submissions. These files still contain personal details, so:
- the repo must stay PRIVATE, and only the research team may be added to it;
- anything committed stays in the git history even after it is deleted, so
  remove personal details you do not need before committing;
- the links in here are real phishing links: never click them.

HOW TO USE
----------
1. Copy _template_submissions.xlsx and rename the copy to
       submissions_YYYY-MM-DD_<your code>.xlsx   (e.g. submissions_2026-11-03_r02.xlsx)
   One file per person per collection day. Do not edit the template itself.
   Never edit SOMEONE ELSE'S file: git cannot merge two people's changes to
   the same Excel file, so one person's work would be lost. Run git pull
   before you start, and commit + push only your own file when done.
   The template has ONE EXAMPLE ROW (the fake GCash message). DELETE that
   row in your copy, or it will end up in the dataset as a real message.
   (Lost the template? Remake it from the repo folder with:
     .venv\Scripts\python.exe scripts\make_crowd_template.py )
2. Add one row per SMS message. Every message must contain a link.
   Columns with fixed answers have a DROPDOWN: click the cell, then the
   small arrow on its right. Hover a header to see what the column means.
   Excel warns if you type something not in the list; fix it unless you
   are sure.
3. Save normally (it stays .xlsx). No need to save as CSV any more.
4. Make the inspector's link list (PowerShell, from the repo folder; give
   everyone's files of that day at once):
     .venv\Scripts\python.exe scripts\crowd_to_urls.py data\crowd\raw\submissions_2026-11-03_r02.xlsx data\crowd\raw\submissions_2026-11-03_r03.xlsx --out D:\LeJetData\input\urls.txt
   then follow docs\inspector_runbook.md from step 4. Do it the same day
   (within 6 hours if possible). The output goes in D:\LeJetData\clean\

COLUMNS (fill all 13; * = dropdown)
-----------------------------------
date_submitted     YYYY-MM-DD HH:MM, Philippine time (UTC+8)     2026-11-03 14:20
sms_text           the full SMS, copied exactly, link included    (see below)
submitter_view *   form: "How would you describe the SMS message?"
                   Suspicious or potentially fraudulent | Appears legitimate
                   or genuine | Not sure
org_type *         form: "What type of sender or organization..." (11 choices,
                   Bank ... Unknown sender, Other)
concern *          form: "What does the message mainly concern?" (18 choices,
                   Account verification ... None of the above, Not sure)
label *            smishing | legitimate (the CHECKED decision; the
                   submitter's own view goes in submitter_view)   smishing
category *         bank | e-wallet | credit_card | loan |         e-wallet
                   bill_payment | delivery_fee | government_aid |
                   investment | other_financial
language *         en | tl | taglish                              en
sender_type *      number | shortcode | alphanumeric | unknown    number
                   (number = 11-digit mobile, alphanumeric = a
                   name like "GCash"; do NOT write the sender itself)
date_received      YYYY-MM-DD, the day the person got the SMS     2026-11-03
label_source *     submitter | bank_confirmed | reviewer          submitter
label_verified_by  code of the SECOND person who checked the      r02
                   label (leave empty until checked)
consent *          1 = person agreed, 0 = did not (0 rows are     1
                   dropped)

RULES
-----
- Financial-related messages only (banks, e-wallets, loans, payments,
  deliveries asking for payment). Collect BOTH smishing and legitimate.
- sms_text may still contain names/numbers here; the builder script
  replaces them with <NAME>, <NUM>, <ACCT>. Leave the link untouched.
- If the SMS has several links, the first one is used.
- Never change a label because the page was dead or looked clean.
- Old .csv files still work with crowd_to_urls.py if they have all 13
  columns. The older 10-column CSVs do not.

Columns that are NOT filled by hand (message_id, link, final_url,
link_domain, template_id, inspection columns, the 37 features) are added
later by src/data/build_crowd_dataset.py and the web inspector.
