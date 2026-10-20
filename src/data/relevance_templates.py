r"""Synthetic SMS wording for build_relevant_dataset.py: the text is written AROUND a link, from what the link says about itself.

Every function returns the full SMS, link included (`{u}` is replaced by the link as it will be stored). Nothing here picks a link;
the builder picks the link first and passes in what it is (a brand, a theme, a readable title or site name), so the words and the
link cannot contradict each other.

Families, by the cell of the design they serve:
    smish(...)            a scam in the words of a brand or a theme, for a link that names that brand or theme          (S1)
    neutral(...)          wording that fits any link and says nothing about risk; used for BOTH classes                (S2, L1)
    scam_benign(...)      scam wording whose link is a real, clean site (a Facebook page, a Google form)               (S3)
    legit_brand(...)      a genuine notice from a brand, for that brand's own link                                     (L1)
    legit_security(...)   a genuine notice that SOUNDS alarming (never share your OTP; new sign-in detected)           (L2)
    legit_share(...)      a person sharing a page, built from the page's own title or site name                        (L1)
    legit_theme(...)      a notice about a theme (a job opening, a sale, a permit) for a link on that theme            (L1)

English and Taglish variants are written by hand, not translated by code. `rng` is a numpy Generator so a seed reproduces a build.
"""
from __future__ import annotations

import re

import numpy as np

from src.data.relevance_lexicon import brand_name

TAGLISH_RATE = 0.35

_PREFIX_SMISH = ["", "", "", "", "", "URGENT: ", "NOTICE: ", "ALERT: ", "FINAL NOTICE: "]
_SUFFIX_SMISH_EN = ["", "", "", " Act now.", " Link expires in {hrs} hours.", " Ignore this message and your access will be closed.",
                    " Failure to respond will result in permanent suspension.", " Do not share this with anyone."]
_SUFFIX_SMISH_TL = ["", "", " Agad na kumilos.", " Mag-e-expire ang link sa {hrs} oras.", " Kapag hindi ka tumugon, ma-su-suspend ang account mo.",
                    " Huwag ibahagi ito kanino man."]
_PREFIX_LEGIT = ["", "", "", "Hi! ", "Hello! ", "Good day, ", "Reminder: "]
_SUFFIX_LEGIT_EN = ["", "", "", " Thank you.", " Thanks for choosing us.", " Have a great day!", " No action is needed if you have already done this."]
_SUFFIX_LEGIT_TL = ["", "", " Salamat po.", " Maraming salamat!", " Ingat po palagi."]

# ----------------------------------------------------------------------------------------------------------- smishing
# theme -> (English with brand, Taglish with brand, English without brand, Taglish without brand)
_SMISH = {
    "account": (
        ["{b}: Your account has been temporarily locked due to unusual sign-in activity. Verify your identity now: {u}",
         "{b} security alert: we detected a login from a new device. If this wasn't you, secure your account at {u}",
         "{b}: Your account will be suspended in {hrs} hours unless you confirm your details here: {u}",
         "{b} Support: your password has expired. Reset it today to avoid losing access: {u}",
         "We noticed suspicious activity on your {b} account. Review and restore access: {u}",
         "{b}: Action required. Confirm your account information to keep your account active: {u}"],
        ["{b}: Na-detect namin ang kakaibang activity sa account mo. I-verify agad para hindi ma-lock: {u}",
         "{b} alert: may nag-login sa ibang device. Kung hindi ikaw 'yan, i-secure ang account mo dito: {u}",
         "{b}: Ma-su-suspend ang account mo sa loob ng {hrs} oras kung hindi mo ito ma-confirm: {u}",
         "Paalala mula sa {b}: expired na ang password mo. I-reset na para hindi mawala ang access: {u}"],
        ["Your account has been locked due to unusual activity. Verify your identity to restore access: {u}",
         "Security alert: sign-in attempt blocked. Confirm it was you or secure your account here: {u}",
         "Your account will be closed in {hrs} hours unless you update your details: {u}",
         "Action required: confirm your login information to avoid suspension: {u}",
         "Your password expires today. Keep your access by updating it here: {u}"],
        ["Na-lock ang account mo dahil sa kakaibang activity. I-verify agad para ma-restore: {u}",
         "Security alert: may blocked na login attempt. I-confirm kung ikaw ito dito: {u}",
         "Ma-su-suspend ang account mo sa loob ng {hrs} oras. I-update ang details dito: {u}"]),
    "payment": (
        ["{b}: A payment of P{amt} is pending on your account. If you did not authorize this, cancel it here: {u}",
         "{b}: Your payment could not be processed. Update your payment method to avoid service interruption: {u}",
         "{b} refund notice: P{amt} is waiting for you. Claim it before it expires: {u}",
         "{b}: We could not verify your recent transaction of P{amt}. Confirm it now: {u}",
         "{b}: Your {b} wallet has been limited. Restore full access by verifying your details: {u}"],
        ["{b}: May pending na payment na P{amt} sa account mo. Kung hindi ikaw ito, i-cancel dito: {u}",
         "{b}: Hindi na-process ang payment mo. I-update ang payment method para hindi maputol ang serbisyo: {u}",
         "{b} refund: may P{amt} na naghihintay sa iyo. I-claim bago mag-expire: {u}"],
        ["Your payment of P{amt} is pending. If you did not make this payment, cancel it here: {u}",
         "Payment failed. Update your billing details now to keep your service active: {u}",
         "You have an unclaimed refund of P{amt}. Claim it within {hrs} hours: {u}",
         "Unusual transaction of P{amt} detected on your card. Review it immediately: {u}"],
        ["May pending na payment na P{amt}. Kung hindi ikaw ito, i-cancel dito: {u}",
         "Failed ang payment mo. I-update ang billing details para hindi maputol ang serbisyo: {u}",
         "May unclaimed refund ka na P{amt}. I-claim sa loob ng {hrs} oras: {u}"]),
    "delivery": (
        ["{b}: Your parcel is on hold due to an unpaid delivery fee of P{amt}. Pay here to reschedule: {u}",
         "{b}: We attempted to deliver your package but the address was incomplete. Update it here: {u}",
         "{b} notice: your shipment {code} is held at customs. Settle the fee to release it: {u}",
         "{b}: Delivery failed. Confirm your address within {hrs} hours or the parcel will be returned: {u}"],
        ["{b}: Naka-hold ang parcel mo dahil sa unpaid delivery fee na P{amt}. Bayaran dito para ma-reschedule: {u}",
         "{b}: Hindi na-deliver ang package mo dahil kulang ang address. I-update dito: {u}",
         "{b}: Nasa customs ang shipment {code}. Bayaran ang fee para ma-release: {u}"],
        ["Your parcel is on hold: unpaid delivery fee of P{amt}. Pay now to reschedule delivery: {u}",
         "Delivery attempt failed. Confirm your address and pay the redelivery fee: {u}",
         "Package {code} could not be delivered. Track and reschedule here: {u}",
         "Your shipment is held at customs. A fee of P{amt} is required to release it: {u}"],
        ["Naka-hold ang parcel mo: may unpaid delivery fee na P{amt}. Bayaran para ma-deliver: {u}",
         "Hindi na-deliver ang package mo. I-confirm ang address at bayaran ang redelivery fee: {u}",
         "Hindi ma-deliver ang package {code}. I-track at i-reschedule dito: {u}"]),
    "prize": (
        ["Congratulations! You've been selected for the {b} anniversary giveaway. Claim your reward: {u}",
         "{b}: You are our lucky winner of P{amt}! Claim it within {hrs} hours: {u}",
         "{b} loyalty reward: you won a free gift. Confirm your delivery details here: {u}"],
        ["Congrats! Napili ka sa {b} anniversary giveaway. I-claim ang reward mo dito: {u}",
         "{b}: Ikaw ang lucky winner ng P{amt}! I-claim sa loob ng {hrs} oras: {u}"],
        ["Congratulations! You have been selected to receive P{amt}. Claim your prize now: {u}",
         "You are today's lucky winner! Complete this short survey to claim your reward: {u}",
         "Your number was chosen for a free gift. Claim before it expires: {u}"],
        ["Congrats! Ikaw ang napili para makatanggap ng P{amt}. I-claim ang premyo mo dito: {u}",
         "Ikaw ang lucky winner ngayong araw! Sagutan ang survey para makuha ang reward: {u}"]),
    "crypto": (
        ["{b}: A withdrawal of {amt} USDT is pending. Confirm now or it will be cancelled: {u}",
         "{b} security: unusual login detected. Verify your wallet to keep your funds safe: {u}",
         "{b}: Your account is restricted. Complete verification to unlock withdrawals: {u}"],
        ["{b}: May pending withdrawal na {amt} USDT. I-confirm agad o made-cancel ito: {u}",
         "{b}: May kakaibang login sa account mo. I-verify ang wallet mo para ligtas ang pondo: {u}"],
        ["Your wallet has a pending withdrawal of {amt} USDT. Confirm it now or it will be cancelled: {u}",
         "Claim your free airdrop of {amt} tokens before the snapshot ends: {u}",
         "Verify your wallet to unlock your funds: {u}"],
        ["May pending withdrawal ang wallet mo na {amt} USDT. I-confirm agad: {u}",
         "I-claim ang libreng airdrop na {amt} tokens bago matapos ang snapshot: {u}"]),
    "gambling": (
        ["{b}: Register now and claim your {pct}% welcome bonus! Play slots and live casino: {u}",
         "{b} is open 24/7! Sign up today and get free credits plus a {pct}% deposit bonus: {u}"],
        ["{b}: Magparehistro na at kunin ang {pct}% welcome bonus! Maglaro ng slots at live casino: {u}",
         "Bukas na ang {b} 24/7! Mag-sign up at kumuha ng libreng credits at {pct}% deposit bonus: {u}"],
        ["Register now and claim your {pct}% welcome bonus! Play slots and live casino: {u}",
         "JACKPOT! Sign up today and get free credits plus a {pct}% daily cashback: {u}",
         "Win up to P{amt}! New players get a free bet. Join now: {u}"],
        ["Magparehistro na at kunin ang {pct}% welcome bonus! Maglaro ng slots at live casino: {u}",
         "Mag-sign up ngayon at kumuha ng libreng credits at {pct}% daily cashback: {u}",
         "Manalo ng hanggang P{amt}! May libreng bet ang mga bagong player. Sumali na: {u}"]),
    "job": (
        ["{b}: Work from home and earn P{n} per day. No experience needed. Apply here: {u}",
         "{b} is hiring online assistants. Daily pay P{n}. Register now: {u}"],
        ["{b}: Work from home at kumita ng P{n} kada araw. Walang experience na kailangan. Mag-apply dito: {u}"],
        ["Hiring! Earn P{n} daily doing simple online tasks. Apply now: {u}",
         "Part-time job offer: work from home, daily salary P{n}. Register here: {u}",
         "Limited slots! Be an online assistant and earn P{n} a day: {u}"],
        ["Hiring! Kumita ng P{n} kada araw sa simpleng online tasks. Mag-apply na: {u}",
         "Part-time job offer: work from home, sahod P{n} kada araw. Mag-register dito: {u}"]),
    "telco": (
        ["{b}: You have {gb}GB of free data waiting. Claim it before it expires: {u}",
         "{b}: Your SIM will be deactivated unless you update your registration today: {u}",
         "{b}: Congratulations! You won a P{amt} load. Claim here: {u}"],
        ["{b}: May {gb}GB na libreng data ka. I-claim bago mag-expire: {u}",
         "{b}: Made-deactivate ang SIM mo kung hindi mo ma-update ang registration ngayon: {u}"],
        ["Your SIM registration is incomplete and will be deactivated today. Update now: {u}",
         "Claim your free {gb}GB data promo before midnight: {u}"],
        ["Hindi pa kumpleto ang SIM registration mo at made-deactivate ngayon. I-update na: {u}",
         "I-claim ang libreng {gb}GB data promo bago maghatinggabi: {u}"]),
    "gov": (
        ["{b}: Your benefits are on hold. Update your records to continue receiving them: {u}",
         "{b} notice: unpaid contribution detected. Settle it now to avoid penalties: {u}"],
        ["{b}: Naka-hold ang benefits mo. I-update ang records para magpatuloy ang pagtanggap: {u}",
         "{b}: May unpaid contribution ka. Bayaran agad para walang penalty: {u}"],
        ["Your government assistance is ready for release. Verify your details to claim: {u}",
         "Unpaid fees detected on your record. Settle today to avoid penalties: {u}"],
        ["Ready na ang ayuda mo para i-release. I-verify ang details para ma-claim: {u}",
         "May unpaid fees sa record mo. Bayaran ngayon para walang penalty: {u}"]),
    "social": (
        ["{b}: Your page violates our community policy. Appeal within {hrs} hours or it will be removed: {u}",
         "{b}: Someone tried to log in to your profile. Secure your account here: {u}",
         "{b} Support: your account has been reported. Verify your identity to avoid a ban: {u}"],
        ["{b}: Lumalabag ang page mo sa community policy. Mag-appeal sa loob ng {hrs} oras o made-delete ito: {u}",
         "{b}: May nag-try mag-login sa profile mo. I-secure ang account mo dito: {u}"],
        ["Your profile was reported. Verify your identity within {hrs} hours to avoid a ban: {u}"],
        ["Na-report ang profile mo. I-verify ang identity mo sa loob ng {hrs} oras para hindi ma-ban: {u}"]),
    "subscription": (
        ["{b}: Your payment failed. Update your billing details to keep your membership: {u}",
         "{b}: Your subscription will be cancelled today. Renew now to keep watching: {u}"],
        ["{b}: Failed ang payment mo. I-update ang billing details para hindi ma-cancel ang membership: {u}"],
        ["Your membership payment failed. Update your billing details to keep your plan: {u}"],
        ["Failed ang payment ng membership mo. I-update ang billing details: {u}"]),
    "shopping": (
        ["{b}: Your order {code} was cancelled due to a payment error. Re-confirm your payment: {u}",
         "{b}: You qualified for a P{amt} voucher. Claim it before it expires: {u}"],
        ["{b}: Na-cancel ang order {code} dahil sa payment error. I-confirm muli ang payment: {u}",
         "{b}: Qualified ka sa P{amt} voucher. I-claim bago mag-expire: {u}"],
        ["Your order was cancelled due to a payment error. Re-confirm your payment: {u}",
         "You qualified for a P{amt} voucher. Claim it before it expires: {u}"],
        ["Na-cancel ang order mo dahil sa payment error. I-confirm muli ang payment: {u}",
         "Qualified ka sa P{amt} voucher. I-claim bago mag-expire: {u}"]),
}

# ---------------------------------------------------------------------------------------------------------- neutral
_NEUTRAL = ["Hi, please check this: {u}", "Here's the link you asked for: {u}", "Please see the details here: {u}",
            "Your document is ready: {u}", "Follow-up on your request: {u}", "Hello! Kindly review this: {u}",
            "More information is available here: {u}", "FYI, details are in this link: {u}", "Please confirm receipt: {u}",
            "Your request has been updated. See: {u}", "Hi, this is about your inquiry: {u}", "Check this out: {u}",
            "Sending you the information: {u}", "Please review before end of day: {u}"]
_NEUTRAL_TL = ["Pakitingnan po ito: {u}", "Eto na yung link: {u}", "Paki-check po ng details dito: {u}", "Para sa inquiry mo, tingnan dito: {u}",
               "Tingnan mo 'to: {u}", "Ito po yung hinihingi ninyo: {u}", "Paki-review po bago mag-end of day: {u}"]

# --------------------------------------------------------------------------------- scam text, clean link (by link brand)
_SCAM_BENIGN = {
    "facebook": ["Earn P{n} daily doing simple online tasks! Message us on Facebook to start: {u}",
                 "Free {gift} giveaway! Like and share our page to claim your prize: {u}",
                 "Hiring online assistants, no experience needed. Send your resume on our page: {u}",
                 "Limited slots! Join our members' group and start earning today: {u}"],
    "instagram": ["Win a free {gift}! Follow our Instagram and send us a message to claim your prize: {u}",
                  "Earn P{n} a day reviewing products. Follow our page to apply: {u}"],
    "whatsapp": ["Congratulations! You won P{n}. Chat with our claims officer on WhatsApp to receive it: {u}",
                 "Work from home, earn P{n} daily. Message us on WhatsApp to get started: {u}"],
    "telegram": ["Join our VIP group and earn P{n} daily. Limited slots: {u}",
                 "Exclusive investment tips, double your money in {hrs} hours. Join now: {u}"],
    "linkedin": ["You're shortlisted! Pay the P{amt} processing fee to confirm your job offer. Details here: {u}",
                 "Remote job offer, P{n} per day. Apply through this profile and send your ID: {u}"],
    "zoom": ["Online orientation for your approved loan. Join now and bring P{amt} for the processing fee: {u}",
             "Your scholarship interview is today. Join the meeting and prepare a P{amt} reservation fee: {u}"],
    "google": ["Congratulations! You were selected for a P{n} cash reward. Fill out this form to claim: {u}",
               "Part-time job opening, P{n} daily. Submit your details using this form: {u}"],
    "microsoft": ["A file was shared with you: 'Payroll update'. Sign in to view and confirm your details: {u}"],
}
_SCAM_BENIGN_TL = {
    "facebook": ["Kumita ng P{n} kada araw sa simpleng online tasks! Mag-message sa amin sa Facebook: {u}",
                 "Free {gift} giveaway! I-like at i-share ang page namin para makuha ang premyo: {u}"],
    "whatsapp": ["Congrats! Nanalo ka ng P{n}. Mag-chat sa claims officer namin sa WhatsApp para makuha ito: {u}"],
    "telegram": ["Sumali sa VIP group namin at kumita ng P{n} kada araw. Limited slots: {u}"],
    "google": ["Congrats! Napili ka para sa P{n} cash reward. Sagutan ang form na ito para makuha: {u}"],
}
_GIFTS = ["iPhone 14", "Samsung phone", "laptop", "P5,000 gift card", "e-bike", "air fryer", "smart TV"]

# ----------------------------------------------------------------------------------------------------- genuine notices
# brand theme -> English, Taglish ({b} brand, {u} link)
_LEGIT_BRAND = {
    "telco": (["{b}: Never run out of data. Check your balance and promos at {u}",
               "{b}: Your promo will expire on {date}. Extend it now: {u}",
               "{b}: Get more value with our latest promos. Browse them here: {u}",
               "{b}: Manage your account and subscriptions with just a tap: {u}",
               "{b}: Your load balance is P{amt}. Top up anytime at {u}"],
              ["{b}: Huwag maubusan ng data. I-check ang balance at promos dito: {u}",
               "{b}: Mag-e-expire ang promo mo sa {date}. I-extend na dito: {u}",
               "{b}: I-manage ang account at subscriptions mo sa isang tap lang: {u}"]),
    "payment": (["{b}: You received P{amt}. View your transaction details in the app or at {u}",
                 "{b}: Your payment of P{amt} was successful. Your receipt is available at {u}",
                 "{b}: Your monthly statement is ready. You can view it at {u}",
                 "{b}: Cash-in of P{amt} is complete. Check your balance at {u}"],
                ["{b}: Nakatanggap ka ng P{amt}. Tingnan ang transaction details sa app o dito: {u}",
                 "{b}: Successful ang payment mo na P{amt}. Nandito ang resibo mo: {u}"]),
    "bank": (["{b}: Your e-statement for this month is ready. View it at {u}",
              "{b}: A deposit of P{amt} was credited to your account. Details at {u}",
              "{b}: Your card ending {code} was used for P{amt}. View the transaction at {u}",
              "{b}: Your loan payment of P{amt} was received. Thank you. Details: {u}"],
             ["{b}: Handa na ang e-statement mo para sa buwang ito. Tingnan dito: {u}",
              "{b}: May P{amt} na na-credit sa account mo. Detalye dito: {u}"]),
    "shopping": (["{b}: Your order {code} has shipped. Track it here: {u}",
                  "{b}: Flash sale starts tonight! Shop your favorites at {u}",
                  "{b}: Your voucher is ready to use. Check your account at {u}",
                  "{b}: Thanks for your order {code}! See your order details at {u}"],
                 ["{b}: Naipadala na ang order {code}. I-track dito: {u}",
                  "{b}: Flash sale ngayong gabi! Mag-shop na sa {u}"]),
    "subscription": (["{b}: Your subscription renews on {date}. Manage your plan at {u}",
                      "{b}: Thanks for subscribing! See what's new this week: {u}",
                      "{b}: Your payment of P{amt} was received. View your billing at {u}"],
                     ["{b}: Magre-renew ang subscription mo sa {date}. I-manage ang plan dito: {u}"]),
    "account": (["{b}: Your account settings were updated. Review them anytime at {u}",
                 "{b}: Your monthly summary is ready. See it at {u}",
                 "{b}: Welcome! Finish setting up your account at {u}"],
                ["{b}: Na-update ang account settings mo. I-review anumang oras dito: {u}"]),
    "social": (["{b}: You have new notifications. See them here: {u}",
                "Join our {b} community for tips and updates: {u}",
                "{b}: Your friend invited you to an event. See the details at {u}"],
               ["{b}: May bago kang notifications. Tingnan dito: {u}",
                "Sumali sa {b} community namin para sa tips at updates: {u}"]),
    "delivery": (["{b}: Your parcel {code} is out for delivery today. Track it at {u}",
                  "{b}: Your package has been delivered. View the proof of delivery at {u}",
                  "{b}: Schedule your delivery or pickup at {u}"],
                 ["{b}: Out for delivery na ang parcel {code}. I-track dito: {u}",
                  "{b}: Na-deliver na ang package mo. Tingnan ang proof of delivery dito: {u}"]),
    "gov": (["{b}: Reminder: your monthly contribution is due on {date}. Check your records at {u}",
             "{b}: Requirements and schedule for your application are posted at {u}",
             "{b}: Advisory for members: online services are available at {u}"],
            ["{b}: Paalala: due na ang buwanang contribution mo sa {date}. I-check ang records dito: {u}",
             "{b}: Nakalagay na ang requirements at schedule ng application mo sa {u}"]),
    "utility": (["{b}: Your bill of P{amt} is due on {date}. Pay online at {u}",
                 "{b}: Scheduled maintenance in your area on {date}. Details at {u}"],
                ["{b}: Due na ang bill mo na P{amt} sa {date}. Magbayad online sa {u}"]),
    "crypto": (["{b}: Your trade of {amt} was completed. View your history at {u}",
                "{b}: Your monthly account summary is ready at {u}"],
               ["{b}: Tapos na ang trade mo na {amt}. Tingnan ang history dito: {u}"]),
    "job": (["{b}: New jobs matching your profile are available. See them at {u}",
             "{b}: Your application was received. Track its status at {u}"],
            ["{b}: May bagong trabahong tugma sa profile mo. Tingnan dito: {u}"]),
    "gaming": (["{b}: Your purchase was successful. View your receipt at {u}",
                "{b}: New season starts soon. See what's coming: {u}"],
               ["{b}: Successful ang purchase mo. Tingnan ang resibo dito: {u}"]),
}

# genuine notices that sound alarming (L2): brand-wide, so they fit any brand's own link
_LEGIT_SECURITY = (
    ["{b}: We will never ask for your OTP, PIN or password. If you get a suspicious message, report it at {u}",
     "{b} reminds you: never click links from unknown senders. Learn how to stay safe: {u}",
     "{b}: A new sign-in to your account was detected. If this was you, no action is needed. Review it at {u}",
     "{b}: Beware of scam texts pretending to be us. Our only official site is {u}",
     "{b}: For your security, never share your one-time PIN with anyone. More tips at {u}",
     "{b} security notice: your password was changed. If you did not do this, contact us through {u}"],
    ["{b}: Hindi namin hihingin ang OTP, PIN o password mo. Kung may kahina-hinalang text, i-report dito: {u}",
     "{b}: Mag-ingat sa scam texts na nagpapanggap na kami. Ito lang ang official site namin: {u}",
     "{b}: Para sa seguridad mo, huwag ibahagi kanino man ang OTP mo. Karagdagang tips: {u}"],
)

# genuine, brand-free notices about a theme (a link on that theme, no brand in it)
_LEGIT_THEME = {
    "job": (["We are hiring! See the open positions and apply at {u}", "Job fair this {day}. Register and view the list of employers at {u}",
             "Your application was received. We will contact you soon. Status: {u}"],
            ["Hiring kami! Tingnan ang mga bakante at mag-apply dito: {u}", "May job fair ngayong {day}. Mag-register dito: {u}"]),
    "shopping": (["Sale ends tonight! Shop now at {u}", "Your cart is waiting. Complete your order at {u}", "New arrivals just dropped. Browse them at {u}"],
                 ["Matatapos ang sale ngayong gabi! Mag-shop na sa {u}", "Naghihintay ang cart mo. Tapusin ang order dito: {u}"]),
    "gov": (["Reminder: requirements and schedules are posted at {u}", "Advisory: online services for residents are available at {u}",
             "Applications for the program are open. Details at {u}"],
            ["Paalala: nakalagay ang requirements at schedule sa {u}", "Bukas na ang application para sa programa. Detalye dito: {u}"]),
    "telco": (["Browse the latest prepaid and broadband plans at {u}", "Your plan details and usage are available at {u}"],
              ["Tingnan ang latest prepaid at broadband plans dito: {u}"]),
    "payment": (["Your invoice is ready. You may view or download it at {u}", "Payment received. Thank you! See your receipt at {u}"],
                ["Handa na ang invoice mo. Tingnan o i-download dito: {u}"]),
    "social": (["Join us this {day}! Event details and RSVP at {u}", "You are invited to our online meetup. Details: {u}"],
               ["Sumali sa amin ngayong {day}! Detalye ng event at RSVP dito: {u}"]),
    "delivery": (["Your parcel is on its way. Track it at {u}", "Delivery update: your order will arrive today. See {u}"],
                 ["Papunta na ang parcel mo. I-track dito: {u}"]),
    "account": (["Your statement is ready. View it at {u}", "Account update: your settings were saved. Review them at {u}"],
                ["Handa na ang statement mo. Tingnan dito: {u}"]),
    "crypto": (["Market report for this week is available at {u}"], ["Available na ang market report ngayong linggo dito: {u}"]),
    "prize": (["Thanks for joining our raffle! Results will be posted at {u}", "Complete our short customer survey here: {u}"],
              ["Salamat sa pagsali sa raffle! Ipo-post ang results sa {u}"]),
    "gambling": (["Responsible gaming tips and support are available at {u}"], ["Mga tips sa responsible gaming dito: {u}"]),
}

# genuine notices in an urgent voice (L2): a real bill, a real security alert, a real expiry, for a brand's own link
_LEGIT_URGENT = (
    ["{b}: URGENT: Your bill of P{amt} is overdue. Pay by {date} to avoid service interruption: {u}",
     "{b}: Security alert! A new device just signed in to your account. Review it now: {u}",
     "{b}: Final reminder: your promo expires tonight! Renew now: {u}",
     "{b}: Your service will be disconnected on {date} unless payment is received. Pay here: {u}",
     "{b}: Important: your password was reset. If this wasn't you, secure your account immediately: {u}",
     "{b}: Limited time! Use your reward points before they expire on {date}: {u}",
     "{b}: Your card ending {code} was used abroad. If this was not you, report it now: {u}"],
    ["{b}: URGENT: Overdue na ang bill mo na P{amt}. Magbayad bago ang {date} para hindi maputol ang serbisyo: {u}",
     "{b}: Security alert! May bagong device na nag-sign in sa account mo. I-review agad: {u}",
     "{b}: Huling paalala: mag-e-expire ang promo mo ngayong gabi! I-renew na: {u}"],
)

# smishing that carries no link: a number to call or text, a chat to join (the number is masked as <NUM>, as in the crowd schema)
_SMISH_NOLINK = (
    ["Congratulations! You won P{amt}. Text your full name and address to <NUM> to claim your prize.",
     "Your parcel {code} is on hold. Call <NUM> now to settle the delivery fee of P{amt}.",
     "URGENT: Your account has been flagged. Call our verification hotline <NUM> immediately.",
     "Hiring! Earn P{n} daily from home. Message <NUM> on WhatsApp to apply.",
     "You have an unclaimed ayuda of P{amt}. Text your name, age and address to <NUM>.",
     "Loan approved! P{amt} is ready for release. Text YES to <NUM> to claim it today.",
     "Hi, is this you? I think I have the wrong number. Add me on Viber <NUM>",
     "Your SIM registration is incomplete and will be deactivated today. Call <NUM> to update.",
     "Dear customer, a suspicious transaction of P{amt} was blocked. Call <NUM> if this was not you.",
     "Limited slots! Earn P{n} a day doing simple tasks. Text JOIN to <NUM>.",
     "Your ATM card has been blocked. To reactivate, call our customer care at <NUM>.",
     "FREE P{amt} load! Text your name and network to <NUM> to receive it before midnight.",
     "You are pre-approved for a P{amt} cash loan, no collateral. Text APPLY to <NUM>.",
     "Sir/Madam, your e-wallet was used for a P{amt} payment. If this is not you, call <NUM> right away.",
     "Congratulations! Your number won the P{amt} raffle. Contact claims officer {name} at <NUM>.",
     "Good day! We have a job opening for online encoders, P{n}/day. Send your resume to <NUM> on Telegram.",
     "Your package is waiting at the courier. Reply with your address or call <NUM> to schedule delivery.",
     "BIR notice: unpaid tax of P{amt} detected under your TIN. Call <NUM> to settle and avoid penalties.",
     "Hello, I saw your profile. Are you free to chat? Message me on WhatsApp <NUM>",
     "Your SSS pension for this month is on hold. Call <NUM> and have your ID ready.",
     "Investment opportunity: turn P{amt} into P{n} in {hrs} hours. Text INVEST to <NUM>.",
     "Your Netflix account will be suspended. Call billing at <NUM> to update your payment."],
    ["Congrats! Nanalo ka ng P{amt}. I-text ang buong pangalan at address mo sa <NUM> para makuha ang premyo.",
     "Naka-hold ang parcel {code}. Tawagan ang <NUM> ngayon para bayaran ang delivery fee na P{amt}.",
     "May unclaimed ayuda ka na P{amt}. I-text ang pangalan, edad at address mo sa <NUM>.",
     "Hiring! Kumita ng P{n} kada araw sa bahay lang. Mag-message sa <NUM> sa WhatsApp para mag-apply.",
     "Approved na ang loan mo! P{amt} ready na. I-text ang YES sa <NUM> para makuha ngayon.",
     "Hi, ikaw ba ito? Mukhang wrong number ako. Add mo ako sa Viber <NUM>",
     "Na-block ang ATM card mo. Para ma-reactivate, tawagan ang customer care sa <NUM>.",
     "LIBRENG P{amt} load! I-text ang pangalan at network mo sa <NUM> bago maghatinggabi.",
     "Pre-approved ka sa P{amt} cash loan, walang collateral. I-text ang APPLY sa <NUM>.",
     "Sir/Madam, nagamit ang e-wallet mo sa P{amt} na payment. Kung hindi ikaw ito, tawagan agad ang <NUM>.",
     "Hello, nakita ko ang profile mo. Free ka ba mag-chat? Mag-message ka sa WhatsApp <NUM>",
     "Naka-hold ang SSS pension mo ngayong buwan. Tawagan ang <NUM> at ihanda ang ID mo."],
)

# genuine but high-pressure wording on a clean page that fits it: a real sale, a real deadline, a real vacancy (L2, brand-free)
_LEGIT_PRESSURE = {
    "shopping": (["FLASH SALE! Up to {pct}% off today only. Don't miss out, shop now: {u}", "LAST CHANCE! Our sale ends tonight at midnight: {u}",
                  "Hurry! Only a few items left in your cart. Check out before they sell out: {u}", "URGENT: Your voucher expires today. Use it now at {u}"],
                 ["FLASH SALE! Hanggang {pct}% off ngayong araw lang. Mag-shop na: {u}", "HULING PAGKAKATAON! Matatapos ang sale ngayong hatinggabi: {u}"]),
    "job": (["Hiring now! Limited slots, apply before the deadline: {u}", "LAST DAY to submit your application. Apply today: {u}",
             "Urgent hiring! Immediate openings, walk-ins welcome. Details and application: {u}"],
            ["Hiring ngayon! Limited slots, mag-apply bago ang deadline: {u}", "HULING ARAW ng application. Mag-apply na: {u}"]),
    "gov": (["URGENT: Deadline for renewal is tomorrow. Submit your requirements online: {u}", "FINAL REMINDER: file your return before {date}. Online filing: {u}",
             "Reminder: unpaid fees may incur penalties after {date}. Pay or check your status at {u}"],
            ["PAALALA: Bukas na ang deadline ng renewal. I-submit ang requirements online: {u}", "HULING PAALALA: mag-file bago ang {date}. Online filing: {u}"]),
    "telco": (["Your plan is about to expire! Renew today to keep your service: {u}", "Last day to claim your promo. See the details: {u}"],
              ["Mag-e-expire na ang plan mo! I-renew ngayon para tuloy ang serbisyo: {u}"]),
    "social": (["Last call! Registration closes tonight. Reserve your slot: {u}", "Don't miss it! Only a few seats left for this {day}'s event: {u}"],
               ["Last call! Magsasara ang registration ngayong gabi. I-reserve ang slot mo: {u}"]),
}

_SHARE_TITLE = ["Thought you'd like this: {title}. {u}", "Read this: {title} - {u}", "Sharing this article, {title}: {u}",
                "Saw this and thought of you ({title}): {u}", "Interesting read on {title}: {u}", "Here's the page about {title}: {u}"]
_SHARE_TITLE_TL = ["Basahin mo 'to: {title}. {u}", "Nakita ko 'to, {title}: {u}", "Eto yung article tungkol sa {title}: {u}"]
_SHARE_SITE = ["Check out {site}: {u}", "Visit {site} for more details: {u}", "More info at {site}: {u}", "Hi! Here's the {site} link: {u}",
               "You can find it on {site}: {u}", "Saw this on {site}: {u}"]
_SHARE_SITE_TL = ["Tingnan mo ang {site}: {u}", "Para sa detalye, bisitahin ang {site}: {u}", "Nakita ko 'to sa {site}: {u}"]

_DAYS = ["Saturday", "Sunday", "Friday", "weekend"]
_DATES = ["June 12", "July 3", "Aug 21", "Sept 9", "Oct 15", "Nov 28", "Dec 5", "Jan 17"]


def _slots(rng: np.random.Generator) -> dict:
    return {"amt": f"{int(rng.integers(2, 60)) * 50:,}", "n": str(int(rng.integers(2, 40)) * 25), "hrs": str(int(rng.choice([6, 12, 24, 48]))),
            "code": f"{rng.choice(list('ABCDEFGHJKLMNPQRSTUVWXYZ'))}{rng.choice(list('ABCDEFGHJKLMNPQRSTUVWXYZ'))}{int(rng.integers(1000, 99999))}",
            "date": str(rng.choice(_DATES)), "day": str(rng.choice(_DAYS)), "gift": str(rng.choice(_GIFTS)),
            "pct": str(int(rng.choice([5, 10, 20, 50, 100, 150, 200]))), "gb": str(int(rng.choice([3, 5, 10, 20, 30, 50])))}


def _fill(template: str, rng: np.random.Generator, brand: str | None, link: str, **extra) -> str:
    values = {**_slots(rng), "u": link, "b": brand_name(brand) if brand else "", **extra}
    return template.format(**values)


def _pick(rng: np.random.Generator, options: list[str]) -> str:
    return str(options[int(rng.integers(len(options)))])


def _lang(rng: np.random.Generator) -> bool:
    return bool(rng.random() < TAGLISH_RATE)


def _wrap(rng: np.random.Generator, core: str, prefixes: list[str], suffixes: list[str]) -> str:
    return (_pick(rng, prefixes) + core + _pick(rng, suffixes)).replace("  ", " ")


def _tidy(text: str) -> str:
    """Link last in the core, then an optional sentence: keep spacing tidy ('url. text' stays valid)."""
    return " ".join(text.split())


def smish(rng: np.random.Generator, link: str, theme: str, brand: str | None = None) -> str:
    """A scam in the words of `brand` (if given) about `theme`. Unknown themes fall back to 'account'."""
    en_b, tl_b, en_n, tl_n = _SMISH.get(theme, _SMISH["account"])
    taglish = _lang(rng)
    pool = (tl_b if taglish else en_b) if brand else (tl_n if taglish else en_n)
    core = _fill(_pick(rng, pool), rng, brand, link)
    suffixes = _SUFFIX_SMISH_TL if taglish else _SUFFIX_SMISH_EN
    if re.search(r"hours|oras", core):                                      # a deadline is already in the sentence: do not state a second one
        suffixes = [x for x in suffixes if "{hrs}" not in x]
    suffix = _fill(_pick(rng, suffixes), rng, brand, link)                   # after the link, never before it
    prefix = "" if brand else _pick(rng, _PREFIX_SMISH)                      # a brand's message opens with its own name
    return _tidy(prefix + core + suffix)


def has_smish(theme: str) -> bool:
    return theme in _SMISH


def neutral(rng: np.random.Generator, link: str) -> str:
    pool = _NEUTRAL_TL if _lang(rng) else _NEUTRAL
    return _tidy(_fill(_pick(rng, pool), rng, None, link))


def scam_benign(rng: np.random.Generator, link: str, brand: str) -> str | None:
    """Scam wording that points at a real, clean `brand` page; None if there is no wording for that brand."""
    if brand not in _SCAM_BENIGN:
        return None
    pool = _SCAM_BENIGN_TL.get(brand) if brand in _SCAM_BENIGN_TL and _lang(rng) else _SCAM_BENIGN[brand]
    return _tidy(_fill(_pick(rng, pool), rng, brand, link))


def has_legit_brand(theme: str) -> bool:
    return theme in _LEGIT_BRAND


def legit_brand(rng: np.random.Generator, link: str, brand: str, theme: str) -> str:
    en, tl = _LEGIT_BRAND[theme]
    pool = tl if _lang(rng) else en
    core = _fill(_pick(rng, pool), rng, brand, link)
    suffix = _pick(rng, _SUFFIX_LEGIT_TL if pool is tl else _SUFFIX_LEGIT_EN)
    return _tidy(core + suffix)


def legit_security(rng: np.random.Generator, link: str, brand: str) -> str:
    en, tl = _LEGIT_SECURITY
    return _tidy(_fill(_pick(rng, tl if _lang(rng) else en), rng, brand, link))


def has_legit_theme(theme: str) -> bool:
    return theme in _LEGIT_THEME


def legit_theme(rng: np.random.Generator, link: str, theme: str) -> str:
    en, tl = _LEGIT_THEME[theme]
    return _tidy(_fill(_pick(rng, tl if _lang(rng) else en), rng, None, link))


def legit_share(rng: np.random.Generator, link: str, title: str = "", site: str = "") -> str | None:
    """A person sharing the page: names its title if the path has one, else its site; None if the link has neither."""
    if title:
        pool = _SHARE_TITLE_TL if _lang(rng) else _SHARE_TITLE
        return _tidy(_fill(_pick(rng, pool), rng, None, link, title=title))
    if site:
        pool = _SHARE_SITE_TL if _lang(rng) else _SHARE_SITE
        return _tidy(_fill(_pick(rng, pool), rng, None, link, site=site))
    return None


def legit_urgent(rng: np.random.Generator, link: str, brand: str) -> str:
    """A genuine notice in an urgent voice (an overdue bill, a new-device alert) for the brand's own link."""
    en, tl = _LEGIT_URGENT
    return _tidy(_fill(_pick(rng, tl if _lang(rng) else en), rng, brand, link))


def has_legit_pressure(theme: str) -> bool:
    return theme in _LEGIT_PRESSURE


def legit_pressure(rng: np.random.Generator, link: str, theme: str) -> str:
    """Genuine but high-pressure wording (a sale ending tonight, a filing deadline) for a clean page on that theme."""
    en, tl = _LEGIT_PRESSURE[theme]
    return _tidy(_fill(_pick(rng, tl if _lang(rng) else en), rng, None, link))


def smish_nolink(rng: np.random.Generator) -> str:
    """A scam with no link: a number to call or text. The number is the masked token <NUM>."""
    en, tl = _SMISH_NOLINK
    taglish = _lang(rng)
    core = _fill(_pick(rng, tl if taglish else en), rng, None, "", name=str(rng.choice(["Mr. Reyes", "Ms. Santos", "Atty. Cruz", "Agent Lim"])))
    prefix = "" if re.match(r"(?i)(hi|hello|good day)", core) else _pick(rng, ["", "", "", "Hi, ", "Hello! ", "URGENT: ", "Good day, ", "NOTICE: "])
    suffix = _pick(rng, ["", "", "", " Limited time only.", " Offer valid today only.", " Reply STOP to unsubscribe."] if not taglish
                   else ["", "", "", " Limitado lang ito.", " Valid ngayong araw lamang."])
    return _tidy(prefix + core + suffix)
