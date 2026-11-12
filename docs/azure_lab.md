# The Azure screenshot lab (for the Le-Jet tool)

What it is: the Azure VM `lejet-vm` also opens the links people paste into the **tool** (the frontend). It takes a picture of
the **whole page** (top to bottom) and of up to **4 more pages of the same site**, like Phish-Blitz does, and sends them back to
your PC. It is only for the live tool. The dataset was collected with the local VirtualBox lab and is not touched.

`lejet-vm` ALSO runs the SMS collection website. The two are kept apart (see "Safety").

## 1. Log in to the VM from your PC (once)

1. Azure portal -> `lejet-vm` -> **Start** (the status must say *Running*).
2. Azure portal -> `lejet-vm` -> *Networking* -> *Network settings*: inbound port **22** must be allowed. Best: *Source = My IP
   address*. If your home internet changes IP, edit that rule or SSH will time out.
3. Use your own key. Make it once with `ssh-keygen -t ed25519 -f $HOME\.ssh\lejet_azure`, then add its public half to the VM:
   portal -> `lejet-vm` -> Help -> **Reset password** -> Mode *Add SSH public key*, Username `azureuser`, and paste the line
   printed by `Get-Content $HOME\.ssh\lejet_azure.pub`. (Azure's own generated `.pem`, if you still have it, works too.)
4. Add this to `C:\Users\<you>\.ssh\config` (create the file if missing). The user name is `azureuser`:
   ```
   Host lejet-azure
     HostName lejet-x7k2q.japaneast.cloudapp.azure.com
     User azureuser
     IdentityFile ~/.ssh/lejet_azure
     IdentitiesOnly yes
     BatchMode yes
     StrictHostKeyChecking accept-new
     ServerAliveInterval 30
   ```
5. Test: `ssh lejet-azure whoami` must print the user name with **no password question**.

## 2. Install the lab on the VM (once; about 5-10 minutes)

```
powershell -File scripts\deploy_to_azure.ps1
```
This copies the code, installs Chromium and Python packages, creates the locked-down user `fetcher`, loads the firewall rule and
starts the safety check. It does not restart or change the website; its last step compares the website's state before and after
and warns if anything differs. After you change Python code later: `scripts\deploy_to_azure.ps1 -CodeOnly`.

Check: `ssh lejet-azure "sudo -n /usr/local/sbin/lejet-azure-gate"` must end with `all checks passed: token written`.

## 3. Use it

Nothing to do: when `~/.ssh/config` has the `lejet-azure` host, the tool uses the Azure lab automatically
(`src/serving/live_inspect.py`). Force one or the other with an environment variable before starting the API:
`LEJET_LAB=azure` or `LEJET_LAB=local` (the two VirtualBox VMs). The evidence page shows the main picture and the other pages.

## 4. "Open all the smishing sites"

Some sites will not open on the first try. The lab tries, in order, until one gives a real page:
1. the normal visit as an Android phone (Chromium, 20 s);
2. the same link as a desktop browser with longer waits (45 s);
3. the other scheme (`http` <-> `https`);
4. with or without `www.`.

It stops early if the name does not exist, and never spends more than about 2 minutes on the retries. A page that is **really
down** (taken down by the host, or the name was deleted) cannot be opened by anyone. For those the result says `dead` and why
(`nxdomain`, `refused`, `reset`, `timeout`, `tls`, `blocked`, `other`).

**Archived copies (optional, off by default).** Set `LEJET_ARCHIVE_FALLBACK=1` before starting the API and a dead page also gets a
look in the Wayback Machine and, if `URLSCAN_API_KEY` is set in that window, in urlscan.io. The answer is always labelled
*ARCHIVED copy, not the live page*. This sends the link's address to those two websites, which is why it is off by default.

**Honest limits.** The VM is in Japan on a datacenter address. Some smishing pages only show to phones in the Philippines and
show nothing (or a blank page) to anything else; those still count as `dead` or as a blank picture. A Philippine proxy/VPN for
Chromium would be the next step if that is common in your links.

## 5. Philippine exit (Mullvad VPN, only for the lab)

Many smishing pages answer a visitor from Japan with "service restricted in this region" (seen on a real casino link). Your
VirtualBox lab used Mullvad with a Philippine exit; the Azure lab can do the same, for the **lab user only**:

1. In your Mullvad account make a **new** WireGuard config for this VM: *WireGuard configuration* -> platform *Linux* -> generate a
   new key -> pick the same country/server as your lab config -> turn *content blocking* off -> download. Save it as
   `D:\LeJetData\wg-azure.conf`. (Do NOT reuse `wg0.conf`: one key on two machines at once breaks both tunnels.)
2. `powershell -File scripts\deploy_to_azure.ps1 -VpnConfig D:\LeJetData\wg-azure.conf`
   It prints the exit IP, country and city that Mullvad sees. Once the country is the one you want, lock it in with
   `... -VpnConfig D:\LeJetData\wg-azure.conf -ExpectCountry Philippines` (use the exact name it printed).
3. Go back to Japan any time: `scripts\deploy_to_azure.ps1 -RemoveVpn`.

How it stays safe: only `fetcher`'s traffic is routed into the tunnel (policy routing by user id). SSH, Caddy and the website use
the normal connection. A firewall kill-switch lets `fetcher` out ONLY through the tunnel, so if it drops, the lab has no network
instead of leaking the VM's Japan address. The key is stored root-only on the VM (the config is rebuilt from validated fields; extra
lines such as PostUp are ignored). The safety gate checks every 4 minutes that the tunnel has a recent handshake and that Mullvad
reports the traffic as coming from a Mullvad exit (and from the expected country, if set); if not, no link is opened. Names the
browser visits are looked up through the tunnel too.

## Safety (why phishing pages are not dangerous to the website)

- The page-opening program runs as user `fetcher`: no login, no sudo, cannot read `/var/lib/lejetweb` (the dataset CSV) and
  `lejet-azure-live` forbids that folder to the unit.
- Chromium keeps its sandbox on; every page goes through the guard proxy (public addresses, ports 80/443 only).
- Its own firewall table (`infra/azure/egress.nft`) refuses the website's ports, SSH, Azure's metadata service and every private
  address for `fetcher`.
- A hard memory cap of 1.5 GB per inspection, plus 2 GB swap, so the website's OCR is never starved.
- `infra/azure/gate.sh` re-checks all of this every 4 minutes. If any check fails the token is deleted and **no link is opened**.
- Everything that comes back is validated on your PC (`validate_output.py`, `validate_shots.py`) before the tool uses it.
- Risk that remains: a browser exploit would run on a machine that also holds the dataset. If that worries you, create a second
  small VM, run the same `deploy_to_azure.ps1` against it (`LEJET_LAB_HOST` alias), and keep the website on `lejet-vm`.

## Cost

A running B2als_v2 uses student credit even when idle. You can **Stop (deallocate)** the VM when you are not demonstrating;
this also stops the website, so only do it when nobody needs to submit messages.
