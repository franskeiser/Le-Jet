# How to run the Le-Jet demo from a cold start

This is the whole procedure, in order. Total time from "everything off" to "first result": about **5 minutes** of waiting.

**What the pieces are (so the steps make sense)**

| Piece | What it does | Where it runs |
|---|---|---|
| **Gateway VM** (`lejet-gw`) | The only door to the internet. It sends the lab's traffic through the VPN and blocks everything else. | VirtualBox |
| **Analysis VM** (`lejet-an`) | The isolated room where a link is really opened. It fetches the page, takes a screenshot and sends back data. | VirtualBox |
| **API** | Receives the message from the web page, asks the lab to open the link, then runs the model. It never opens a link itself. | Your PC, port **5000** |
| **Web page** | The Le-Jet screen you paste the message into. | Your PC, port **5050** |

> Port 5173 on your PC is a *different* app. Le-Jet uses **5050**.

---

## 1. Start the lab (order matters: gateway first)

Open **PowerShell** (a normal one is fine) and run:

```powershell
cd D:\Le-Jet
& "C:\Program Files\Oracle\VirtualBox\VBoxManage.exe" startvm lejet-gw --type headless
```

Wait about **1 minute**, then:

```powershell
& "C:\Program Files\Oracle\VirtualBox\VBoxManage.exe" startvm lejet-an --type headless
```

Wait about **2 minutes**. The VPN on the gateway and the safety self-test on the analysis VM start by themselves.

**Load your SSH key** (the lab runs only while the key is loaded; you may be asked for the key passphrase):

```powershell
ssh-add $HOME\.ssh\lejet_ed25519
```

If it says the agent is not running or is disabled (you locked it at the end of a previous day), run this once in an **administrator** PowerShell, then try `ssh-add` again:

```powershell
Set-Service ssh-agent -StartupType Manual; Start-Service ssh-agent
```

**Check that the lab is ready** (should print `token-age ...` with a small number, and `1` or more for the worker):

```powershell
ssh lejet-an 'if [ -f /run/lejet-preflight-ok ]; then echo token-age $(( $(date +%s) - $(cat /run/lejet-preflight-ok) ))s; else echo no-token-yet; fi; echo worker-ready-lines $(journalctl -b 0 -u lejet-live.service --no-pager | grep -c ready)'
```

- `no-token-yet`: the safety self-test has not finished. Wait 1 minute and run it again.
- `worker-ready-lines 0`: the fast worker is still starting. You can already use the demo; it just uses the slower path for the first link.

---

## 2. Start the API (terminal 1)

```powershell
cd D:\Le-Jet
.\.venv\Scripts\python.exe -m src.serving.api
```

It uses the live-trained model (`models\fusion\rf_vec_live`) by itself when that folder is there, and prints `Model folder: ...` so you can see which one it loaded (`/health` shows it too). To force another one, add `--fusion-dir models\fusion\rf_vec`. It then prints `Le-Jet API at http://127.0.0.1:5000 ...` and stays quiet. **Leave this window open.**

Check it (in another PowerShell window):

```powershell
Invoke-RestMethod http://127.0.0.1:5000/health | ConvertTo-Json
```

You want `"live_lab": { "ready": true, ... }`. If `ready` is `false`, read the `note` next to it (VMs not running, or the SSH key is not loaded).

## 3. Start the web page (terminal 2)

```powershell
cd D:\Le-Jet\frontend
npm run dev
```

Open **http://localhost:5050** in your browser. **Leave this window open too.**

---

## 4. Use it

1. Paste an SMS that contains a link into the box.
2. **Wait a moment (about 1.5 seconds).** A line appears: *"Link found: the lab is already inspecting it."* The lab has started opening the link while you read.
3. Click **CHECK NOW**. You see the lab's real steps with their real times:
   *check the link → send it into the lab → isolation gate → open the page in the VM → screenshot → bring the evidence back → page rank.*
4. The result shows the verdict, the reasons (the words and facts that pushed it), where the link really went, the **screenshot** the lab took, domain age, and so on. **Tingnan ang buong detalye** opens the research view.

**How long it takes**

| Situation | Time |
|---|---|
| Normal link, nothing else running | about 40 seconds |
| You paste, talk for a minute, then click Check | almost no waiting |
| A safety self-test is running when the link arrives (about 1 link in 6) | up to 45 seconds more |
| Right after a cold start (the fast worker is still starting) | up to 70 seconds |

**Things to try**

- A **legitimate** link, for example `https://www.python.org/`.
- A **smishing** link from the team's list. The list in the documents is *defanged* (`hxxps://` and `[.]`). To use one, change `hxxps` to `https` and `[.]` to `.` in your SMS text. Never open it in your own browser.
- A **dead** link (a domain that does not exist). It still gives a verdict from the text and the link's spelling, and the page says "page not loaded" with the reason, and shows no screenshot.
- A message **with no link**: judged on the text alone.

**What the page tells you honestly**

- *"The lab could not open this link just now, so the verdict uses the page it saw earlier"* → the lab failed, but the lab had seen this link before.
- *"The isolated lab is not running, so the link was not opened..."* → the VMs are off or the SSH key is not loaded.

---

## 5. If something goes wrong

| What you see | What to do |
|---|---|
| Page says the lab is not running | Check both VMs are running (`& "C:\Program Files\Oracle\VirtualBox\VBoxManage.exe" list runningvms` shows two) and that `ssh-add -l` lists your key. |
| `/health` says `ready: false` | Same as above. Wait 1 more minute after starting the VMs. |
| A link seems stuck for more than 3 minutes | The VM may be frozen. Restart it: `& "C:\Program Files\Oracle\VirtualBox\VBoxManage.exe" controlvm lejet-an reset`, wait about 2 minutes, then try again. |
| Many jobs waiting / strange extra steps | Close other browser tabs that show the Le-Jet page. Each open tab can start its own lab job. |
| `npm` or `node` not found | `$env:Path = "C:\Program Files\nodejs;" + $env:Path` then run `npm run dev` again. |
| Port 5000 or 5050 already in use | An old API or page is still running. Close its window, or run the "stop everything" commands in section 6. Starting the API a second time now stops with "port 5000 is already in use"; it no longer shares the port with the first one. |
| The page loads but nothing happens when you click | Check terminal 1 shows the API is still running. |

---

## 6. Stop everything (when you are done practising)

Press **Ctrl+C** in terminal 1 (API) and terminal 2 (web page). Then shut the lab down (analysis first, then gateway):

```powershell
ssh lejet-an "sudo -n /usr/bin/systemctl poweroff"
```

Wait about 20 seconds, then:

```powershell
& "C:\Program Files\Oracle\VirtualBox\VBoxManage.exe" controlvm lejet-gw acpipowerbutton
```

Check that both are off (this prints nothing when they are):

```powershell
& "C:\Program Files\Oracle\VirtualBox\VBoxManage.exe" list runningvms
```

**If you closed a window by accident, or Ctrl+C does not stop it**, stop the API and the web page from any PowerShell:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'src\.serving\.api|vite\.js' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

Check that ports 5000 and 5050 are free (it prints nothing when they are):

```powershell
Get-NetTCPConnection -LocalPort 5000,5050 -State Listen -ErrorAction SilentlyContinue
```

This does not touch the other app on port 5173. Also close any browser tab that still shows the Le-Jet page, so it cannot start more lab jobs.

**At the end of the day, lock the SSH key** (administrator PowerShell):

```powershell
ssh-add -D; Stop-Service ssh-agent; Set-Service ssh-agent -StartupType Disabled
```

---

## 7. Safety rules (always)

- **Never open a scanned link in your own browser.** Only the analysis VM opens links.
- Never paste the VPN config, the account number, the SSH key, the key passphrase or the `ops` password into any file, chat or message.
- Do not use any other network setup for the VMs (no shared folders, no clipboard sharing, no USB).
- The lab VMs treat everything coming out of the analysis VM as untrusted data; the strict validators on your PC decide what is accepted. Do not skip them.

## 8. Where things are

| What | Where |
|---|---|
| How the live inspection and the fast worker work | `docs/live_set_and_serving.md` |
| The dataset the demo model was trained and tested on | `data/live/` (not in git) and the same document |
| The VM bundle for re-deploying the lab (needs the `ops` password) | `D:\LeJetData\bundle_new\lejet-bundle.tar.gz`, deploy with `scripts\deploy_to_vm.ps1 -Bundle <that file>` |
| Lab results, screenshots, quarantine | `D:\LeJetData\` (outside git and OneDrive) |
